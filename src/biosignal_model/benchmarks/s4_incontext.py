"""reference_s4_incontext: the S4 reference (benchmarks/s4.py) with an in-context head over
retrieved training records (lead's design, 2026-10-29). Each query crop is classified with
the K most similar training records (by the cosine of their embeddings) and their labels as
context, never a record of the query's own patient.

1. Encoder: the stack of S4Model (Conv1d encoder, n_layers S4 blocks with their LayerNorms and
   channel dropout) without its decoder; the embedding is the mean over time of the last block.
2. Memory (no gradient): before epoch 1 (the initial weights) and after each training epoch, in
   eval mode, the embedding of every training record (folds 1-8) = the mean of its crop
   embeddings over the evaluation crops (crop_starts with valid_stride_fraction), L2-normalized,
   with the record's label vector and patient id.
3. Retrieval (no gradient): for the L2-normalized query embedding, the context_k memory records
   with the largest cosine similarity, every record of the query's own patient left out (for
   validation and test queries too; their patients are not in folds 1-8).
4. Tokens (width d_model): context i = W_e e_i + W_y y_i + s_i v_sim + t_ctx; query = W_e q +
   t_query (q the query's un-normalized embedding, with gradient). W_e is shared, W_e and W_y
   are linear maps without bias (t_ctx and t_query are the offsets); context_labels "zeros"
   gives y_i = 0 (the control: the same capacity without the neighbours' labels).
5. In-context head: icl_layers pre-norm transformer encoder layers (icl_heads, icl_feed_forward,
   GELU, the recipe's dropout) over the set [query, context_1..K], no positional encoding;
   LayerNorm and Linear(d, classes) on the query token.
6. Auxiliary head Linear(d, classes) on q. Loss = BCE(in-context logits) + aux_weight x BCE(aux
   logits). Validation, test and the choice of the epoch use the in-context logits only, over
   the crops as reference_s4 (validation_on, test_on).
7. Otherwise the recipe of reference_s4 (AdamW, constant lr, one random crop per record and
   epoch, the last partial batch dropped) and its stop rule (patience_epochs); the recipe parts
   of benchmarks/supervised.py other than the stop rule and save_chosen_to are not part of this
   design and are refused.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import model_validator
from torch import nn

from biosignal_model.benchmarks.s4 import S4Layer, S4Settings, _channel_dropout
from biosignal_model.benchmarks.xresnet1d import crop_starts

TOKEN_STD = 0.02  # initial standard deviation of v_sim, t_ctx and t_query (as ViT's class token)


class S4InContextSettings(S4Settings):
    """reference_s4_incontext: the recipe of reference_s4 (every field of S4Settings) and the
    in-context settings; none has a default."""

    context_k: int  # retrieved training records per query
    icl_layers: int  # transformer encoder layers of the in-context head
    icl_heads: int
    icl_feed_forward: int  # hidden width of their feed-forward part
    aux_weight: float  # of the auxiliary head's BCE
    context_labels: Literal["labels", "zeros"]  # "zeros": no label input (the control arm)

    @model_validator(mode="after")
    def _published_recipe(self) -> S4InContextSettings:
        if (
            self.loss != "bce" or self.ema_decay is not None or self.augmentation is not None
            or self.schedule != "constant" or self.distillation is not None
        ):  # fmt: skip
            raise ValueError(
                "reference_s4_incontext takes the published recipe: loss bce, no ema_decay, "
                "augmentation or distillation, schedule constant"
            )
        if self.context_k < 1:
            raise ValueError("context_k is at least 1")
        return self


class S4Encoder(nn.Module):
    """The stack of S4Model (benchmarks/s4.py) without its decoder: the mean over time of the
    last block's output, (B, leads, L) -> (B, d_model)."""

    def __init__(self, s: S4Settings, inputs: int, length: int) -> None:
        super().__init__()
        self.dropout = s.dropout
        self.encoder = nn.Conv1d(inputs, s.d_model, 1)
        self.layers = nn.ModuleList(S4Layer(s, length, 1.0) for _ in range(s.n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(s.d_model) for _ in range(s.n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            z = _channel_dropout(layer(x), self.dropout, self.training)
            x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)
        return x.mean(dim=-1)


@dataclass(frozen=True)
class Memory:
    """The training records as context: L2-normalized embeddings (N, d), label vectors (N, C)
    and patient codes (N,), all without gradient."""

    embeddings: torch.Tensor
    labels: torch.Tensor
    patients: torch.Tensor


@torch.no_grad()
def retrieve(
    query_unit: torch.Tensor, query_patients: torch.Tensor, memory: Memory, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The k memory records most similar to each L2-normalized query (cosine similarity), never
    one of the query's own patient: (indices (B, k), similarities (B, k))."""
    sim = query_unit @ memory.embeddings.T  # (B, N)
    own = query_patients[:, None] == memory.patients[None, :]
    sim = sim.masked_fill(own, float("-inf"))
    sims, indices = sim.topk(k, dim=1)
    return indices, sims


class S4InContextModel(nn.Module):
    """The encoder, the token maps, the in-context head and the auxiliary head (module
    docstring, points 1 and 4-6)."""

    def __init__(self, s: S4InContextSettings, inputs: int, classes: int, length: int) -> None:
        super().__init__()
        d = s.d_model
        self.k, self.context_labels = s.context_k, s.context_labels
        self.encoder = S4Encoder(s, inputs, length)
        self.embed = nn.Linear(d, d, bias=False)  # W_e, shared by query and context
        self.label_embed = nn.Linear(classes, d, bias=False)  # W_y
        self.v_sim = nn.Parameter(torch.randn(d) * TOKEN_STD)
        self.t_ctx = nn.Parameter(torch.randn(d) * TOKEN_STD)
        self.t_query = nn.Parameter(torch.randn(d) * TOKEN_STD)
        # One layer each (not nn.TransformerEncoder, whose deep copies start every layer with
        # the same weights).
        self.icl = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d, s.icl_heads, s.icl_feed_forward, s.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )  # fmt: skip
            for _ in range(s.icl_layers)
        )
        self.icl_norm = nn.LayerNorm(d)
        self.icl_head = nn.Linear(d, classes)
        self.aux_head = nn.Linear(d, classes)

    def label_input(self, labels: torch.Tensor) -> torch.Tensor:
        """y_i of the context tokens: the labels, or zeros for context_labels "zeros"."""
        return labels if self.context_labels == "labels" else torch.zeros_like(labels)

    def heads(
        self, q_raw: torch.Tensor, patients: torch.Tensor, memory: Memory
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(in-context logits, auxiliary logits), each (B, classes), from query embeddings."""
        with torch.no_grad():
            indices, sims = retrieve(F.normalize(q_raw.detach(), dim=-1), patients, memory, self.k)
            e = memory.embeddings[indices]  # (B, K, d)
            y = self.label_input(memory.labels[indices])  # (B, K, C)
        context = (
            self.embed(e) + self.label_embed(y) + sims.unsqueeze(-1) * self.v_sim + self.t_ctx
        )
        query = self.embed(q_raw) + self.t_query
        h = torch.cat([query.unsqueeze(1), context], dim=1)  # the set, no positions
        for layer in self.icl:
            h = layer(h)
        icl: torch.Tensor = self.icl_head(self.icl_norm(h[:, 0]))
        aux: torch.Tensor = self.aux_head(q_raw)
        return icl, aux

    def forward(
        self, x: torch.Tensor, patients: torch.Tensor, memory: Memory
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (B, leads, L) -> 2 x (B, classes)
        return self.heads(self.encoder(x), patients, memory)


@torch.no_grad()
def build_memory(
    encoder: nn.Module, data: torch.Tensor, ids: np.ndarray, labels: torch.Tensor,
    patients: torch.Tensor, starts: list[int], size: int, batch_size: int,
) -> Memory:  # fmt: skip
    """The memory of the records `ids` (module docstring, point 2); the encoder is in eval mode
    while it runs and back in its former mode after."""
    was_training = encoder.training
    encoder.eval()
    out = []
    for k in range(0, len(ids), batch_size):
        batch = data[torch.from_numpy(ids[k : k + batch_size]).to(data.device)]
        crops = torch.stack([batch[:, :, a : a + size] for a in starts], 1)
        e = encoder(crops.flatten(0, 1)).view(len(batch), len(starts), -1).mean(1)
        out.append(F.normalize(e, dim=-1))
    encoder.train(was_training)
    index = torch.from_numpy(ids).to(data.device)
    return Memory(torch.cat(out), labels[index], patients[index])


def train_and_score(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, patients: np.ndarray,
    train: tuple[int, ...], validation: tuple[int, ...], test: tuple[int, ...], rate: float,
    s: S4InContextSettings, seed: int, device: torch.device,
    log: Callable[[dict[str, float]], None], prefix: str,
    macro_auc: Callable[[np.ndarray, np.ndarray], tuple[float, int]],
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """benchmarks/s4.py train_and_score with the in-context model; `patients`: one integer code
    per record (the same code for the records of one patient). Logged per epoch as
    reference_s4, and epoch_validation_macro_auc_aux (the auxiliary head) and memory_refresh_s
    (the refresh after the epoch). Returned: the scores and the chosen epoch's in-context
    predictions on the validation and test folds (as scored: validation_on, test_on)."""
    random.seed(seed)
    torch.manual_seed(seed)
    tr, va, te = (np.flatnonzero(np.isin(folds, f)) for f in (train, validation, test))
    _, counts = np.unique(patients[tr], return_counts=True)
    if s.context_k > len(tr) - int(counts.max()):
        raise ValueError(f"context_k {s.context_k} is more than the records of other patients")
    data = torch.from_numpy(x.astype(np.float32)).to(device)  # normalize False: as recorded
    target = torch.from_numpy(y.astype(np.float32)).to(device)
    patient = torch.from_numpy(patients.astype(np.int64)).to(device)
    size = int(round(s.input_size_s * rate))
    model = S4InContextModel(s, x.shape[1], y.shape[1], size).to(device)
    log({f"{prefix}/parameters": float(sum(p.numel() for p in model.parameters())),
         f"{prefix}/memory_records": float(len(tr))})  # fmt: skip
    optimizer = torch.optim.AdamW(model.parameters(), lr=s.learning_rate, betas=s.betas,
                                  eps=s.eps, weight_decay=s.weight_decay)  # fmt: skip
    per_epoch = len(tr) // s.batch_size  # drop_last
    length = x.shape[2]
    starts = crop_starts(length, size, int(round(size * s.valid_stride_fraction)))

    def refresh() -> tuple[Memory, float]:
        started = time.monotonic()
        memory = build_memory(model.encoder, data, tr, target, patient, starts, size,
                              s.batch_size)  # fmt: skip
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return memory, time.monotonic() - started

    def predict(memory: Memory, ids: np.ndarray, on: str) -> tuple[np.ndarray, np.ndarray]:
        """(in-context, auxiliary) predictions per record, over its crops."""
        model.eval()
        icl_out, aux_out = [], []
        with torch.no_grad():
            for k in range(0, len(ids), s.batch_size):
                index = torch.from_numpy(ids[k : k + s.batch_size]).to(device)
                batch = data[index]
                crops = torch.stack([batch[:, :, a : a + size] for a in starts], 1)
                who = patient[index].repeat_interleave(len(starts))  # crops in record order
                icl, aux = model(crops.flatten(0, 1), who, memory)
                for p, out in ((icl, icl_out), (aux, aux_out)):
                    p = p.view(len(batch), len(starts), -1)
                    out.append((torch.sigmoid(p) if on == "probabilities" else p).mean(1))
        return torch.cat(icl_out).cpu().numpy(), torch.cat(aux_out).cpu().numpy()

    memory, _ = refresh()  # from the initial weights
    best_val, best_test, best_epoch, labels_n = -1.0, 0.0, 0, 0
    best_p_val, best_p_test = np.zeros((0, y.shape[1])), np.zeros((0, y.shape[1]))
    for epoch in range(1, s.epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr)).numpy()]
        losses = []
        for k in range(per_epoch):
            ids = order[k * s.batch_size : (k + 1) * s.batch_size]
            first = [0 if length == size else random.randint(0, length - size - 1) for _ in ids]  # noqa: S311
            batch = torch.stack([data[i, :, a : a + size] for i, a in zip(ids, first, strict=True)])
            index = torch.from_numpy(ids).to(device)
            goal = target[index]
            icl, aux = model(batch, patient[index], memory)
            loss = F.binary_cross_entropy_with_logits(icl, goal)
            loss = loss + s.aux_weight * F.binary_cross_entropy_with_logits(aux, goal)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(loss.item())
        memory, refresh_s = refresh()
        p_val, p_val_aux = predict(memory, va, s.validation_on)
        auc_val, _ = macro_auc(y[va], p_val)
        record = {f"{prefix}/epoch": float(epoch),
                  f"{prefix}/epoch_train_loss": float(np.mean(losses)),
                  f"{prefix}/epoch_validation_macro_auc": auc_val,
                  f"{prefix}/epoch_validation_macro_auc_aux": macro_auc(y[va], p_val_aux)[0],
                  f"{prefix}/memory_refresh_s": refresh_s}  # fmt: skip
        if auc_val > best_val:  # as reference_s4: the best fold-9 macro-AUC
            best_val, best_epoch = auc_val, epoch
            best_p_val, best_p_test = p_val, predict(memory, te, s.test_on)[0]
            best_test, labels_n = macro_auc(y[te], best_p_test)
            record[f"{prefix}/chosen_epoch"] = float(epoch)
            record[f"{prefix}/chosen_test_macro_auc"] = best_test
            if s.save_chosen_to is not None:
                torch.save(model.state_dict(), s.save_chosen_to)
        log(record)
        if s.patience_epochs is not None and epoch - best_epoch >= s.patience_epochs:
            log({f"{prefix}/stopped_epoch": float(epoch)})
            break
    return {"macro_auc": best_test, "labels": float(labels_n), "epoch": float(best_epoch),
            "validation_macro_auc": best_val}, (best_p_val, best_p_test)  # fmt: skip

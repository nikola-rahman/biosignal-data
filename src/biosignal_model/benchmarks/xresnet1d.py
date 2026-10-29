"""The xresnet1d101 baseline of the PTB-XL benchmark (Strodthoff et al. 2021, arXiv 2004.13701),
ported from its code without fastai, to check our benchmark pipeline: trained through our
records, labels, folds and macro-AUC, it should score what the paper reports (Table II, 100 Hz,
"all": 0.925 +- 0.008). Sources, github.com/helme/ecg_ptbxl_benchmarking (master, read
2026-10-29), each choice below names its file:

- architecture: code/models/xresnet1d.py (XResNet1d, ResBlock, ConvLayer, xresnet1d101 = expansion
  4, layers [3, 4, 23, 3]) and code/models/basic_conv1d.py (create_head1d, weight_init);
  fastai_model.py passes kernel_size=5, ps_head=0.5, lin_ftrs_head=[128];
- data: code/utils/utils.py preprocess_signals (one StandardScaler over every sample of every
  lead of the training folds); code/experiments/scp_experiment.py (folds 1-8 train, 9
  validation, 10 test; 100 Hz records);
- crops: code/models/fastai_model.py (input_size 2.5 s; training: a random crop of each record
  per epoch; prediction: crops of 2.5 s every 1.25 s, aggregate_fn "max") and
  code/models/timeseries_utils.py TimeseriesDatasetCrops (random start randint(0, T - n - 1));
- training: fastai_model.py fit (learn.model.apply(weight_init); fit_one_cycle(50, 1e-2);
  bs 128; wd 1e-2; binary cross-entropy on logits; no early stopping: the last epoch is
  scored); fastai v1 defaults: Adam betas (0.9, 0.99) with beta1 following the momentum
  schedule (0.95 -> 0.85 -> 0.95), decoupled weight decay on every weight (true_wd, bn_wd),
  one-cycle with pct_start 0.3, div_factor 25, final_div 25e4, cosine annealing, the train
  loader dropping its last partial batch.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torch import nn


class ReferenceSettings(BaseModel):
    """The recipe of the reference model; none has a default (see the module docstring)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_size_s: float  # 2.5
    valid_stride_fraction: float  # 0.5: prediction crops every input_size / 2
    aggregate: str  # "max" over a record's crops (their aggregate_fn default)
    batch_size: int  # 128
    epochs: int  # 50
    max_lr: float  # 1e-2
    weight_decay: float  # 1e-2, decoupled, on every weight
    pct_start: float  # 0.3
    div_factor: float  # 25
    final_div: float  # 25e4 (div_factor * 1e4)
    moms: tuple[float, float]  # (0.95, 0.85)
    beta2: float  # 0.99
    eps: float  # 1e-8 (torch Adam)
    kernel_size: int  # 5
    ps_head: float  # 0.5
    lin_ftrs_head: list[int]  # [128]


def _conv_layer(ni: int, nf: int, ks: int, stride: int = 1, act: bool = True) -> nn.Sequential:
    """ConvLayer of xresnet1d.py with norm Batch and bn_1st: conv (no bias), batch norm, ReLU."""
    layers: list[nn.Module] = [
        nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=(ks - 1) // 2, bias=False),
        nn.BatchNorm1d(nf),
    ]
    if act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    """ResBlock of xresnet1d.py, expansion 4 (bottleneck), pool AvgPool first on the identity."""

    def __init__(self, ni: int, nf: int, stride: int, kernel_size: int) -> None:
        super().__init__()
        nh = nf
        ni, nf = ni * 4, nf * 4
        self.convs = nn.Sequential(
            _conv_layer(ni, nh, 1),
            _conv_layer(nh, nh, kernel_size, stride=stride),
            _conv_layer(nh, nf, 1, act=False),
        )
        idpath: list[nn.Module] = []
        if ni != nf:
            idpath.append(_conv_layer(ni, nf, 1, act=False))
        if stride != 1:
            idpath.insert(0, nn.AvgPool1d(2, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.act(self.convs(x) + self.idpath(x))
        return out


class _ConcatPool(nn.Module):
    """AdaptiveConcatPool1d of basic_conv1d.py: max and mean over time, concatenated."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.amax(dim=-1), x.mean(dim=-1)], dim=1)


def xresnet1d101(
    input_channels: int, num_classes: int, kernel_size: int, ps_head: float,
    lin_ftrs_head: list[int],
) -> nn.Sequential:  # fmt: skip
    """XResNet1d(ResBlock, 4, [3, 4, 23, 3]) with create_head1d, then weight_init."""
    stem_szs = [input_channels, 32, 32, 64]
    stem = [_conv_layer(stem_szs[i], stem_szs[i + 1], 5, stride=2 if i == 0 else 1)
            for i in range(3)]  # fmt: skip
    block_szs = [64 // 4, 64, 64, 64, 64]
    blocks = []
    for i, n in enumerate([3, 4, 23, 3]):
        stride = 1 if i == 0 else 2
        blocks.append(nn.Sequential(*[
            _ResBlock(block_szs[i] if k == 0 else block_szs[i + 1], block_szs[i + 1],
                      stride if k == 0 else 1, kernel_size)
            for k in range(n)
        ]))  # fmt: skip
    # create_head1d: lin_ftrs [2 * nf, *lin_ftrs_head, nc]; dropout ps_head / 2 before the inner
    # layers and ps_head before the last; ReLU after the inner layers; batch norm before each
    nf = block_szs[-1] * 4
    lin = [2 * nf, *lin_ftrs_head, num_classes]
    ps = [ps_head / 2] * (len(lin) - 2) + [ps_head]
    head: list[nn.Module] = [_ConcatPool()]
    for k, (n_in, n_out, p) in enumerate(zip(lin[:-1], lin[1:], ps, strict=True)):
        head += [nn.BatchNorm1d(n_in)] + ([nn.Dropout(p)] if p else []) + [nn.Linear(n_in, n_out)]
        if k < len(lin) - 2:
            head.append(nn.ReLU(inplace=True))
    model = nn.Sequential(*stem, nn.MaxPool1d(kernel_size=3, stride=2, padding=1), *blocks,
                          nn.Sequential(*head))  # fmt: skip
    model.apply(_weight_init)
    return model


def _weight_init(m: nn.Module) -> None:
    """weight_init of basic_conv1d.py (applied in fit, after the model's own init)."""
    if isinstance(m, nn.Conv1d | nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def _annealing_cos(start: float, end: float, pct: float) -> float:
    return end + (start - end) / 2 * (math.cos(math.pi * pct) + 1)


def one_cycle(step: int, total: int, s: ReferenceSettings) -> tuple[float, float]:
    """(learning rate, beta1) at `step` of `total`: fastai v1 OneCycleScheduler."""
    a1 = int(total * s.pct_start)
    low, high, end = s.max_lr / s.div_factor, s.max_lr, s.max_lr / s.final_div
    if step < a1:
        pct = step / a1
        return _annealing_cos(low, high, pct), _annealing_cos(s.moms[0], s.moms[1], pct)
    pct = (step - a1) / (total - a1)
    return _annealing_cos(high, end, pct), _annealing_cos(s.moms[1], s.moms[0], pct)


def crop_starts(length: int, size: int, stride: int) -> list[int]:
    """TimeseriesDatasetCrops with chunk_length = min_chunk_length = size: the starts of the
    whole chunks (a chunk shorter than `size` and every later one are dropped)."""
    return [k for k in range(0, length, stride) if min(k + size, length) - k >= size]


def train_and_score(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, train: tuple[int, ...],
    validation: tuple[int, ...], test: tuple[int, ...], rate: float, s: ReferenceSettings,
    seed: int, device: torch.device, log: Callable[[dict[str, float]], None], prefix: str,
    macro_auc: Callable[[np.ndarray, np.ndarray], tuple[float, int]],
) -> dict[str, float]:  # fmt: skip
    """Train on the training folds with the recipe; the last epoch is scored on the test folds
    (the fold-9 macro-AUC of each epoch is logged, not used). `x`: [records, leads, samples]."""
    random.seed(seed)
    torch.manual_seed(seed)
    tr, va, te = (np.flatnonzero(np.isin(folds, f)) for f in (train, validation, test))
    mean, std = float(x[tr].astype(np.float64).mean()), float(x[tr].astype(np.float64).std())
    data = torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device)
    target = torch.from_numpy(y.astype(np.float32)).to(device)
    size = int(s.input_size_s * rate)
    model = xresnet1d101(x.shape[1], y.shape[1], s.kernel_size, s.ps_head, s.lin_ftrs_head)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=s.max_lr / s.div_factor,
                                  betas=(s.moms[0], s.beta2), eps=s.eps,
                                  weight_decay=s.weight_decay)  # fmt: skip
    per_epoch = len(tr) // s.batch_size  # drop_last
    total, step = per_epoch * s.epochs, 0
    length = x.shape[2]

    def predict(ids: np.ndarray) -> np.ndarray:
        model.eval()
        starts = crop_starts(length, size, int(size * s.valid_stride_fraction))
        out = []
        with torch.no_grad():
            for k in range(0, len(ids), s.batch_size):
                batch = data[torch.from_numpy(ids[k : k + s.batch_size]).to(device)]
                crops = torch.stack([batch[:, :, a : a + size] for a in starts], 1)
                p = torch.sigmoid(model(crops.flatten(0, 1))).view(len(batch), len(starts), -1)
                out.append((p.amax(1) if s.aggregate == "max" else p.mean(1)).cpu().numpy())
        return np.concatenate(out)

    for epoch in range(1, s.epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr)).numpy()]
        losses = []
        for k in range(per_epoch):
            ids = order[k * s.batch_size : (k + 1) * s.batch_size]
            starts = [0 if length == size else random.randint(0, length - size - 1) for _ in ids]
            batch = torch.stack(
                [data[i, :, a : a + size] for i, a in zip(ids, starts, strict=True)]
            )
            lr, beta1 = one_cycle(step, total, s)
            for group in optimizer.param_groups:
                group["lr"], group["betas"] = lr, (beta1, s.beta2)
            loss = F.binary_cross_entropy_with_logits(model(batch), target[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            step += 1
            losses.append(loss.item())
        auc_val, _ = macro_auc(y[va], predict(va))
        log({f"{prefix}/epoch": float(epoch), f"{prefix}/epoch_train_loss": float(np.mean(losses)),
             f"{prefix}/epoch_validation_macro_auc": auc_val, f"{prefix}/lr": lr})  # fmt: skip
    auc, labels_n = macro_auc(y[te], predict(te))
    return {"macro_auc": auc, "labels": float(labels_n), "epoch": float(s.epochs),
            "validation_macro_auc": auc_val}  # fmt: skip

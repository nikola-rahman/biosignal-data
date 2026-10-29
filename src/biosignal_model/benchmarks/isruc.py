"""Benchmark: sleep staging on ISRUC-Sleep cohort I (subgroup I), 5 classes, by subject.

The protocol of CBraMod (Wang et al. 2025, arXiv 2412.07236, ICLR 2025), as its code
does it (github.com/wjq-learning/CBraMod, checked on 2026-10-29: preprocessing/ISRUC/
prepare_ISRUC_1.py, datasets/isruc_dataset.py, models/model_for_isruc.py,
finetune_trainer.py, finetune_evaluator.py), which REVE (arXiv 2510.21585) and CSBrain
follow:

- the 100 subjects of cohort I, one night each (`<n>/<n>.rec`); subjects 1-80 train, 81-90
  validation, 91-100 test (isruc_dataset.split_dataset: the first 80, the next 10, the
  last 10 of subjects 1..100);
- the stages of the first expert (`<n>_1.txt`, one line per 30 s epoch), mapped
  {0: W, 1: N1, 2: N2, 3: N3, 5: REM} (label2id; any other value is refused here, as it
  would stop their script);
- each night cut into sequences of 20 consecutive epochs, the last epochs that do not fill
  a sequence left out, in every split (the scored epochs are the same as theirs);
- each epoch through the encoder, then a sequence model over the 20 epoch features (their
  head: Linear to 512, GELU, one Transformer layer of width 512, 4 heads, 2048 feed-forward,
  pre-norm, GELU; a linear classifier per epoch);
- cross-entropy with label smoothing; the epoch with the best Cohen's kappa on the
  validation subjects is scored once on the test subjects;
- metrics over all test epochs: balanced accuracy, Cohen's kappa, weighted F1 (sklearn's
  definitions).

What differs, by design: CBraMod reads 6 EEG channels (F3-A2, C3-A2, O1-A2, F4-A1, C4-A1,
O2-A1: columns 2-7 of the .rec file), filters them (0.3-35 Hz band-pass, 50 Hz notch) and
divides by 100. Here the model gets the stored samples in physical units, at their native
rates, unfiltered: `channels: eeg6` takes the same 6 EEG channels, `channels: all` every
channel of the recording (EEG, EOG, chin and leg EMG, ECG, respiration, SpO2, at 200, 25
and 12.5 Hz): the comparison of one signal type with all of them together.

Methods, as in the PTB-XL benchmark:

- `fine_tuning`: the pretrained encoder and the sequence head, every weight trained, the
  encoder at its learning rate, the head at its own (CBraMod: backbone 1e-4, the rest
  1e-3 * sqrt(batch / 256)), AdamW, cosine decay per step to a minimum, gradient clipping;
- `from_scratch`: the same from random weights, every weight at the head's rate;
- `linear_probing` / `linear_probing_random`: the encoder frozen (pretrained, or random
  weights), one softmax regression on each epoch's mean feature (no sequence context), its
  L2 chosen on the validation subjects by kappa: a diagnostic.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict

from biosignal_data.read.standard import StandardDataset
from biosignal_data.storage.stores import Store
from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.batch import Batch, Packer
from biosignal_model.benchmarks import LOG_EVERY_STEPS, Log
from biosignal_model.benchmarks.ptbxl import _mean_features
from biosignal_model.model import Encoder, ModelConfig

DATASET = "isr-uc/isruc-sleep"
SUBJECTS = tuple(range(1, 101))
TRAIN, VALIDATION, TEST = SUBJECTS[:80], SUBJECTS[80:90], SUBJECTS[90:]
STAGES = {"0": 0, "1": 1, "2": 2, "3": 3, "5": 4}  # CBraMod's label2id
STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
EPOCH_S = 30.0
EEG6 = ("F3-A2", "C3-A2", "O1-A2", "F4-A1", "C4-A1", "O2-A1")


class IsrucSettings(BaseModel):
    """Every setting of the benchmark; none has a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channels: Literal["eeg6", "all"]
    sequence_epochs: int  # 30 s epochs in one sequence (CBraMod: 20)
    rows: int  # rows of the model's max_tokens in one forward pass, at most
    read_threads: int  # requests to the store at the same time
    workers: int  # data loader processes that cut and pack batches; 0: the main process
    linear_probing: bool  # also linear probing of the frozen encoder (pretrained and random)
    linear_probing_l2: list[float]
    linear_probing_steps: int
    methods: list[Literal["fine_tuning", "from_scratch"]]
    fine_tuning_epochs: int  # the best epoch on the validation subjects is the one scored
    fine_tuning_patience: int | None  # stop after this many epochs without a better kappa
    fine_tuning_encoder_learning_rate: float  # from_scratch: every weight at the head's rate
    fine_tuning_head_learning_rate: float
    fine_tuning_min_learning_rate: float  # the end of the cosine decay
    fine_tuning_weight_decay: float
    fine_tuning_sequences_per_step: int
    fine_tuning_label_smoothing: float
    fine_tuning_grad_clip: float
    head_width: int
    head_layers: int
    head_heads: int
    head_feed_forward: int
    head_dropout: float
    seed: int


@dataclass
class Night:
    """One subject's night: the chosen channels as stored, and the stage of each epoch."""

    subject: int
    record: str
    channels: list[Channel]  # in the order of `samples`
    samples: list[np.ndarray]  # stored values of each channel, whole night
    stages: np.ndarray  # class of each 30 s epoch
    length: int  # epochs in one sequence

    @property
    def sequences(self) -> int:
        """Whole sequences of the night; the epochs after the last are not used."""
        return len(self.stages) // self.length

    def epoch(self, k: int) -> Example:
        """Epoch k in physical units, every channel at its own rate."""
        signals = []
        for c, stored in zip(self.channels, self.samples, strict=True):
            n = round(EPOCH_S * c.sampling_rate)
            raw = stored[k * n : (k + 1) * n].astype(np.float64)
            values = raw * (1.0 if c.scale is None else c.scale) + (c.offset or 0.0)
            signals.append(Signal(values.astype(np.float32), c.sampling_rate, 0.0))
        origin = Origin(DATASET, self.record, f"{DATASET}:{self.subject}", k * EPOCH_S,
                        tuple(self.channels))  # fmt: skip
        return Example(tuple(signals), origin)


def stages(text: str, record: str) -> np.ndarray:
    """The classes of a hypnogram file (one stage per line, blank lines skipped)."""
    out = []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        if value not in STAGES:
            raise ValueError(f"{DATASET}: {record} has stage {value!r}, not one of {list(STAGES)}")
        out.append(STAGES[value])
    return np.array(out, dtype=np.int64)


def _signal_type(name: str) -> str:
    """For analysis only (the model gets no types). The X and DC channels are left `other`:
    the dataset's pages do not say which is which."""
    if name in ("LOC-A2", "ROC-A1"):
        return "eog"
    if name == "SaO2":
        return "spo2"
    return "eeg" if name[:2] in ("F3", "C3", "O1", "F4", "C4", "O2") else "other"


def load(store: Store, s: IsrucSettings, log: Log) -> list[Night]:
    """The 100 nights of cohort I: the chosen channels as stored, and the stages."""
    dataset = StandardDataset.open(store, DATASET, "latest")
    by_record: dict[str, list[dict[str, Any]]] = {}
    for c in dataset.channels.to_pylist():
        by_record.setdefault(c["recording_id"], []).append(c)
    started = time.monotonic()
    nights = []
    for n in SUBJECTS:
        record = f"{n}/{n}"
        rows = sorted(by_record[record], key=lambda c: (c["array"], c["index"]))
        if s.channels == "eeg6":
            named = {c["name"]: c for c in rows}
            missing = [name for name in EEG6 if name not in named]
            if missing:
                raise ValueError(f"{DATASET}: {record} has no channel {missing}")
            rows = [named[name] for name in EEG6]
        arrays = sorted({c["array"] for c in rows})
        stored = dataset.read_many([(record, a) for a in arrays], threads=s.read_threads)
        channels = [
            Channel(c["array"], c["index"], float(c["sampling_rate"]), c["scale"], c["offset"],
                    c["invalid_value"], c["name"], _signal_type(c["name"]), c["unit"])
            for c in rows
        ]  # fmt: skip
        samples = [stored[record, c.array][c.index] for c in channels]
        labels = stages(dataset.file(f"{n}/{n}_1.txt").decode("ascii"), record)
        epochs = min(len(x) / (EPOCH_S * c.sampling_rate) for x, c in zip(samples, channels))
        if len(labels) != math.floor(epochs):
            raise ValueError(
                f"{DATASET}: {record} has {len(labels)} stages and {epochs:g} epochs of signal"
            )
        nights.append(Night(n, record, channels, samples, labels, s.sequence_epochs))
    log({"load/nights": float(len(nights)), "load/seconds": time.monotonic() - started})
    return nights


def _packer(model: ModelConfig, max_tokens: int, rows: int) -> Packer:
    return Packer(
        datasets=[DATASET], max_tokens=max_tokens, samples_per_token=model.samples_per_token,
        max_channels=model.max_channels, rows=rows, mask_ratio=0.0, seed=0,
    )  # fmt: skip


def _items(nights: list[Night], subjects: Iterable[int]) -> list[tuple[int, int]]:
    """(night index, sequence index) of every whole sequence of the given subjects."""
    wanted = set(subjects)
    return [(i, j) for i, n in enumerate(nights) if n.subject in wanted for j in range(n.sequences)]


def _per_pass(nights: list[Night], packer: Packer, s: IsrucSettings) -> int:
    """Sequences that fit one forward pass (`rows` rows of max_tokens)."""
    tokens = max(len(packer.tokens(n.epoch(0)).slot) for n in nights)
    per_row = packer.max_tokens // tokens
    if per_row == 0:
        raise ValueError(f"one epoch is {tokens} tokens, more than max_tokens {packer.max_tokens}")
    rows_per_sequence = math.ceil(s.sequence_epochs / per_row)
    if rows_per_sequence > s.rows:
        raise ValueError(f"a sequence needs {rows_per_sequence} rows; `rows` is {s.rows}")
    return s.rows // rows_per_sequence


class _Sequences(torch.utils.data.Dataset[tuple[Batch, list[tuple[int, int]], torch.Tensor]]):
    """Batch k packs the epochs of the sequences `groups[k]`, in order: (batch, where each
    epoch is, the stages [sequences, sequence_epochs]); the slots are seeded by (seed, k)."""

    def __init__(
        self, nights: list[Night], groups: list[list[tuple[int, int]]], packer: Packer,
        length: int, seed: str,
    ) -> None:  # fmt: skip
        self.nights, self.groups, self.packer, self.length, self.seed = (
            nights, groups, packer, length, seed
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, k: int) -> tuple[Batch, list[tuple[int, int]], torch.Tensor]:
        rng = random.Random(f"{self.seed}/{k}")  # noqa: S311 - not secret
        self.packer.reseed(rng.getrandbits(63))
        examples, labels = [], []
        for i, j in self.groups[k]:
            night = self.nights[i]
            first = j * self.length
            examples += [night.epoch(e) for e in range(first, first + self.length)]
            labels.append(night.stages[first : first + self.length])
        batch, where = self.packer.pack(examples, trim=True)
        return batch, where, torch.from_numpy(np.stack(labels))


def _as_is(item: Any) -> Any:
    return item


def _loader(
    nights: list[Night], groups: list[list[tuple[int, int]]], packer: Packer, s: IsrucSettings,
    seed: str,
) -> Iterable[tuple[Batch, list[tuple[int, int]], torch.Tensor]]:  # fmt: skip
    return torch.utils.data.DataLoader(
        _Sequences(nights, groups, packer, s.sequence_epochs, seed), batch_size=None,
        shuffle=False, num_workers=s.workers, collate_fn=_as_is,
        prefetch_factor=4 if s.workers else None,
    )  # fmt: skip


class SequenceHead(torch.nn.Module):
    """CBraMod's ISRUC head on the features of the epochs of a sequence: a projection with
    GELU, a Transformer encoder over the sequence, a linear classifier per epoch."""

    def __init__(self, width: int, s: IsrucSettings) -> None:
        super().__init__()
        self.project = torch.nn.Linear(width, s.head_width)
        layer = torch.nn.TransformerEncoderLayer(
            s.head_width, s.head_heads, s.head_feed_forward, s.head_dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )  # fmt: skip
        self.sequence = torch.nn.TransformerEncoder(
            layer, s.head_layers, enable_nested_tensor=False
        )
        self.classify = torch.nn.Linear(s.head_width, len(STAGE_NAMES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # [n, L, width] -> [n, L, 5]
        out: torch.Tensor = self.classify(self.sequence(F.gelu(self.project(features))))
        return out


def scores(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Balanced accuracy, Cohen's kappa and weighted F1, as sklearn computes them."""
    k = len(STAGE_NAMES)
    cm = np.zeros((k, k), dtype=np.float64)
    np.add.at(cm, (truth, predicted), 1)
    support, picked, n = cm.sum(1), cm.sum(0), cm.sum()
    present = support > 0
    recall = np.divide(np.diag(cm), support, out=np.zeros(k), where=present)
    precision = np.divide(np.diag(cm), picked, out=np.zeros(k), where=picked > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(k),
                   where=precision + recall > 0)  # fmt: skip
    observed, expected = np.trace(cm) / n, float((support * picked).sum()) / n**2
    return {
        "balanced_accuracy": float(recall[present].mean()),
        "kappa": float((observed - expected) / (1 - expected)) if expected < 1 else 0.0,
        "weighted_f1": float((f1 * support).sum() / n),
    }


def end_to_end(
    config: ModelConfig, nights: list[Night], max_tokens: int, s: IsrucSettings,
    device: torch.device, start: dict[str, torch.Tensor] | None, log: Log, prefix: str,
) -> dict[str, float]:  # fmt: skip
    """The encoder and the sequence head, every weight trained on the training subjects;
    the epoch with the best kappa on the validation subjects is scored on the test
    subjects. `start`: pretrained weights, or None (from scratch)."""
    torch.manual_seed(s.seed)
    model = Encoder(config).to(device)
    if start is not None:
        model.load_state_dict(start)
    head = SequenceHead(config.width, s).to(device)
    encoder_lr = s.fine_tuning_encoder_learning_rate if start is not None else (
        s.fine_tuning_head_learning_rate
    )
    used = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    optimizer = torch.optim.AdamW(
        [{"params": used, "lr": encoder_lr},
         {"params": list(head.parameters()), "lr": s.fine_tuning_head_learning_rate}],
        weight_decay=s.fine_tuning_weight_decay,
    )  # fmt: skip
    packer = _packer(config, max_tokens, s.rows)
    per_pass = _per_pass(nights, packer, s)
    train = _items(nights, TRAIN)
    steps_per_epoch = math.ceil(len(train) / s.fine_tuning_sequences_per_step)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=s.fine_tuning_epochs * steps_per_epoch,
        eta_min=s.fine_tuning_min_learning_rate,
    )
    rng = random.Random(s.seed)  # noqa: S311 - not secret
    best: dict[str, float] = {"validation_kappa": -2.0}
    best_epoch, step = 0, 0
    for epoch in range(1, s.fine_tuning_epochs + 1):
        model.train()
        head.train()
        rng.shuffle(train)
        started, losses, window = time.monotonic(), [], []
        # A step: fine_tuning_sequences_per_step sequences, in passes of `per_pass`; one
        # data loader for the epoch's passes, the gradients summed over a step's passes.
        n = s.fine_tuning_sequences_per_step
        passes, last = [], []  # last: whether a pass ends its step, and the step's epochs
        for k in range(0, len(train), n):
            step_items = train[k : k + n]
            for a in range(0, len(step_items), per_pass):
                passes.append(step_items[a : a + per_pass])
                ends = a + per_pass >= len(step_items)
                last.append((ends, len(step_items) * s.sequence_epochs))
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        batches = _loader(nights, passes, packer, s, f"{s.seed}/{epoch}")
        for (batch, where, y), (ends, total) in zip(batches, last, strict=True):
            features = _mean_features(model, batch, where, device)
            logits = head(features.view(len(y), s.sequence_epochs, -1))
            loss = F.cross_entropy(
                logits.reshape(-1, len(STAGE_NAMES)), y.to(device).reshape(-1),
                label_smoothing=s.fine_tuning_label_smoothing, reduction="sum",
            ) / total  # fmt: skip
            loss.backward()  # type: ignore[no-untyped-call]
            loss_sum += loss.item()
            if not ends:
                continue
            torch.nn.utils.clip_grad_norm_(used + list(head.parameters()), s.fine_tuning_grad_clip)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window.append(loss_sum)
            loss_sum = 0.0
            if step % LOG_EVERY_STEPS == 0:
                log({f"{prefix}/train_loss": float(np.mean(window)), f"{prefix}/step": float(step)})
                losses += window
                window = []
        losses += window
        seconds = time.monotonic() - started
        val = scores(*_predict(model, head, nights, VALIDATION, packer, per_pass, s, device))
        record = {
            f"{prefix}/epoch": float(epoch), f"{prefix}/epoch_train_loss": float(np.mean(losses)),
            f"{prefix}/epoch_seconds": seconds,
        }  # fmt: skip
        record |= {f"{prefix}/epoch_validation_{k}": v for k, v in val.items()}
        if val["kappa"] > best["validation_kappa"]:  # the test subjects for the chosen epoch
            best_epoch = epoch
            test = scores(*_predict(model, head, nights, TEST, packer, per_pass, s, device))
            best = {f"validation_{k}": v for k, v in val.items()} | test
            record |= {f"{prefix}/chosen_epoch": float(epoch)}
            record |= {f"{prefix}/chosen_test_{k}": v for k, v in test.items()}
        log(record)
        if s.fine_tuning_patience is not None and epoch - best_epoch >= s.fine_tuning_patience:
            break
    return best | {"epoch": float(best_epoch), "epochs_run": float(epoch)}


@torch.no_grad()
def _predict(
    model: Encoder, head: SequenceHead, nights: list[Night], subjects: Iterable[int],
    packer: Packer, per_pass: int, s: IsrucSettings, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(true stages, predicted stages) of every epoch of the whole sequences of `subjects`."""
    model.eval()
    head.eval()
    items = _items(nights, subjects)
    groups = [items[a : a + per_pass] for a in range(0, len(items), per_pass)]
    truth, predicted = [], []
    for batch, where, y in _loader(nights, groups, packer, s, f"{s.seed}/predict"):
        features = _mean_features(model, batch, where, device)
        logits = head(features.view(len(y), s.sequence_epochs, -1))
        truth.append(y.reshape(-1).numpy())
        predicted.append(logits.argmax(-1).reshape(-1).cpu().numpy())
    return np.concatenate(truth), np.concatenate(predicted)


@torch.no_grad()
def embed(
    model: Encoder, nights: list[Night], packer: Packer, per_pass: int, s: IsrucSettings,
    device: torch.device, log: Log, name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """(mean feature, stage, subject) of every epoch of the whole sequences of every night."""
    model.eval()
    items = _items(nights, SUBJECTS)
    groups = [items[a : a + per_pass] for a in range(0, len(items), per_pass)]
    features, labels, subjects, started = [], [], [], time.monotonic()
    for k, (batch, where, y) in enumerate(_loader(nights, groups, packer, s, f"{s.seed}/embed"), 1):
        features.append(_mean_features(model, batch, where, device).cpu())
        labels.append(y.reshape(-1).numpy())
        subjects += [nights[i].subject for i, _ in groups[k - 1] for _ in range(s.sequence_epochs)]
        if k % max(1, len(groups) // 10) == 0 or k == len(groups):
            log({f"embed/{name}/batches": float(k), f"embed/{name}/of": float(len(groups)),
                 f"embed/{name}/seconds": time.monotonic() - started})  # fmt: skip
    return torch.cat(features).numpy(), np.concatenate(labels), np.array(subjects)


def _fit(x: torch.Tensor, y: torch.Tensor, l2: float, steps: int) -> torch.Tensor:
    """Weights [d + 1, classes] of a softmax regression (bias not penalized)."""
    w = torch.zeros(x.shape[1] + 1, len(STAGE_NAMES), dtype=torch.float64, device=x.device,
                    requires_grad=True)  # fmt: skip
    optimizer = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(x @ w[:-1] + w[-1], y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return w.detach()


def linear_probing(
    features: np.ndarray, labels: np.ndarray, subjects: np.ndarray, s: IsrucSettings, log: Log,
    prefix: str, device: torch.device,
) -> dict[str, float]:  # fmt: skip
    """L2 chosen on the validation subjects by kappa, then one fit on the training subjects,
    scored on the test subjects. Each L2 is logged with its validation kappa."""
    train, val, test = (np.isin(subjects, g) for g in (TRAIN, VALIDATION, TEST))
    mean, std = features[train].mean(0), features[train].std(0) + 1e-8
    x = torch.from_numpy((features - mean) / std).double().to(device)
    y = torch.from_numpy(labels).to(device)
    best_l2, best_kappa = s.linear_probing_l2[0], -2.0
    for l2 in s.linear_probing_l2:
        w = _fit(x[train], y[train], l2, s.linear_probing_steps)
        kappa = scores(labels[val], (x[val] @ w[:-1] + w[-1]).argmax(-1).cpu().numpy())["kappa"]
        log({f"{prefix}/l2": l2, f"{prefix}/validation_kappa": kappa})
        if kappa > best_kappa:
            best_l2, best_kappa = l2, kappa
    w = _fit(x[train], y[train], best_l2, s.linear_probing_steps)
    test_scores = scores(labels[test], (x[test] @ w[:-1] + w[-1]).argmax(-1).cpu().numpy())
    return test_scores | {"l2": best_l2, "validation_kappa": best_kappa}


def run(
    store: Store, checkpoint: dict[str, Any], settings: IsrucSettings, device: torch.device,
    log: Log,
) -> dict[str, float]:  # fmt: skip
    """All scores of one checkpoint, as a flat dict for logging: `<method>/<name>`."""
    from biosignal_model.train import TrainConfig  # noqa: PLC0415 - avoids a cycle

    train_config = TrainConfig.from_checkpoint(checkpoint["config"])
    config = train_config.model_settings()
    max_tokens = train_config.tokens.max_tokens
    nights = load(store, settings, log)
    packer = _packer(config, max_tokens, settings.rows)
    per_pass = _per_pass(nights, packer, settings)
    test_epochs = sum(n.sequences for n in nights if n.subject in TEST) * settings.sequence_epochs
    results: dict[str, float] = {
        "nights": float(len(nights)), "channels": float(len(nights[0].channels)),
        "test_epochs": float(test_epochs),
    }  # fmt: skip
    if settings.linear_probing:
        pretrained = Encoder(config).to(device)
        pretrained.load_state_dict(checkpoint["model"])
        torch.manual_seed(settings.seed)
        untrained = Encoder(config).to(device)
        for name, model in (("linear_probing", pretrained), ("linear_probing_random", untrained)):
            features, labels, subjects = embed(
                model, nights, packer, per_pass, settings, device, log, name
            )
            scored = linear_probing(features, labels, subjects, settings, log, name, device)
            results |= {f"{name}/{k}": v for k, v in scored.items()}
            log({f"{name}/kappa": scored["kappa"]})
    starts = {"fine_tuning": checkpoint["model"], "from_scratch": None}
    for name in settings.methods:
        scored = end_to_end(config, nights, max_tokens, settings, device, starts[name], log, name)
        results |= {f"{name}/{k}": v for k, v in scored.items()}
        log({f"{name}/kappa": scored["kappa"]})
    return results

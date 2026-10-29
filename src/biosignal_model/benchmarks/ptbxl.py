"""Benchmark: ECG statements on PTB-XL, with the protocol of Strodthoff et al. 2021.

"Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL" (arXiv 2004.13701),
and its code (github.com/helme/ecg_ptbxl_benchmarking, utils.compute_label_aggregations
and select_data): for each task the labels of a record are the statements of its
`scp_codes` (every key, whatever its likelihood) that belong to the task, from
`scp_statements.csv`:

- `all`: every statement; `diagnostic`: the diagnostic ones; `subdiagnostic` and
  `superdiagnostic`: their `diagnostic_subclass` and `diagnostic_class`; `form` and
  `rhythm`: the statements of those groups.

A record without a label of the task is left out of it. Folds (`strat_fold`, no patient in
two folds): 1-8 train, 9 validation, 10 test. Metric: macro-AUC on fold 10, over the labels
with both classes there. The paper's main tables use the 100 Hz records (`rate: 100`);
`rate: 500` uses the original 500 Hz records, a different input from the published one.

For one checkpoint and each task: `fine_tuning` (the pretrained encoder and a linear head,
every weight trained on folds 1-8 with layer-dependent learning rates, the epoch chosen on
fold 9; the recipe of arXiv 2509.25095 when its settings are used), `from_scratch` (the same
from random weights, one learning rate), and `linear_probing` / `linear_probing_random` (a
logistic regression per label on the mean feature of the frozen encoder, pretrained or from
random weights, its L2 chosen on fold 9). Folds 9 and 10 are held out of pretraining
(index.yaml).
"""

from __future__ import annotations

import ast
import csv
import io
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from biosignal_data.read.standard import StandardDataset
from biosignal_data.storage.stores import Store
from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.batch import Batch, Packer
from biosignal_model.benchmarks import LOG_EVERY_STEPS, Log
from biosignal_model.model import Encoder, ModelConfig

DATASET = "physionet/ptb-xl"
TASKS = ("all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm")
TRAIN, VALIDATION, TEST = (1, 2, 3, 4, 5, 6, 7, 8), (9,), (10,)


class PtbxlSettings(BaseModel):
    """Every setting of the benchmark; none has a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate: Literal[100, 500]
    tasks: list[Literal["all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm"]]
    rows: int  # rows of the model's max_tokens in one batch, at most
    read_threads: int  # requests to the store at the same time
    workers: int  # data loader processes that crop and pack batches; 0: the main process
    linear_probing: bool  # also linear probing of the frozen encoder (pretrained and random)
    # fine_tuning: the pretrained encoder and a head; from_scratch: the same from random weights
    methods: list[Literal["fine_tuning", "from_scratch"]]
    linear_probing_l2: list[float]
    linear_probing_steps: int
    fine_tuning_epochs: int  # the best epoch on fold 9 is the one scored
    # Stop after this many epochs without a better fold-9 macro-AUC (None: all epochs). The
    # epochs after a stop cannot be chosen; on "all" the chosen epoch was 15-19 of 100 and
    # the AUC fell after it (v1 and v2, 2026-10-28/29; 25 asked for by the user).
    fine_tuning_patience: int | None
    fine_tuning_learning_rate: float  # of the head; the encoder's parts get lower ones
    fine_tuning_weight_decay: float
    fine_tuning_examples_per_step: int
    # Layer-dependent learning rates, as the ECG foundation model benchmark (arXiv 2509.25095,
    # section 3.3; github.com/AI4HealthUOL/ecg-fm-benchmarking, fm_ecg.py get_params): the
    # encoder's upper part trains at the head's rate x factor, its lower part (the input
    # embedding and the first blocks) at x factor^2.
    # Training from random weights uses one rate for all (their template_model.py).
    fine_tuning_discriminative_factor: float
    fine_tuning_lower_blocks_fraction: float  # of the blocks in the lower part (ECG-JEPA: 3/12)
    # Each training example is a random crop of this length, drawn again every epoch; a
    # validation or test record is scored as the mean probability of its non-overlapping
    # crops (their main_lite.py: input-size, stride-fraction-valtest 1, aggregate by mean).
    # None: the whole record.
    fine_tuning_crop_s: float | None
    seed: int


@dataclass
class Records:
    """The records of one rate: their examples, folds and statements, in one order."""

    ids: list[str]
    examples: list[Example]
    folds: np.ndarray
    statements: list[list[str]]  # the keys of scp_codes


def labels(
    records: Records, statements: list[dict[str, str]], task: str
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(multi-hot [n, classes], class names, mask of the records the task keeps), as the
    authors' compute_label_aggregations and select_data with min_samples 0."""
    by_code = {s[""]: s for s in statements}

    def is_(code: str, group: str) -> bool:
        return code in by_code and by_code[code].get(group) in ("1", "1.0")

    per_record: list[set[str]] = []
    for codes in records.statements:
        if task == "all":
            found = set(codes)
        elif task == "diagnostic":
            found = {c for c in codes if is_(c, "diagnostic")}
        elif task in ("subdiagnostic", "superdiagnostic"):
            column = "diagnostic_subclass" if task == "subdiagnostic" else "diagnostic_class"
            found = {
                by_code[c][column] for c in codes if is_(c, "diagnostic") and by_code[c][column]
            }
        else:  # form, rhythm
            found = {c for c in codes if is_(c, task)}
        per_record.append(found)
    classes = sorted(set().union(*per_record))
    keep = np.array([bool(f) for f in per_record])
    y = np.array([[c in f for c in classes] for f in per_record], dtype=np.float32)
    return y, classes, keep


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    """The mean ROC AUC over the labels with both classes in `y` (ties get average ranks),
    and how many labels that is."""
    aucs = []
    for k in range(y.shape[1]):
        positive = y[:, k] == 1
        n_pos, n_neg = int(positive.sum()), int((~positive).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(p[:, k])
        aucs.append((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs)), len(aucs)


def load(store: Store, settings: PtbxlSettings, log: Log) -> tuple[Records, list[dict[str, str]]]:
    """Every record of the chosen rate, read once into memory, and the statement table."""
    dataset = StandardDataset.open(store, DATASET, "latest")

    def table(name: str) -> list[dict[str, str]]:  # kept with the standard version
        data = dataset.file(name)
        return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))

    rows = table("ptbxl_database.csv")
    statements = table("scp_statements.csv")
    column = "filename_lr" if settings.rate == 100 else "filename_hr"
    array = f"{settings.rate}Hz"
    channels = dataset.channels.to_pylist()
    by_record: dict[str, list[dict[str, Any]]] = {}
    for c in channels:
        by_record.setdefault(c["recording_id"], []).append(c)

    started = time.monotonic()
    stored = dataset.read_many(
        [(row[column], array) for row in rows], threads=settings.read_threads, physical=True
    )
    log({"load/records": float(len(stored)), "load/seconds": time.monotonic() - started})

    def example(row: dict[str, str]) -> Example:
        record = row[column]
        mine = sorted(
            (c for c in by_record[record] if c["array"] == array), key=lambda c: c["index"]
        )
        values = stored[record, array]
        signals = tuple(
            Signal(values[c["index"]].astype(np.float32), float(settings.rate), 0.0) for c in mine
        )
        origin_channels = tuple(
            Channel(array, c["index"], float(settings.rate), c["scale"], c["offset"],
                    c["invalid_value"], c["name"], "ecg", c["unit"])
            for c in mine
        )  # fmt: skip
        return Example(
            signals, Origin(DATASET, record, f"{DATASET}:{row['patient_id']}", 0.0, origin_channels)
        )

    examples = [example(row) for row in rows]
    return Records(
        ids=[r[column] for r in rows],
        examples=examples,
        folds=np.array([int(r["strat_fold"]) for r in rows]),
        statements=[list(ast.literal_eval(r["scp_codes"])) for r in rows],
    ), statements


def _packer(model: ModelConfig, max_tokens: int, rows: int) -> Packer:
    return Packer(
        datasets=[DATASET], max_tokens=max_tokens, samples_per_token=model.samples_per_token,
        max_channels=model.max_channels, rows=rows, mask_ratio=0.0, seed=0,
    )  # fmt: skip


def _pooled(
    model: Encoder, examples: list[Example], packer: Packer, device: torch.device
) -> torch.Tensor:
    """The mean feature over all tokens of each example, [len(examples), width]."""
    batch, where = packer.pack(examples, trim=True)
    return _mean_features(model, batch, where, device)


def _mean_features(
    model: Encoder, batch: Batch, where: list[tuple[int, int]], device: torch.device
) -> torch.Tensor:
    """The mean feature over the tokens of each (row, document) in `where`, in one pass
    over the batch: [len(where), width]."""
    x = batch.inputs.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = model.features(x).float()
    rows, length, width = features.shape
    key = (torch.arange(rows, device=device)[:, None] * length + x.document).flatten()
    real = x.document.flatten() >= 0
    sums = torch.zeros(rows * length, width, device=device, dtype=features.dtype)
    sums.index_add_(0, key[real], features.reshape(-1, width)[real])
    counts = torch.bincount(key[real], minlength=rows * length)
    chosen = torch.tensor([r * length + d for r, d in where], device=device)
    return sums[chosen] / counts[chosen, None]


class _Packed(torch.utils.data.Dataset[tuple[Batch, list[tuple[int, int]]]]):
    """Batch k packs the examples `groups[k]`, each through `prepare` (e.g. a random crop):
    made in data loader workers, with random choices seeded by (seed, k), so a batch is the
    same whichever worker makes it."""

    def __init__(
        self, examples: list[Example], groups: list[list[int]],
        prepare: Callable[[Example, random.Random], Example], packer: Packer, seed: str,
    ) -> None:  # fmt: skip
        self.examples, self.groups, self.prepare = examples, groups, prepare
        self.packer, self.seed = packer, seed

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, k: int) -> tuple[Batch, list[tuple[int, int]]]:
        rng = random.Random(f"{self.seed}/{k}")  # noqa: S311 - not secret
        self.packer.reseed(rng.getrandbits(63))
        chosen = [self.prepare(self.examples[i], rng) for i in self.groups[k]]
        return self.packer.pack(chosen, trim=True)


def _as_is(item: Any) -> Any:
    return item


def _packed(
    examples: list[Example], groups: list[list[int]],
    prepare: Callable[[Example, random.Random], Example], packer: Packer, seed: str,
    workers: int,
) -> Iterable[tuple[Batch, list[tuple[int, int]]]]:  # fmt: skip
    """The batches of `groups`, in order, prepared ahead by `workers` processes."""
    return torch.utils.data.DataLoader(
        _Packed(examples, groups, prepare, packer, seed), batch_size=None, shuffle=False,
        num_workers=workers, collate_fn=_as_is, prefetch_factor=4 if workers else None,
    )  # fmt: skip


def _unchanged(example: Example, rng: random.Random) -> Example:
    return example


def _batches(n: int, packer: Packer, examples: list[Example], rows: int) -> list[list[int]]:
    """Indices in groups that fit one batch (every example of a rate has the same size)."""
    tokens = len(packer.tokens(examples[0]).slot)
    per_batch = max(1, (packer.max_tokens // tokens) * rows)
    return [list(range(k, min(n, k + per_batch))) for k in range(0, n, per_batch)]


@torch.no_grad()
def embed(
    model: Encoder, examples: list[Example], max_tokens: int, rows: int, device: torch.device,
    log: Log, name: str,
) -> np.ndarray:  # fmt: skip
    """The mean feature of every example; progress as `embed/<name>/...`."""
    model.eval()
    packer = _packer(model.config, max_tokens, rows)
    batches = _batches(len(examples), packer, examples, rows)
    parts, started = [], time.monotonic()
    for k, ids in enumerate(batches, 1):
        parts.append(_pooled(model, [examples[i] for i in ids], packer, device).cpu())
        if k % max(1, len(batches) // 10) == 0 or k == len(batches):
            log({f"embed/{name}/batches": float(k), f"embed/{name}/of": float(len(batches)),
                 f"embed/{name}/seconds": time.monotonic() - started})  # fmt: skip
    return torch.cat(parts).numpy()


def _fit(x: torch.Tensor, y: torch.Tensor, l2: float, steps: int) -> torch.Tensor:
    """Weights [d + 1, classes] of one logistic regression per label (bias not penalized)."""
    w = torch.zeros(x.shape[1] + 1, y.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ w[:-1] + w[-1]
        loss = F.binary_cross_entropy_with_logits(logits, y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return w.detach()


def linear_probing(
    features: np.ndarray, y: np.ndarray, folds: np.ndarray, s: PtbxlSettings, log: Log,
    prefix: str,
) -> dict[str, float]:  # fmt: skip
    """L2 chosen on fold 9 by macro-AUC, then one fit on folds 1-8, scored on fold 10. Each
    L2 tried is logged with its fold-9 macro-AUC, under `prefix`."""
    train, val, test = (np.isin(folds, f) for f in (TRAIN, VALIDATION, TEST))
    mean, std = features[train].mean(0), features[train].std(0) + 1e-8
    x = torch.from_numpy((features - mean) / std).double()
    yt = torch.from_numpy(y).double()
    best, best_auc = s.linear_probing_l2[0], -1.0
    for l2 in s.linear_probing_l2:
        w = _fit(x[train], yt[train], l2, s.linear_probing_steps)
        auc, _ = macro_auc(y[val], (x[val] @ w[:-1] + w[-1]).numpy())
        log({f"{prefix}/l2": l2, f"{prefix}/validation_macro_auc": auc})
        if auc > best_auc:
            best, best_auc = l2, auc
    w = _fit(x[train], yt[train], best, s.linear_probing_steps)
    auc, n = macro_auc(y[test], (x[test] @ w[:-1] + w[-1]).numpy())
    return {"macro_auc": auc, "labels": float(n), "l2": best, "validation_macro_auc": best_auc}


def end_to_end(
    config: ModelConfig, records: Records, y: np.ndarray, max_tokens: int, s: PtbxlSettings,
    device: torch.device, start: dict[str, torch.Tensor] | None, log: Log, prefix: str,
) -> dict[str, float]:  # fmt: skip
    """The encoder and a linear head on its mean feature, every weight trained on folds 1-8
    (layer-dependent learning rates and crops: see PtbxlSettings); the epoch with the best
    macro-AUC on fold 9 is scored on fold 10 (at most `fine_tuning_patience` epochs after it are
    run). `start`: pretrained weights, or None.
    Logged under `prefix`: the training loss every LOG_EVERY_STEPS steps, and per epoch its
    mean training loss, fold-9 macro-AUC, time and speed; fold 10 only for a chosen epoch."""
    torch.manual_seed(s.seed)
    model = Encoder(config).to(device)
    if start is not None:
        model.load_state_dict(start)
    head = torch.nn.Linear(config.width, y.shape[1]).to(device)
    factor = s.fine_tuning_discriminative_factor if start is not None else 1.0
    groups = parameter_groups(model, head, s.fine_tuning_learning_rate, factor,
                              s.fine_tuning_lower_blocks_fraction)  # fmt: skip
    optimizer = torch.optim.AdamW(groups, weight_decay=s.fine_tuning_weight_decay)
    packer = _packer(config, max_tokens, s.rows)
    train = [i for i in range(len(records.ids)) if records.folds[i] in TRAIN]
    rng = random.Random(s.seed)  # noqa: S311 - not secret

    def predict(indices: list[int]) -> np.ndarray:
        examples = [records.examples[i] for i in indices]
        return _predict(model, head, examples, s.fine_tuning_crop_s, packer, s.rows, device,
                        s.workers, f"{s.seed}/predict")  # fmt: skip

    def crop(example: Example, rng: random.Random) -> Example:
        return _random_crop(example, s.fine_tuning_crop_s, rng)

    val = [i for i in range(len(records.ids)) if records.folds[i] in VALIDATION]
    test = [i for i in range(len(records.ids)) if records.folds[i] in TEST]
    # A fixed part of folds 1-8, as large as fold 9, scored in eval mode each epoch: its
    # loss next to the fold-9 loss shows overfitting on one scale.
    rng_eval = random.Random(s.seed)  # noqa: S311 - not secret
    train_eval = sorted(rng_eval.sample(train, min(len(train), len(val))))
    best_val, best_test, best_epoch, labels_n = -1.0, 0.0, 0, 0
    step = 0
    for epoch in range(1, s.fine_tuning_epochs + 1):
        model.train()
        rng.shuffle(train)
        started, losses, window = time.monotonic(), [], []
        n = s.fine_tuning_examples_per_step
        steps = [train[k : k + n] for k in range(0, len(train), n)]
        batches = _packed(records.examples, steps, crop, packer, f"{s.seed}/{epoch}", s.workers)
        for ids, (batch, where) in zip(steps, batches, strict=True):
            pooled = _mean_features(model, batch, where, device)
            target = torch.from_numpy(y[ids]).to(device)
            loss = F.binary_cross_entropy_with_logits(head(pooled), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            step += 1
            window.append(loss.item())
            if step % LOG_EVERY_STEPS == 0:
                log({f"{prefix}/train_loss": float(np.mean(window)), f"{prefix}/step": float(step)})
                losses += window
                window = []
        losses += window
        seconds = time.monotonic() - started
        p_val = predict(val)
        auc_val, _ = macro_auc(y[val], p_val)
        record = {
            f"{prefix}/epoch": float(epoch),
            f"{prefix}/epoch_train_loss": float(np.mean(losses)),
            f"{prefix}/epoch_train_eval_loss": _bce(y[train_eval], predict(train_eval)),
            f"{prefix}/epoch_validation_loss": _bce(y[val], p_val),
            f"{prefix}/epoch_validation_macro_auc": auc_val,
            f"{prefix}/epoch_seconds": seconds,
            f"{prefix}/examples_per_s": len(train) / seconds,
        }
        if auc_val > best_val:  # fold 10 is scored for the epoch fold 9 chooses
            best_val, best_epoch = auc_val, epoch
            best_test, labels_n = macro_auc(y[test], predict(test))
            record[f"{prefix}/chosen_epoch"] = float(epoch)
            record[f"{prefix}/chosen_test_macro_auc"] = best_test
        log(record)
        if s.fine_tuning_patience is not None and epoch - best_epoch >= s.fine_tuning_patience:
            break
    return {"macro_auc": best_test, "labels": float(labels_n), "epoch": float(best_epoch),
            "validation_macro_auc": best_val, "epochs_run": float(epoch)}  # fmt: skip


@torch.no_grad()
def _predict(
    model: Encoder, head: torch.nn.Module, examples: list[Example], crop_s: float | None,
    packer: Packer, rows: int, device: torch.device, workers: int, seed: str,
) -> np.ndarray:  # fmt: skip
    """The probabilities of each example: the mean over its crops."""
    model.eval()
    crops, owner = [], []
    for n, example in enumerate(examples):
        for c in _crops(example, crop_s):
            crops.append(c)
            owner.append(n)
    out = []
    groups = _batches(len(crops), packer, crops, rows)
    for batch, where in _packed(crops, groups, _unchanged, packer, seed, workers):
        pooled = _mean_features(model, batch, where, device)
        out.append(torch.sigmoid(head(pooled)).float().cpu())
    p = torch.cat(out).numpy()
    sums = np.zeros((len(examples), p.shape[1]))
    np.add.at(sums, owner, p)
    return sums / np.bincount(owner, minlength=len(examples))[:, None]


def parameter_groups(
    model: Encoder, head: torch.nn.Module, lr: float, factor: float, lower_fraction: float
) -> list[dict[str, Any]]:
    """AdamW groups: the head at `lr`; the encoder's upper blocks and final norm at
    lr x factor; its input embedding and lowest blocks (`lower_fraction` of them, rounded) at
    lr x factor^2. The pretraining head (`model.head`) is not used by a task and left out."""
    lower_blocks = round(len(model.blocks) * lower_fraction)
    embedding = [model.samples, model.context, model.slot]
    lower = [p for m in embedding for p in m.parameters()] + [model.hidden]
    lower += [p for b in model.blocks[:lower_blocks] for p in b.parameters()]
    upper = [p for b in model.blocks[lower_blocks:] for p in b.parameters()]
    upper += list(model.norm.parameters())
    return [
        {"params": list(head.parameters()), "lr": lr},
        {"params": upper, "lr": lr * factor},
        {"params": lower, "lr": lr * factor * factor},
    ]


def _crop(example: Example, start_s: float, length_s: float) -> Example:
    """`length_s` of every signal from `start_s` on (whole samples at each signal's rate)."""
    signals = []
    for sig in example.signals:
        first = round(start_s * sig.sampling_rate)
        n = round(length_s * sig.sampling_rate)
        signals.append(Signal(sig.values[first : first + n], sig.sampling_rate, 0.0))
    return Example(tuple(signals), example.origin)


def _duration_s(example: Example) -> float:
    return min(len(sig.values) / sig.sampling_rate for sig in example.signals)


def _random_crop(example: Example, length_s: float | None, rng: random.Random) -> Example:
    if length_s is None:
        return example
    rate = min(sig.sampling_rate for sig in example.signals)
    starts = round((_duration_s(example) - length_s) * rate)
    return _crop(example, rng.randint(0, max(0, starts)) / rate, length_s)


def _crops(example: Example, length_s: float | None) -> list[Example]:
    """The non-overlapping crops of a record, from its start; a rest shorter than one crop
    is left out (their min_chunk_length)."""
    if length_s is None:
        return [example]
    n = int(_duration_s(example) / length_s + 1e-9)
    return [_crop(example, k * length_s, length_s) for k in range(n)]


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy of probabilities `p` (as in training, over all labels)."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def run(
    store: Store, checkpoint: dict[str, Any], settings: PtbxlSettings, device: torch.device,
    log: Log,
) -> dict[str, float]:  # fmt: skip
    """All scores of one checkpoint, as a flat dict for logging: `<task>/<method>/<name>`."""
    from biosignal_model.train import TrainConfig  # noqa: PLC0415 - avoids a cycle

    train_config = TrainConfig.from_checkpoint(checkpoint["config"])
    config = train_config.model_settings()
    max_tokens = train_config.tokens.max_tokens
    records, statements = load(store, settings, log)
    pretrained = Encoder(config).to(device)
    pretrained.load_state_dict(checkpoint["model"])
    torch.manual_seed(settings.seed)
    untrained = Encoder(config).to(device)
    features = {}
    if settings.linear_probing:
        features = {
            "linear_probing": embed(
                pretrained, records.examples, max_tokens, settings.rows, device, log,
                "linear_probing",
            ),
            "linear_probing_random": embed(
                untrained, records.examples, max_tokens, settings.rows, device, log,
                "linear_probing_random",
            ),
        }
    results: dict[str, float] = {
        "records": float(len(records.ids)),
        "rate_hz": float(settings.rate),
    }
    for task in settings.tasks:
        y, classes, keep = labels(records, statements, task)
        kept = Records(
            [records.ids[i] for i in np.flatnonzero(keep)],
            [records.examples[i] for i in np.flatnonzero(keep)],
            records.folds[keep], [records.statements[i] for i in np.flatnonzero(keep)],
        )  # fmt: skip
        results[f"{task}/classes"] = float(len(classes))
        results[f"{task}/records"] = float(keep.sum())
        for name, f in features.items():
            scores = linear_probing(
                f[keep], y[keep], records.folds[keep], settings, log, f"{task}/{name}"
            )
            results |= {f"{task}/{name}/{k}": v for k, v in scores.items()}
            log({f"{task}/{name}/macro_auc": scores["macro_auc"]})
        starts = {"fine_tuning": checkpoint["model"], "from_scratch": None}
        for name in settings.methods:
            start = starts[name]
            scores = end_to_end(
                config, kept, y[keep], max_tokens, settings, device, start, log, f"{task}/{name}"
            )
            results |= {f"{task}/{name}/{k}": v for k, v in scores.items()}
            log({f"{task}/{name}/macro_auc": scores["macro_auc"]})
    return results

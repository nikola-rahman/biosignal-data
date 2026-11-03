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
import copy
import csv
import functools
import io
import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.stats import rankdata

from biosignal_data.read.standard import StandardDataset
from biosignal_data.storage.stores import Store
from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.batch import Batch, Packer, Tokenizer
from biosignal_model.benchmarks import LOG_EVERY_STEPS, Log, TaskModel
from biosignal_model.benchmarks.s4 import S4ChannelsSettings, S4Settings
from biosignal_model.benchmarks.s4 import train_and_score as train_and_score_s4
from biosignal_model.benchmarks.s4_incontext import S4InContextSettings
from biosignal_model.benchmarks.s4_incontext import train_and_score as train_and_score_incontext
from biosignal_model.benchmarks.xresnet1d import ReferenceSettings, train_and_score
from biosignal_model.model import Encoder, ModelConfig

DATASET = "physionet/ptb-xl"
TASKS = ("all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm")
TRAIN, VALIDATION, TEST = (1, 2, 3, 4, 5, 6, 7, 8), (9,), (10,)


class FromScratchModel(BaseModel):
    """The size and regularization of a from_scratch model (the global transformer; the
    tokenizer's own sizes are in its settings): for small proxy models of the v3 bench."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    width: int
    layers: int
    heads: int
    feed_forward: int
    dropout: float
    drop_path: float


class CooldownRungs(BaseModel):
    """WSD cooldown branches (Hägele et al. 2024, arXiv 2405.18392): the main run trains at a
    constant rate after warmup; at each rung epoch its state is kept, a branch decays the rate
    linearly to 0 over max(`min_epochs`, round(`fraction` x rung)) more epochs and is scored,
    and the main run goes on from the kept state. A rung's score is its branch's: a cut on it
    does not keep fast learners that a mid-schedule snapshot favours (coordinator,
    2026-10-29). The last rung is the end of the main run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs: list[int]
    fraction: float
    min_epochs: int


class PtbxlSettings(BaseModel):
    """Every setting of the benchmark; none has a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rate: Literal[100, 500]
    tasks: list[Literal["all", "diagnostic", "subdiagnostic", "superdiagnostic", "form", "rhythm"]]
    rows: int  # rows of the model's max_tokens in one batch, at most
    read_threads: int  # requests to the store at the same time
    workers: int  # data loader processes that crop and pack batches; 0: the main process
    linear_probing: bool  # also linear probing of the frozen encoder (pretrained and random)
    # False: a random channel slot per record, as in pretraining (the model is not told which
    # lead is which). True: one fixed slot per lead name, so fine-tuning can learn the leads
    # (a CNN gets them in a fixed order). Test asked for by the user, 2026-10-29.
    fixed_slots: bool
    # fine_tuning: the pretrained encoder and a head; from_scratch: the same from random weights
    # reference_xresnet1d101: the paper's own baseline (benchmarks/xresnet1d.py) trained through
    # this pipeline, with its recipe (`reference`), to check the pipeline itself;
    # reference_s4: the supervised S4 of Mehari & Strodthoff (benchmarks/s4.py, `reference_s4`);
    # reference_s4_channels: the same with the leads as a set (S4ChannelModel);
    # reference_s4_incontext: the same with an in-context head over retrieved training records
    # (benchmarks/s4_incontext.py)
    methods: list[
        Literal[
            "fine_tuning", "from_scratch", "reference_xresnet1d101", "reference_s4",
            "reference_s4_channels", "reference_s4_incontext",
        ]
    ]  # fmt: skip
    reference: ReferenceSettings | None  # the recipe of reference_xresnet1d101; None: not run
    reference_s4: S4Settings | None  # the recipe of reference_s4; None: not run
    reference_s4_channels: S4ChannelsSettings | None  # of reference_s4_channels; None: not run
    reference_s4_incontext: S4InContextSettings | None  # of reference_s4_incontext; None: not run
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
    # Validation and test records: crops every `evaluation_stride_fraction` x the crop length
    # (1.0: no overlap, as arXiv 2509.25095; 0.5: Strodthoff et al. 2021, stride_length_valid =
    # input_size // 2), and their probabilities joined by mean (2509.25095) or max (Strodthoff,
    # aggregate_fn "max").
    evaluation_stride_fraction: float
    evaluation_aggregate: Literal["mean", "max"]
    # Where the channel slot embedding trains: "lower" (the input embedding's group, lr x
    # factor^2; pretraining taught it only "same channel") or "head" (at the head's rate, so
    # fixed slots can learn the leads; user, 2026-10-29).
    fine_tuning_slot_group: Literal["lower", "head"]
    # The optimizer: AdamW with these betas, linear warmup over `fine_tuning_warmup_epochs`,
    # then a constant rate or a cosine decay to 0 at fine_tuning_epochs; gradients clipped to
    # this norm (None: not clipped). Pretraining used beta2 0.95, warmup and clip 1.0; the
    # recipe of 2509.25095 is constant, without warmup or clipping.
    fine_tuning_betas: tuple[float, float]
    fine_tuning_warmup_epochs: float
    fine_tuning_schedule: Literal["constant", "cosine"]
    fine_tuning_grad_clip: float | None
    # "linear": a linear head on the mean feature. "reference": the xresnet1d101 head of
    # Strodthoff et al. (fastai create_head1d, lin_ftrs [128], ps 0.5): mean and max over the
    # tokens, BatchNorm, dropout 0.25, Linear(2 x width, 128), ReLU, BatchNorm, dropout 0.5,
    # Linear(128, classes).
    fine_tuning_head: Literal["linear", "reference"]
    # The token size (samples per token) of from_scratch runs; None: the checkpoint's. Only for
    # from_scratch, whose weights are new: a test of the token size without pretraining (at
    # 100 Hz the checkpoint's 64 samples are 0.64 s, about a whole beat; user, 2026-10-29).
    from_scratch_samples_per_token: int | None
    # The tokenizer of from_scratch runs (docs/v3-tokenizer.md); None: the checkpoint's. Only for
    # from_scratch runs alone, like from_scratch_samples_per_token (the v3 tokenizer bench).
    from_scratch_tokenizer: Tokenizer | None
    # The size and regularization of from_scratch models; None: the checkpoint's (proxy models
    # of the v3 bench, user 2026-10-29: "small proxy models ... successive halving").
    from_scratch_model: FromScratchModel | None
    # Cooldown branches (CooldownRungs) of fine_tuning / from_scratch; the run's scores are its
    # rungs' and fold 9 chooses the rung. Needs the constant schedule, no patience, and the
    # last rung at fine_tuning_epochs. None: one run, the best fold-9 epoch is chosen.
    fine_tuning_rungs: CooldownRungs | None
    seed: int

    @model_validator(mode="after")
    def _rungs_fit(self) -> PtbxlSettings:
        r = self.fine_tuning_rungs
        if r is None:
            return self
        if self.fine_tuning_schedule != "constant" or self.fine_tuning_patience is not None:
            raise ValueError("fine_tuning_rungs need the constant schedule and no patience")
        if r.epochs != sorted(set(r.epochs)) or r.epochs[-1] != self.fine_tuning_epochs:
            raise ValueError("rung epochs must increase and end at fine_tuning_epochs")
        if r.epochs[0] < self.fine_tuning_warmup_epochs or r.min_epochs < 1:
            raise ValueError("rungs start after the warmup; a branch has at least 1 epoch")
        return self


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


def label_groups(classes: list[str], statements: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """The columns of each statement group among `classes` (task "all"): diagnostic, form and
    rhythm, by the flags of scp_statements.csv (a statement can be in more than one)."""
    by_code = {s[""]: s for s in statements}
    return {
        group: np.array([by_code.get(c, {}).get(group) in ("1", "1.0") for c in classes])
        for group in ("diagnostic", "form", "rhythm")
    }


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """The ROC AUC of one label (ties get average ranks); None without both classes."""
    positive = y == 1
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(p)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_auc(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    """The mean ROC AUC over the labels with both classes in `y` (ties get average ranks),
    and how many labels that is."""
    aucs = [a for k in range(y.shape[1]) if (a := _auc(y[:, k], p[:, k])) is not None]
    return float(np.mean(aucs)), len(aucs)


def per_label(
    classes: list[str], y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray,
    p_test: np.ndarray,
) -> dict[str, float]:  # fmt: skip
    """Each label's score for the chosen epoch: `label/<code>/validation_positives`,
    `.../test_positives` (fold 9, fold 10) and `.../validation_auc`, `.../test_auc` (only
    where the fold has both classes)."""
    out: dict[str, float] = {}
    for k, code in enumerate(classes):
        for fold, y, p in (("validation", y_val, p_val), ("test", y_test, p_test)):
            out[f"label/{code}/{fold}_positives"] = float((y[:, k] == 1).sum())
            if (auc := _auc(y[:, k], p[:, k])) is not None:
                out[f"label/{code}/{fold}_auc"] = auc
    return out


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


def _packer(model: ModelConfig, max_tokens: int, rows: int, slots: dict[str, int] | None) -> Packer:
    return Packer(
        datasets=[DATASET], max_tokens=max_tokens, samples_per_token=model.samples_per_token,
        max_channels=model.max_channels, rows=rows, mask_ratio=0.0, seed=0, slots=slots,
        tokenizer=model.tokenizer,
    )  # fmt: skip


def lead_slots(examples: list[Example], fixed: bool) -> dict[str, int] | None:
    """None (random slots), or one slot per channel name of the records, in name order."""
    if not fixed:
        return None
    names = sorted({c.name for e in examples for c in e.origin.channels})
    return {name: k for k, name in enumerate(names)}


def _pooled(
    model: TaskModel, examples: list[Example], packer: Packer, device: torch.device
) -> torch.Tensor:
    """The mean feature over all tokens of each example, [len(examples), width]."""
    batch, where = packer.pack(examples, trim=True)
    return _mean_features(model, batch, where, device)


def _mean_features(
    model: TaskModel, batch: Batch, where: list[tuple[int, int]], device: torch.device,
    with_max: bool = False,
) -> torch.Tensor:  # fmt: skip
    """The mean feature over the tokens of each (row, document) in `where`, in one pass
    over the batch: [len(where), width]; `with_max`: the mean and the maximum, concatenated,
    [len(where), 2 x width]."""
    x = batch.inputs.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = model.features(x).float()
    rows, length, width = features.shape
    key = (torch.arange(rows, device=device)[:, None] * length + x.document).flatten()
    real = x.document.flatten() >= 0
    flat = features.reshape(-1, width)[real]
    sums = torch.zeros(rows * length, width, device=device, dtype=features.dtype)
    sums.index_add_(0, key[real], flat)
    counts = torch.bincount(key[real], minlength=rows * length)
    chosen = torch.tensor([r * length + d for r, d in where], device=device)
    mean = sums[chosen] / counts[chosen, None]
    if not with_max:
        return mean
    highest = torch.full((rows * length, width), -torch.inf, device=device, dtype=features.dtype)
    highest.scatter_reduce_(0, key[real][:, None].expand(-1, width), flat, "amax")
    return torch.cat([mean, highest[chosen]], dim=-1)


class ReferenceHead(torch.nn.Module):
    """The xresnet1d101 head of Strodthoff et al. on [mean, max] features (see
    PtbxlSettings.fine_tuning_head)."""

    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * width), torch.nn.Dropout(0.25),
            torch.nn.Linear(2 * width, 128), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128), torch.nn.Dropout(0.5), torch.nn.Linear(128, classes),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out


def _schedule(s: PtbxlSettings, steps_per_epoch: int) -> Callable[[int], float]:
    """The learning-rate factor of each optimizer step (see PtbxlSettings)."""
    warmup = s.fine_tuning_warmup_epochs * steps_per_epoch
    total = s.fine_tuning_epochs * steps_per_epoch

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if s.fine_tuning_schedule == "constant":
            return 1.0
        done = (step - warmup) / max(1.0, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, done)))

    return factor


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
    model: TaskModel, examples: list[Example], max_tokens: int, rows: int, device: torch.device,
    log: Log, name: str, slots: dict[str, int] | None,
) -> np.ndarray:  # fmt: skip
    """The mean feature of every example; progress as `embed/<name>/...`."""
    model.eval()
    packer = _packer(model.config, max_tokens, rows, slots)
    batches = _batches(len(examples), packer, examples, rows)
    parts, started = [], time.monotonic()
    for k, ids in enumerate(batches, 1):
        parts.append(_pooled(model, [examples[i] for i in ids], packer, device).cpu())
        if k % max(1, len(batches) // 10) == 0 or k == len(batches):
            log({f"embed/{name}/batches": float(k), f"embed/{name}/of": float(len(batches)),
                 f"embed/{name}/seconds": time.monotonic() - started})  # fmt: skip
    return torch.cat(parts).numpy()


def _fit(x: torch.Tensor, y: torch.Tensor, l2: float, steps: int) -> torch.Tensor:
    """Weights [d + 1, classes] of one logistic regression per label (bias not penalized), on
    the device of `x`."""
    w = torch.zeros(
        x.shape[1] + 1, y.shape[1], dtype=torch.float64, device=x.device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ w[:-1] + w[-1]
        loss = F.binary_cross_entropy_with_logits(logits, y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach()


def linear_probing(
    features: np.ndarray, y: np.ndarray, folds: np.ndarray, s: PtbxlSettings, log: Log,
    prefix: str, device: torch.device,
) -> dict[str, float]:  # fmt: skip
    """L2 chosen on fold 9 by macro-AUC, then one fit on folds 1-8, scored on fold 10. Each
    L2 tried is logged with its fold-9 macro-AUC, under `prefix`."""
    train, val, test = (np.isin(folds, f) for f in (TRAIN, VALIDATION, TEST))
    mean, std = features[train].mean(0), features[train].std(0) + 1e-8
    # On the GPU: on a busy CPU one fit of the 71 "all" labels took 15-20 minutes (2026-10-29).
    x = torch.from_numpy((features - mean) / std).double().to(device)
    yt = torch.from_numpy(y).double().to(device)
    best, best_auc = s.linear_probing_l2[0], -1.0
    for l2 in s.linear_probing_l2:
        w = _fit(x[train], yt[train], l2, s.linear_probing_steps)
        auc, _ = macro_auc(y[val], (x[val] @ w[:-1] + w[-1]).cpu().numpy())
        log({f"{prefix}/l2": l2, f"{prefix}/validation_macro_auc": auc})
        if auc > best_auc:
            best, best_auc = l2, auc
    w = _fit(x[train], yt[train], best, s.linear_probing_steps)
    auc, n = macro_auc(y[test], (x[test] @ w[:-1] + w[-1]).cpu().numpy())
    return {"macro_auc": auc, "labels": float(n), "l2": best, "validation_macro_auc": best_auc}


@dataclass
class _Best:
    """The chosen epoch (or rung) so far: fold 9 chooses; its fold-10 scores and the
    probabilities on fold 9 and fold 10."""

    chosen: tuple[np.ndarray, np.ndarray]
    validation: float = -1.0
    test: float = 0.0
    epoch: int = 0
    labels: int = 0
    groups: dict[str, float] = field(default_factory=dict)


@dataclass
class _FineTuning:
    """The state of one fine-tuning run: the encoder, the head, the optimizer and its schedule,
    the step count, and the rates at the last rung (a cooldown branch decays from them)."""

    model: Encoder
    head: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    trained: list[torch.nn.Parameter]
    packer: Packer
    examples: list[Example]
    y: np.ndarray
    s: PtbxlSettings
    device: torch.device
    log: Log
    prefix: str
    with_max: bool  # the head takes the mean and the maximum feature (the reference head)
    step: int = 0
    fork_lrs: list[float] = field(default_factory=list)

    def predict(self, indices: list[int]) -> np.ndarray:
        s = self.s
        return _predict(self.model, self.head, [self.examples[i] for i in indices],
                        s.fine_tuning_crop_s, self.packer, s.rows, self.device, s.workers,
                        f"{s.seed}/predict", s.evaluation_stride_fraction,
                        s.evaluation_aggregate, self.with_max)  # fmt: skip

    def train_epoch(
        self, epoch: int, order: list[int], rate: Callable[[int], float] | None
    ) -> float:
        """One epoch over `order`: its mean training loss. `rate`: None, the scheduler sets the
        learning rate (the main run); else the rate of the epoch's step k is the fork's rate x
        rate(k) (a cooldown branch; it moves no step counter)."""
        s, n = self.s, self.s.fine_tuning_examples_per_step
        self.model.train()
        self.head.train()
        losses: list[float] = []
        window: list[float] = []
        steps = [order[k : k + n] for k in range(0, len(order), n)]
        crop = functools.partial(_crop_to, s.fine_tuning_crop_s)
        batches = _packed(self.examples, steps, crop, self.packer, f"{s.seed}/{epoch}", s.workers)
        for k, (ids, (batch, where)) in enumerate(zip(steps, batches, strict=True)):
            if rate is not None:
                for g, fork_lr in zip(self.optimizer.param_groups, self.fork_lrs, strict=True):
                    g["lr"] = fork_lr * rate(k)
            pooled = _mean_features(self.model, batch, where, self.device, self.with_max)
            target = torch.from_numpy(self.y[ids]).to(self.device)
            loss = F.binary_cross_entropy_with_logits(self.head(pooled), target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if s.fine_tuning_grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.trained, s.fine_tuning_grad_clip)
            self.optimizer.step()
            window.append(loss.item())
            if rate is None:
                self.scheduler.step()
                self.step += 1
                if self.step % LOG_EVERY_STEPS == 0:
                    self.log({f"{self.prefix}/train_loss": float(np.mean(window)),
                              f"{self.prefix}/step": float(self.step)})  # fmt: skip
                    losses += window
                    window = []
        return float(np.mean(losses + window))


def _crop_to(length_s: float | None, example: Example, rng: random.Random) -> Example:
    """_random_crop with the crop length first (a picklable partial for the data workers)."""
    return _random_crop(example, length_s, rng)


def _groups_auc(
    groups: dict[str, np.ndarray] | None, y: np.ndarray, p: np.ndarray
) -> dict[str, float]:
    """The macro-AUC over each label group's columns."""
    return {name: macro_auc(y[:, c], p[:, c])[0] for name, c in (groups or {}).items()}


def _cooldown_rung(
    run: _FineTuning, epoch: int, rungs: CooldownRungs, rng: random.Random, train: list[int],
    folds: tuple[list[int], list[int], list[int]], groups: dict[str, np.ndarray] | None,
    best: _Best,
) -> dict[str, float]:  # fmt: skip
    """The cooldown branch at rung `epoch`: from the main run's state, kept and put back after
    it (not after the last rung), the rate decays linearly to 0 over cooldown_epochs; scored
    on (fold 9, the fixed part of folds 1-8, fold 10) = `folds`. Fold 9 chooses the rung
    (`best`). Returns the rung's scores."""
    val, train_eval, test = folds
    y = run.y
    last = epoch == rungs.epochs[-1]
    kept = None
    if not last:
        kept = _Kept.of(run.model, run.head, run.optimizer, run.scheduler, rng, train, run.device)
    cooldown = cooldown_epochs(rungs, epoch)
    per_epoch = -(-len(train) // run.s.fine_tuning_examples_per_step)
    run.fork_lrs = [g["lr"] for g in run.optimizer.param_groups]
    branch_rng = random.Random()  # noqa: S311 - not secret
    branch_rng.setstate(rng.getstate())
    order = list(train)
    for b in range(cooldown):
        branch_rng.shuffle(order)
        run.train_epoch(epoch + 1 + b, order, _linear_to_zero(b * per_epoch, cooldown * per_epoch))
    p_val, p_train, p_test = run.predict(val), run.predict(train_eval), run.predict(test)
    auc_val, _ = macro_auc(y[val], p_val)
    auc_test, n_test = macro_auc(y[test], p_test)
    val_loss, train_eval_loss = _bce(y[val], p_val), _bce(y[train_eval], p_train)
    rung = {
        "epoch": float(epoch), "cooldown_epochs": float(cooldown),
        "validation_macro_auc": auc_val, "test_macro_auc": auc_test,
        "validation_loss": val_loss, "train_eval_loss": train_eval_loss,
        "train_minus_validation_loss": train_eval_loss - val_loss,
    } | {f"validation_macro_auc_{k}": v for k, v in _groups_auc(groups, y[val], p_val).items()} | {
        f"test_macro_auc_{k}": v for k, v in _groups_auc(groups, y[test], p_test).items()
    }  # fmt: skip
    run.log({f"{run.prefix}/rung_{k}": v for k, v in rung.items()})
    if auc_val > best.validation:  # fold 9 chooses the rung
        best.validation, best.test, best.epoch, best.labels = auc_val, auc_test, epoch, n_test
        best.groups = _groups_auc(groups, y[test], p_test)
        best.chosen = p_val, p_test
    if kept is not None:
        kept.put_back(run.model, run.head, run.optimizer, run.scheduler, rng, train)
    return {f"rung_{epoch}/{k}": v for k, v in rung.items()}


def end_to_end(
    config: ModelConfig, records: Records, y: np.ndarray, max_tokens: int, s: PtbxlSettings,
    device: torch.device, start: dict[str, torch.Tensor] | None, log: Log, prefix: str,
    scored_groups: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """The encoder and a linear head on its mean feature, every weight trained on folds 1-8
    (layer-dependent learning rates and crops: see PtbxlSettings); the epoch with the best
    macro-AUC on fold 9 is scored on fold 10 (at most `fine_tuning_patience` epochs after it are
    run). `start`: pretrained weights, or None.
    Logged under `prefix`: the training loss every LOG_EVERY_STEPS steps, and per epoch its
    mean training loss, fold-9 macro-AUC, time and speed; fold 10 only for a chosen epoch.
    `scored_groups`: columns of label groups (label_groups), also scored on their own
    (macro-AUC over the group's labels), on fold 9 each epoch and on fold 10 for a chosen
    epoch; fold 9 alone (all labels) still chooses the epoch. With `fine_tuning_rungs`: a
    cooldown branch at each rung (CooldownRungs), scored on fold 9 and fold 10 with its loss
    gap (`rung_<epoch>/...` in the scores, `rung_...` in the log); fold 9 chooses the rung and
    the main run's epochs are not scored on fold 10. Also returned: the chosen epoch's (or
    rung's) probabilities on fold 9 and fold 10 (records in order)."""
    torch.manual_seed(s.seed)
    model = Encoder(config).to(device)
    if start is not None:
        model.load_state_dict(start)
    with_max = s.fine_tuning_head == "reference"
    head: torch.nn.Module = (
        ReferenceHead(config.width, y.shape[1]) if with_max
        else torch.nn.Linear(config.width, y.shape[1])
    ).to(device)  # fmt: skip
    factor = s.fine_tuning_discriminative_factor if start is not None else 1.0
    groups = parameter_groups(model, head, s.fine_tuning_learning_rate, factor,
                              s.fine_tuning_lower_blocks_fraction,
                              s.fine_tuning_slot_group)  # fmt: skip
    optimizer = torch.optim.AdamW(
        groups, weight_decay=s.fine_tuning_weight_decay, betas=s.fine_tuning_betas
    )
    packer = _packer(config, max_tokens, s.rows, lead_slots(records.examples, s.fixed_slots))
    train = [i for i in range(len(records.ids)) if records.folds[i] in TRAIN]
    rng = random.Random(s.seed)  # noqa: S311 - not secret
    per_epoch = -(-len(train) // s.fine_tuning_examples_per_step)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(s, per_epoch))
    run = _FineTuning(model, head, optimizer, scheduler, [p for g in groups for p in g["params"]],
                      packer, records.examples, y, s, device, log, prefix, with_max)  # fmt: skip
    val = [i for i in range(len(records.ids)) if records.folds[i] in VALIDATION]
    test = [i for i in range(len(records.ids)) if records.folds[i] in TEST]
    # A fixed part of folds 1-8, as large as fold 9, scored in eval mode each epoch: its
    # loss next to the fold-9 loss shows overfitting on one scale.
    rng_eval = random.Random(s.seed)  # noqa: S311 - not secret
    train_eval = sorted(rng_eval.sample(train, min(len(train), len(val))))
    rungs = s.fine_tuning_rungs
    best = _Best((np.zeros((0, y.shape[1])), np.zeros((0, y.shape[1]))))
    scores: dict[str, float] = {}
    for epoch in range(1, s.fine_tuning_epochs + 1):
        rng.shuffle(train)
        started = time.monotonic()
        train_loss = run.train_epoch(epoch, train, None)
        seconds = time.monotonic() - started
        p_val = run.predict(val)
        auc_val, _ = macro_auc(y[val], p_val)
        record = {
            f"{prefix}/epoch": float(epoch),
            f"{prefix}/epoch_train_loss": train_loss,
            f"{prefix}/epoch_train_eval_loss": _bce(y[train_eval], run.predict(train_eval)),
            f"{prefix}/epoch_validation_loss": _bce(y[val], p_val),
            f"{prefix}/epoch_validation_macro_auc": auc_val,
            f"{prefix}/epoch_seconds": seconds,
            f"{prefix}/examples_per_s": len(train) / seconds,
        }
        for name, value in _groups_auc(scored_groups, y[val], p_val).items():
            record[f"{prefix}/epoch_validation_macro_auc/{name}"] = value
        if rungs is None and auc_val > best.validation:  # fold 10 for the epoch fold 9 chooses
            p_test = run.predict(test)
            best.validation, best.epoch, best.chosen = auc_val, epoch, (p_val, p_test)
            best.test, best.labels = macro_auc(y[test], p_test)
            best.groups = _groups_auc(scored_groups, y[test], p_test)
            record[f"{prefix}/chosen_epoch"] = float(epoch)
            record[f"{prefix}/chosen_test_macro_auc"] = best.test
            for name, value in best.groups.items():
                record[f"{prefix}/chosen_test_macro_auc/{name}"] = value
        log(record)
        if rungs is not None and epoch in rungs.epochs:
            scores |= _cooldown_rung(run, epoch, rungs, rng, train, (val, train_eval, test),
                                     scored_groups, best)  # fmt: skip
        if s.fine_tuning_patience is not None and epoch - best.epoch >= s.fine_tuning_patience:
            break
    return {"macro_auc": best.test, "labels": float(best.labels), "epoch": float(best.epoch),
            "validation_macro_auc": best.validation, "epochs_run": float(epoch)} | {
        f"macro_auc_{name}": value for name, value in best.groups.items()
    } | scores, best.chosen  # fmt: skip


def _linear_to_zero(done: int, total: int) -> Callable[[int], float]:
    """The rate factor of a cooldown branch's step k of an epoch after `done` of its `total`
    steps: linear from 1 to 0."""
    return lambda k: 1.0 - (done + k) / total


@dataclass
class _Kept:
    """The main run's state at a rung: weights, optimizer, scheduler, random states, data
    order."""

    model: dict[str, Any]
    head: dict[str, Any]
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    torch_rng: torch.Tensor
    cuda_rng: list[torch.Tensor] | None
    rng: tuple[Any, ...]
    train: list[int]

    @staticmethod
    def of(
        model: torch.nn.Module, head: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler, rng: random.Random, train: list[int],
        device: torch.device,
    ) -> _Kept:  # fmt: skip
        return _Kept(
            copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()),
            copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scheduler.state_dict()),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            rng.getstate(), list(train),
        )  # fmt: skip

    def put_back(
        self, model: torch.nn.Module, head: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler, rng: random.Random, train: list[int],
    ) -> None:  # fmt: skip
        model.load_state_dict(self.model)
        head.load_state_dict(self.head)
        optimizer.load_state_dict(self.optimizer)
        scheduler.load_state_dict(self.scheduler)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        rng.setstate(self.rng)
        train[:] = self.train


def cooldown_epochs(rungs: CooldownRungs, epoch: int) -> int:
    """The length of the cooldown branch at a rung epoch, in epochs."""
    return max(rungs.min_epochs, round(rungs.fraction * epoch))


@torch.no_grad()
def _predict(
    model: TaskModel, head: torch.nn.Module, examples: list[Example], crop_s: float | None,
    packer: Packer, rows: int, device: torch.device, workers: int, seed: str,
    stride_fraction: float, aggregate: Literal["mean", "max"], with_max: bool,
) -> np.ndarray:  # fmt: skip
    """The probabilities of each example: the mean or the maximum over its crops."""
    model.eval()
    head.eval()
    crops, owner = [], []
    for n, example in enumerate(examples):
        for c in _crops(example, crop_s, stride_fraction):
            crops.append(c)
            owner.append(n)
    out = []
    groups = _batches(len(crops), packer, crops, rows)
    for batch, where in _packed(crops, groups, _unchanged, packer, seed, workers):
        pooled = _mean_features(model, batch, where, device, with_max)
        out.append(torch.sigmoid(head(pooled)).float().cpu())
    p = torch.cat(out).numpy()
    if aggregate == "max":
        highest = np.zeros((len(examples), p.shape[1]))
        np.maximum.at(highest, owner, p)
        return highest
    sums = np.zeros((len(examples), p.shape[1]))
    np.add.at(sums, owner, p)
    return sums / np.bincount(owner, minlength=len(examples))[:, None]


def parameter_groups(
    model: TaskModel, head: torch.nn.Module, lr: float, factor: float, lower_fraction: float,
    slot_group: Literal["lower", "head"],
) -> list[dict[str, Any]]:  # fmt: skip
    """AdamW groups from `model.layer_groups()`: the head at `lr`; the upper layers and the
    output part at lr x factor; the input part and the lowest layers (`lower_fraction` of them,
    rounded) at lr x factor^2; the channel identity (slot embedding) with the lower part or the
    head (`slot_group`). The pretraining heads are not in the layer groups."""
    g = model.layer_groups()
    lower_layers = round(len(g.layers) * lower_fraction)
    lower = g.input + (g.channel if slot_group == "lower" else [])
    lower += [p for layer in g.layers[:lower_layers] for p in layer]
    upper = [p for layer in g.layers[lower_layers:] for p in layer] + g.output
    top = list(head.parameters()) + (g.channel if slot_group == "head" else [])
    return [
        {"params": top, "lr": lr},
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


def _crops(example: Example, length_s: float | None, stride_fraction: float) -> list[Example]:
    """The crops of a record, from its start, every `stride_fraction` x `length_s` (1: no
    overlap); a rest shorter than one crop is left out (their min_chunk_length)."""
    if length_s is None:
        return [example]
    stride = length_s * stride_fraction
    n = int((_duration_s(example) - length_s) / stride + 1e-9) + 1
    return [_crop(example, k * stride, length_s) for k in range(max(0, n))]


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy of probabilities `p` (as in training, over all labels)."""
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _from_scratch_config(config: ModelConfig, settings: PtbxlSettings) -> ModelConfig:
    """`config` with the from_scratch_* settings that are set; each is only for from_scratch
    runs alone (no pretrained weights fit the changed model)."""
    tokens = settings.from_scratch_samples_per_token
    tokenizer = settings.from_scratch_tokenizer
    model = settings.from_scratch_model
    changes: list[tuple[str, dict[str, Any] | None]] = [
        (
            "from_scratch_samples_per_token",
            None if tokens is None else {"samples_per_token": tokens},
        ),
        ("from_scratch_tokenizer", None if tokenizer is None else {"tokenizer": tokenizer}),
        ("from_scratch_model", None if model is None else model.model_dump()),
    ]
    for name, fields_set in changes:
        if fields_set is None:
            continue
        if settings.methods != ["from_scratch"] or settings.linear_probing:
            raise ValueError(f"{name} is only for from_scratch runs alone")
        config = replace(config, **fields_set)
    return config


def _probe_features(
    config: ModelConfig, checkpoint: dict[str, Any], records: Records, max_tokens: int,
    settings: PtbxlSettings, device: torch.device, log: Log,
) -> dict[str, np.ndarray]:  # fmt: skip
    """The linear-probing features of the pretrained and of a random encoder."""
    pretrained = Encoder(config).to(device)
    pretrained.load_state_dict(checkpoint["model"])
    torch.manual_seed(settings.seed)
    untrained = Encoder(config).to(device)
    slots = lead_slots(records.examples, settings.fixed_slots)
    return {
        name: embed(encoder, records.examples, max_tokens, settings.rows, device, log, name, slots)
        for name, encoder in (("linear_probing", pretrained), ("linear_probing_random", untrained))
    }


def _method_scores(
    name: str, task: str, config: ModelConfig, kept: Records, y: np.ndarray, max_tokens: int,
    settings: PtbxlSettings, device: torch.device, checkpoint: dict[str, Any], log: Log,
    groups: dict[str, np.ndarray] | None,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """The scores and the fold-9 and fold-10 probabilities of method `name` on one task."""
    prefix = f"{task}/{name}"
    if name == "reference_xresnet1d101":
        return _reference(kept, y, settings, device, log, prefix)
    if name in ("reference_s4", "reference_s4_channels", "reference_s4_incontext"):
        return _reference_s4(kept, y, settings, device, log, prefix, name)
    start = {"fine_tuning": checkpoint["model"], "from_scratch": None}[name]
    return end_to_end(config, kept, y, max_tokens, settings, device, start, log, prefix,
                      scored_groups=groups)  # fmt: skip


def run(
    store: Store, checkpoint: dict[str, Any], settings: PtbxlSettings, device: torch.device,
    log: Log,
) -> dict[str, float]:  # fmt: skip
    """All scores of one checkpoint, as a flat dict for logging: `<task>/<method>/<name>`."""
    from biosignal_model.train import TrainConfig  # noqa: PLC0415 - avoids a cycle

    train_config = TrainConfig.from_checkpoint(checkpoint["config"])
    config = _from_scratch_config(train_config.model_settings(), settings)
    max_tokens = train_config.tokens.max_tokens
    records, statements = load(store, settings, log)
    features = {}
    if settings.linear_probing:
        features = _probe_features(config, checkpoint, records, max_tokens, settings, device, log)
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
                f[keep], y[keep], records.folds[keep], settings, log, f"{task}/{name}", device
            )
            results |= {f"{task}/{name}/{k}": v for k, v in scores.items()}
            log({f"{task}/{name}/macro_auc": scores["macro_auc"]})
        groups = label_groups(classes, statements) if task == "all" else None
        for name in settings.methods:
            scores, (p_val, p_test) = _method_scores(
                name, task, config, kept, y[keep], max_tokens, settings, device, checkpoint,
                log, groups,
            )  # fmt: skip
            val, test = np.isin(kept.folds, VALIDATION), np.isin(kept.folds, TEST)
            scores |= per_label(classes, y[keep][val], p_val, y[keep][test], p_test)
            if groups is not None and name.startswith("reference"):  # end_to_end scores its own
                for group, c in groups.items():
                    scores[f"macro_auc_{group}"] = macro_auc(y[keep][test][:, c], p_test[:, c])[0]
            results |= {f"{task}/{name}/{k}": v for k, v in scores.items()}
            log({f"{task}/{name}/macro_auc": scores["macro_auc"]})
    return results


def _records_array(records: Records) -> np.ndarray:
    """[records, leads, samples] in the order of the files (I, II, III, aVR, aVL, aVF, V1-V6, as
    the papers' wfdb.rdsamp), in physical units (mV)."""
    lengths = {sig.values.size for e in records.examples for sig in e.signals}
    leads = {len(e.signals) for e in records.examples}
    if len(lengths) != 1 or len(leads) != 1:
        raise ValueError(f"records of different lengths {lengths} or leads {leads}")
    x = np.stack([np.stack([sig.values for sig in e.signals]) for e in records.examples])
    if not np.isfinite(x).all():
        raise ValueError("a record has invalid samples")
    return x


def _reference_s4(
    records: Records, y: np.ndarray, s: PtbxlSettings, device: torch.device, log: Log, prefix: str,
    name: str,
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """reference_s4, reference_s4_channels or reference_s4_incontext on the records (see
    benchmarks/s4.py, benchmarks/s4_incontext.py); also the chosen epoch's predictions on the
    validation and test folds. The in-context variant gets each record's patient (patient_id of
    ptbxl_database.csv, as the records' subject_id)."""
    recipe = {"reference_s4": s.reference_s4, "reference_s4_channels": s.reference_s4_channels,
              "reference_s4_incontext": s.reference_s4_incontext}[name]  # fmt: skip
    if recipe is None:
        raise ValueError(f"{name} needs the `{name}` settings")
    if isinstance(recipe, S4InContextSettings):
        _, patients = np.unique([e.origin.subject_id for e in records.examples],
                                return_inverse=True)  # fmt: skip
        return train_and_score_incontext(
            _records_array(records),
            y,
            records.folds,
            patients,
            TRAIN,
            VALIDATION,
            TEST,
            float(s.rate),
            recipe,
            s.seed,
            device,
            log,
            prefix,
            macro_auc,
        )
    return train_and_score_s4(
        _records_array(records),
        y,
        records.folds,
        TRAIN,
        VALIDATION,
        TEST,
        float(s.rate),
        recipe,
        s.seed,
        device,
        log,
        prefix,
        macro_auc,
    )


def _reference(
    records: Records, y: np.ndarray, s: PtbxlSettings, device: torch.device, log: Log, prefix: str
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:
    """reference_xresnet1d101 on the records: the leads in the order of the files (I, II, III,
    aVR, aVL, aVF, V1-V6, as the paper's wfdb.rdsamp), in physical units (mV)."""
    if s.reference is None:
        raise ValueError("reference_xresnet1d101 needs the `reference` settings")
    lengths = {sig.values.size for e in records.examples for sig in e.signals}
    leads = {len(e.signals) for e in records.examples}
    if len(lengths) != 1 or len(leads) != 1:
        raise ValueError(f"records of different lengths {lengths} or leads {leads}")
    x = np.stack([np.stack([sig.values for sig in e.signals]) for e in records.examples])
    if not np.isfinite(x).all():
        raise ValueError("a record has invalid samples")
    return train_and_score(
        x,
        y,
        records.folds,
        TRAIN,
        VALIDATION,
        TEST,
        float(s.rate),
        s.reference,
        s.seed,
        device,
        log,
        prefix,
        macro_auc,
    )

"""Benchmark: heart rate from the wrist PPG of PPG-DaLiA, by subject (leave one out).

The protocol of the dataset's authors (Reiss et al. 2019, "Deep PPG: large-scale heart
rate estimation with convolutional neural networks", Sensors 19(14):3079, sections 3.2 and
4.1, checked on 2026-10-29) and of its readme (PPG_FieldStudy_readme.pdf, I.2 and III.3):

- 15 subjects, each `S<n>.pkl` ("synchronised and labelled data"); the label of window i is
  the ECG-based heart rate of the 8 s from 2i s ("window length: 8 seconds, window shift:
  2 seconds"); the signals are cut with the same windows: wrist BVP (PPG) at 64 Hz and the
  wrist 3-axis accelerometer at 32 Hz;
- leave-one-subject-out ("LOSO"): each subject is the test subject once; the model is
  trained on the others, of which some are kept for validation, and "for testing, we chose
  the model with the lowest validation loss" (here: the epoch with the lowest validation
  MAE). The validation subjects of each fold are given by the run's settings (`folds`);
  the paper draws them at random (2 of 9 on IEEE_Test) and does not give them for DaLiA;
- the metric: the mean absolute error in beats per minute over the windows of a subject
  (their equation 1), and the mean of the 15 subjects' MAEs (their "All" column, e.g. the
  CNN ensemble's 7.65 bpm).

Inputs: `ppg+acc` gives the model the BVP and the accelerometer, as Deep PPG and the later
supervised methods do (PULSE, KID-PPG, EnhancePPG); `ppg` the BVP alone, as the PPG
foundation models are reported (PaPaGei, MOMENT; not on this split). Signals in physical
units at their native rates, unfiltered. The target is standardised with the training
subjects' mean and standard deviation; the loss is the absolute error (EnhancePPG: "trained
to minimize the MAE").

Methods, as in the PTB-XL benchmark: `fine_tuning` (encoder and a linear head on the mean
feature, every weight trained, layer-dependent learning rates), `from_scratch` (the same
from random weights, one learning rate), and `linear_probing` / `linear_probing_random`
(the encoder frozen, ridge regression on the mean feature, its L2 chosen on the validation
subjects; a diagnostic).
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
from pydantic import BaseModel, ConfigDict

from biosignal_data.read.standard import StandardDataset
from biosignal_data.storage.stores import Store
from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.batch import Batch, Packer
from biosignal_model.benchmarks import LOG_EVERY_STEPS, Log
from biosignal_model.benchmarks.ptbxl import _mean_features, parameter_groups
from biosignal_model.model import Encoder, ModelConfig

DATASET = "uci/ppg-dalia"
SUBJECTS = tuple(f"S{n}" for n in range(1, 16))
WINDOW_S, SHIFT_S = 8.0, 2.0
BVP_RATE, ACC_RATE = 64.0, 32.0


class DaliaFold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    test: list[str]
    validation: list[str]


class DaliaSettings(BaseModel):
    """Every setting of the benchmark; none has a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inputs: Literal["ppg", "ppg+acc"]
    folds: list[DaliaFold]  # each subject in one fold's test set
    rows: int  # rows of the model's max_tokens in one batch, at most
    read_threads: int
    workers: int  # data loader processes that cut and pack batches; 0: the main process
    linear_probing: bool
    linear_probing_l2: list[float]  # ridge strengths, on standardised features
    methods: list[Literal["fine_tuning", "from_scratch"]]
    fine_tuning_epochs: int
    fine_tuning_patience: int | None  # stop after this many epochs without a better MAE
    fine_tuning_learning_rate: float  # of the head; the encoder's parts get lower ones
    fine_tuning_weight_decay: float
    fine_tuning_examples_per_step: int
    fine_tuning_discriminative_factor: float
    fine_tuning_lower_blocks_fraction: float
    seed: int


@dataclass
class Subject:
    """One subject: the wrist signals in physical units and the label of each window."""

    name: str
    record: str
    bvp: np.ndarray  # (samples,) at 64 Hz
    acc: np.ndarray  # (3, samples) at 32 Hz
    heart_rate: np.ndarray  # bpm, one per window
    channels: tuple[Channel, ...]  # BVP, ACC x, y, z

    def window(self, i: int, inputs: str) -> Example:
        b = round(i * SHIFT_S * BVP_RATE)
        signals = [Signal(self.bvp[b : b + round(WINDOW_S * BVP_RATE)].astype(np.float32),
                          BVP_RATE, 0.0)]  # fmt: skip
        if inputs == "ppg+acc":
            a = round(i * SHIFT_S * ACC_RATE)
            n = round(WINDOW_S * ACC_RATE)
            signals += [
                Signal(row[a : a + n].astype(np.float32), ACC_RATE, 0.0) for row in self.acc
            ]
        channels = self.channels if inputs == "ppg+acc" else self.channels[:1]
        origin = Origin(DATASET, self.record, f"{DATASET}:{self.name}", i * SHIFT_S, channels)
        return Example(tuple(signals), origin)


def load(store: Store, s: DaliaSettings, log: Log) -> list[Subject]:
    """Every subject's wrist BVP and accelerometer, and the heart rate of each window."""
    dataset = StandardDataset.open(store, DATASET, "latest")
    records = {n: f"PPG_FieldStudy/{n}/{n}" for n in SUBJECTS}
    started = time.monotonic()
    stored = dataset.read_many(
        [(r, a) for r in records.values() for a in ("64Hz", "32Hz")],
        threads=s.read_threads, physical=True,
    )  # fmt: skip
    labels: dict[str, list[tuple[float, float]]] = {r: [] for r in records.values()}
    for e in dataset.events.to_pylist():
        if e["annotator"] == "label" and e["recording_id"] in labels:
            labels[e["recording_id"]].append((float(e["onset_s"]), float(e["text"])))
    names = {(c["recording_id"], c["array"], c["index"]): c for c in dataset.channels.to_pylist()}
    subjects = []
    for name, record in records.items():
        windows = sorted(labels[record])
        if [t for t, _ in windows] != [i * SHIFT_S for i in range(len(windows))]:
            raise ValueError(f"{DATASET}: {record}: the labels are not one per {SHIFT_S:g} s")
        bvp, acc = stored[record, "64Hz"], stored[record, "32Hz"]
        if bvp.shape[0] != 1 or acc.shape[0] != 3:
            raise ValueError(
                f"{DATASET}: {record}: {bvp.shape[0]} BVP, {acc.shape[0]} ACC channels"
            )
        last = (len(windows) - 1) * SHIFT_S + WINDOW_S
        if last > bvp.shape[1] / BVP_RATE or last > acc.shape[1] / ACC_RATE:
            raise ValueError(f"{DATASET}: {record}: the last window ends after the signals")
        channels = tuple(
            Channel(array, i, rate, 1.0, 0.0, None, names[record, array, i]["name"], kind,
                    names[record, array, i]["unit"])
            for array, i, rate, kind in [("64Hz", 0, BVP_RATE, "ppg")]
            + [("32Hz", k, ACC_RATE, "accel") for k in range(3)]
        )  # fmt: skip
        hr = np.array([v for _, v in windows])
        subjects.append(Subject(name, record, bvp[0], acc, hr, channels))
    log({"load/subjects": float(len(subjects)), "load/seconds": time.monotonic() - started})
    return subjects


def _packer(model: ModelConfig, max_tokens: int, rows: int) -> Packer:
    return Packer(
        datasets=[DATASET], max_tokens=max_tokens, samples_per_token=model.samples_per_token,
        max_channels=model.max_channels, rows=rows, mask_ratio=0.0, seed=0,
    )  # fmt: skip


Item = tuple[int, int]  # (subject index, window)


def _items(subjects: list[Subject], names: Iterable[str]) -> list[Item]:
    wanted = set(names)
    return [(k, i) for k, sub in enumerate(subjects) if sub.name in wanted
            for i in range(len(sub.heart_rate))]  # fmt: skip


def _per_batch(subjects: list[Subject], packer: Packer, s: DaliaSettings) -> int:
    """Windows that fit one batch (every window has the same size)."""
    tokens = len(packer.tokens(subjects[0].window(0, s.inputs)).slot)
    return (packer.max_tokens // tokens) * s.rows


class _Windows(torch.utils.data.Dataset[tuple[Batch, list[tuple[int, int]]]]):
    """Batch k packs the windows `groups[k]`, slots seeded by (seed, k)."""

    def __init__(
        self, subjects: list[Subject], groups: list[list[Item]], packer: Packer, inputs: str,
        seed: str,
    ) -> None:  # fmt: skip
        self.subjects, self.groups, self.packer, self.inputs, self.seed = (
            subjects, groups, packer, inputs, seed
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, k: int) -> tuple[Batch, list[tuple[int, int]]]:
        rng = random.Random(f"{self.seed}/{k}")  # noqa: S311 - not secret
        self.packer.reseed(rng.getrandbits(63))
        chosen = [self.subjects[j].window(i, self.inputs) for j, i in self.groups[k]]
        return self.packer.pack(chosen, trim=True)


def _as_is(item: Any) -> Any:
    return item


def _loader(
    subjects: list[Subject], groups: list[list[Item]], packer: Packer, s: DaliaSettings, seed: str
) -> Iterable[tuple[Batch, list[tuple[int, int]]]]:
    return torch.utils.data.DataLoader(
        _Windows(subjects, groups, packer, s.inputs, seed), batch_size=None, shuffle=False,
        num_workers=s.workers, collate_fn=_as_is, prefetch_factor=4 if s.workers else None,
    )  # fmt: skip


def _targets(subjects: list[Subject], items: list[Item]) -> np.ndarray:
    return np.array([subjects[j].heart_rate[i] for j, i in items])


def mae_by_subject(
    subjects: list[Subject], items: list[Item], predicted: np.ndarray
) -> dict[str, float]:
    """Subject -> mean absolute error (bpm) over its windows."""
    errors: dict[str, list[float]] = {}
    for (j, i), p in zip(items, predicted, strict=True):
        errors.setdefault(subjects[j].name, []).append(abs(p - subjects[j].heart_rate[i]))
    return {name: float(np.mean(e)) for name, e in errors.items()}


def end_to_end(
    config: ModelConfig, subjects: list[Subject], fold: DaliaFold, max_tokens: int,
    s: DaliaSettings, device: torch.device, start: dict[str, torch.Tensor] | None, log: Log,
    prefix: str,
) -> dict[str, float]:  # fmt: skip
    """One fold: trained on the subjects in neither `fold.test` nor `fold.validation`; the
    epoch with the lowest validation MAE is scored on the test subjects. Returns each test
    subject's MAE (`mae/<subject>`), the validation MAE and the chosen epoch."""
    torch.manual_seed(s.seed)
    model = Encoder(config).to(device)
    if start is not None:
        model.load_state_dict(start)
    head = torch.nn.Linear(config.width, 1).to(device)
    factor = s.fine_tuning_discriminative_factor if start is not None else 1.0
    groups = parameter_groups(model, head, s.fine_tuning_learning_rate, factor,
                              s.fine_tuning_lower_blocks_fraction)  # fmt: skip
    optimizer = torch.optim.AdamW(groups, weight_decay=s.fine_tuning_weight_decay)
    packer = _packer(config, max_tokens, s.rows)
    held = set(fold.test) | set(fold.validation)
    train = _items(subjects, [n for n in SUBJECTS if n not in held])
    val, test = _items(subjects, fold.validation), _items(subjects, fold.test)
    y_train = _targets(subjects, train)
    mean, std = float(y_train.mean()), float(y_train.std())
    per_batch_eval = _per_batch(subjects, packer, s)
    per_batch = min(per_batch_eval, s.fine_tuning_examples_per_step)
    rng = random.Random(s.seed)  # noqa: S311 - not secret

    def predict(items: list[Item]) -> np.ndarray:  # in bpm
        scaled = _predict(model, head, subjects, items, packer, per_batch_eval, s, device)
        return scaled * std + mean

    best: dict[str, float] = {"validation_mae": math.inf}
    best_epoch, step = 0, 0
    for epoch in range(1, s.fine_tuning_epochs + 1):
        model.train()
        rng.shuffle(train)
        started, losses, window = time.monotonic(), [], []
        steps = [train[k : k + per_batch] for k in range(0, len(train), per_batch)]
        batches = _loader(subjects, steps, packer, s, f"{s.seed}/{epoch}")
        for ids, (batch, where) in zip(steps, batches, strict=True):
            target = torch.from_numpy((_targets(subjects, ids) - mean) / std).float().to(device)
            out = head(_mean_features(model, batch, where, device)).squeeze(-1)
            loss = torch.nn.functional.l1_loss(out, target)
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
        val_mae = float(np.mean(np.abs(predict(val) - _targets(subjects, val))))
        record = {
            f"{prefix}/epoch": float(epoch), f"{prefix}/epoch_train_loss": float(np.mean(losses)),
            f"{prefix}/epoch_validation_mae": val_mae,
            f"{prefix}/epoch_seconds": time.monotonic() - started,
        }  # fmt: skip
        if val_mae < best["validation_mae"]:  # the test subjects for the chosen epoch
            best_epoch = epoch
            by_subject = mae_by_subject(subjects, test, predict(test))
            best = {"validation_mae": val_mae} | {f"mae/{k}": v for k, v in by_subject.items()}
            record |= {f"{prefix}/chosen_epoch": float(epoch)}
            record |= {f"{prefix}/chosen_test_mae/{k}": v for k, v in by_subject.items()}
        log(record)
        if s.fine_tuning_patience is not None and epoch - best_epoch >= s.fine_tuning_patience:
            break
    return best | {"epoch": float(best_epoch), "epochs_run": float(epoch)}


@torch.no_grad()
def _predict(
    model: Encoder, head: torch.nn.Module, subjects: list[Subject], items: list[Item],
    packer: Packer, per_batch: int, s: DaliaSettings, device: torch.device,
) -> np.ndarray:  # fmt: skip
    """The head's standardised output for each window."""
    model.eval()
    groups = [items[k : k + per_batch] for k in range(0, len(items), per_batch)]
    out = [
        head(_mean_features(model, batch, where, device)).squeeze(-1).float().cpu()
        for batch, where in _loader(subjects, groups, packer, s, f"{s.seed}/predict")
    ]
    return torch.cat(out).numpy() if out else np.zeros(0)


@torch.no_grad()
def embed(
    model: Encoder, subjects: list[Subject], packer: Packer, s: DaliaSettings,
    device: torch.device, log: Log, name: str,
) -> tuple[list[Item], np.ndarray]:  # fmt: skip
    """The mean feature of every window of every subject."""
    model.eval()
    items = _items(subjects, SUBJECTS)
    per_batch = _per_batch(subjects, packer, s)
    groups = [items[k : k + per_batch] for k in range(0, len(items), per_batch)]
    parts, started = [], time.monotonic()
    for k, (batch, where) in enumerate(_loader(subjects, groups, packer, s, f"{s.seed}/embed"), 1):
        parts.append(_mean_features(model, batch, where, device).cpu())
        if k % max(1, len(groups) // 10) == 0 or k == len(groups):
            log({f"embed/{name}/batches": float(k), f"embed/{name}/of": float(len(groups)),
                 f"embed/{name}/seconds": time.monotonic() - started})  # fmt: skip
    return items, torch.cat(parts).numpy()


def _ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Weights [d + 1] of ridge regression (bias not penalized), in closed form."""
    xb = np.hstack([x, np.ones((len(x), 1))])
    penalty = l2 * np.eye(xb.shape[1])
    penalty[-1, -1] = 0.0
    weights: np.ndarray = np.linalg.solve(xb.T @ xb + penalty, xb.T @ y)
    return weights


def linear_probing(
    subjects: list[Subject], items: list[Item], features: np.ndarray, fold: DaliaFold,
    s: DaliaSettings, log: Log, prefix: str,
) -> dict[str, float]:  # fmt: skip
    """One fold: ridge regression on the training subjects' standardised features, its L2
    chosen by the validation MAE; each test subject's MAE."""
    name_of = np.array([subjects[j].name for j, _ in items])
    y = _targets(subjects, items)
    held = set(fold.test) | set(fold.validation)
    train = ~np.isin(name_of, list(held))
    val, test = np.isin(name_of, fold.validation), np.isin(name_of, fold.test)
    mean, std = features[train].mean(0), features[train].std(0) + 1e-8
    x = (features - mean) / std
    best_l2, best_mae = s.linear_probing_l2[0], math.inf
    for l2 in s.linear_probing_l2:
        w = _ridge(x[train], y[train], l2)
        mae = float(np.mean(np.abs(x[val] @ w[:-1] + w[-1] - y[val])))
        log({f"{prefix}/l2": l2, f"{prefix}/validation_mae": mae})
        if mae < best_mae:
            best_l2, best_mae = l2, mae
    w = _ridge(x[train], y[train], best_l2)
    test_items = [it for it, t in zip(items, test, strict=True) if t]
    by_subject = mae_by_subject(subjects, test_items, x[test] @ w[:-1] + w[-1])
    return {f"mae/{k}": v for k, v in by_subject.items()} | {
        "l2": best_l2, "validation_mae": best_mae,
    }  # fmt: skip


def _summary(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """The folds' results in one dict: each test subject's MAE, their mean (the paper's
    "All"), and the mean validation MAE and chosen epoch of the folds."""
    out: dict[str, float] = {}
    for scored in per_fold:
        out |= {k: v for k, v in scored.items() if k.startswith("mae/")}
    out["mae"] = float(np.mean([v for k, v in out.items() if k.startswith("mae/")]))
    for key in ("validation_mae", "epoch", "epochs_run", "l2"):
        values = [f[key] for f in per_fold if key in f]
        if values:
            out[f"mean_{key}"] = float(np.mean(values))
    return out


def run(
    store: Store, checkpoint: dict[str, Any], settings: DaliaSettings, device: torch.device,
    log: Log,
) -> dict[str, float]:  # fmt: skip
    """All scores of one checkpoint, as a flat dict for logging: `<method>/<name>`."""
    from biosignal_model.train import TrainConfig  # noqa: PLC0415 - avoids a cycle

    tested = [n for f in settings.folds for n in f.test]
    if sorted(tested) != sorted(SUBJECTS):
        raise ValueError(f"the folds' test subjects must be each of {SUBJECTS} once: {tested}")
    for f in settings.folds:
        if set(f.test) & set(f.validation):
            raise ValueError(f"a fold has test subjects in its validation set: {f}")
    train_config = TrainConfig.from_checkpoint(checkpoint["config"])
    config = train_config.model_settings()
    max_tokens = train_config.tokens.max_tokens
    subjects = load(store, settings, log)
    packer = _packer(config, max_tokens, settings.rows)
    results: dict[str, float] = {
        "subjects": float(len(subjects)),
        "windows": float(sum(len(sub.heart_rate) for sub in subjects)),
    }
    if settings.linear_probing:
        pretrained = Encoder(config).to(device)
        pretrained.load_state_dict(checkpoint["model"])
        torch.manual_seed(settings.seed)
        untrained = Encoder(config).to(device)
        for name, model in (("linear_probing", pretrained), ("linear_probing_random", untrained)):
            items, features = embed(model, subjects, packer, settings, device, log, name)
            per_fold = [
                linear_probing(subjects, items, features, f, settings, log, f"{name}/fold{k}")
                for k, f in enumerate(settings.folds)
            ]
            summary = _summary(per_fold)
            results |= {f"{name}/{k}": v for k, v in summary.items()}
            log({f"{name}/mae": summary["mae"]})
    starts = {"fine_tuning": checkpoint["model"], "from_scratch": None}
    for name in settings.methods:
        per_fold = [
            end_to_end(config, subjects, f, max_tokens, settings, device, starts[name], log,
                       f"{name}/fold{k}")
            for k, f in enumerate(settings.folds)
        ]  # fmt: skip
        summary = _summary(per_fold)
        results |= {f"{name}/{k}": v for k, v in summary.items()}
        log({f"{name}/mae": summary["mae"]})
    return results

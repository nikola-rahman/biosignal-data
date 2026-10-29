"""Benchmark: per-minute sleep apnea detection from one ECG lead (PhysioNet apnea-ecg).

The official protocol of the PhysioNet/CinC Challenge 2000: a label (N or A) for each
minute of 70 overnight ECG recordings at 100 Hz; the learning set is a01-a20, b01-b05,
c01-c10, the test set x01-x35. The test records are held out of pretraining (index.yaml).

What is measured, for one checkpoint:

- `finetune`, the main result, the one to compare with published results: the pretrained
  encoder and a linear head, every weight trained end to end on the learning set with the
  labels (binary cross-entropy), scored once on the test set. The published methods are
  trained on the labels too.
- `supervised`: the same, from random weights: what pretraining adds to the same model.
- `probe` and `random`, a diagnostic: the encoder frozen (pretrained, or random weights).
  Each labelled minute, with `context_s` of ECG before and after it (less at the ends of
  a record), goes through the same token encoding as in pretraining, nothing hidden; its
  embedding is the mean of the encoder's features over the tokens of the minute itself. A
  logistic regression on the embeddings is chosen by cross-validation on the learning set
  (folds by record), trained on the whole learning set, and scored once on the test set.
  It shows whether pretraining learned anything a task can use; it is not a comparison
  with the published methods, which train every weight.

Scores: accuracy, sensitivity (apnea minutes found), specificity, and the area under the
ROC curve, over all test minutes.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict
from scipy.stats import rankdata

from biosignal_data.read.standard import StandardDataset
from biosignal_data.storage.stores import Store
from biosignal_data.training.loader import Channel, Example, Origin, Signal
from biosignal_model.batch import Packer
from biosignal_model.benchmarks import LOG_EVERY_STEPS, Log
from biosignal_model.model import Encoder, ModelConfig

DATASET = "physionet/apnea-ecg"
LEARNING = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)
TEST = [f"x{i:02d}" for i in range(1, 36)]
MINUTE_S = 60.0


class ApneaSettings(BaseModel):
    """Every setting of the benchmark; none has a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context_s: float  # ECG before and after the labelled minute
    rows: int  # rows of the model's max_tokens in one batch
    probe_l2: list[float]  # the L2 strengths the cross-validation chooses from
    probe_folds: int
    probe_steps: int  # LBFGS steps of one logistic regression
    # Fine-tuning and supervised from zero use the same training settings.
    supervised_epochs: int
    supervised_learning_rate: float
    supervised_weight_decay: float
    supervised_examples_per_step: int
    seed: int


@dataclass
class Minutes:
    """The labelled minutes of the records of one set."""

    records: list[str]
    examples: list[Example]
    labels: np.ndarray  # 1 = apnea
    record_of: np.ndarray  # index into `records`, per minute
    in_minute: list[tuple[float, float]]  # the minute, in the example's own time


def minutes(dataset: StandardDataset, records: list[str], context_s: float) -> Minutes:
    """The labelled minutes of `records`, each as an example with its context."""
    events = dataset.events.to_pylist()
    labels: dict[str, list[tuple[float, str]]] = {r: [] for r in records}
    for e in events:
        if e["annotator"] == "apn" and e["recording_id"] in labels:
            labels[e["recording_id"]].append((e["onset_s"], e["label"]))
    examples, values, record_of, spans = [], [], [], []
    for index, record in enumerate(records):
        if not labels[record]:
            raise ValueError(f"{DATASET}: {record} has no apn labels in the standard layer")
        signal = _ecg(dataset, record)
        rate = 100.0
        channel = Channel("100Hz", 0, rate, None, None, None, "ECG", "ecg", "mV")
        for onset, label in sorted(labels[record]):
            if label not in ("N", "A"):
                raise ValueError(f"{DATASET}: {record} at {onset} s has label {label!r}")
            begin = max(0.0, onset - context_s)
            end = min(signal.size / rate, onset + MINUTE_S + context_s)
            first, last = math.ceil(round(begin * rate, 9)), math.ceil(round(end * rate, 9))
            piece = signal[first:last].astype(np.float32)
            examples.append(
                Example(
                    (Signal(piece, rate, first / rate - begin),),
                    Origin(DATASET, record, f"{DATASET}:{record}", begin, (channel,)),
                )
            )
            values.append(1 if label == "A" else 0)
            record_of.append(index)
            spans.append((onset - begin, onset - begin + MINUTE_S))
    return Minutes(records, examples, np.array(values), np.array(record_of), spans)


def _ecg(dataset: StandardDataset, record: str) -> np.ndarray:
    """The whole ECG of a record (its one channel at 100 Hz), in physical values."""
    values: np.ndarray = dataset.read(record, "100Hz", physical=True)[0]
    return values


def _packer(model: ModelConfig, max_tokens: int, rows: int) -> Packer:
    return Packer(
        datasets=[DATASET], max_tokens=max_tokens, samples_per_token=model.samples_per_token,
        max_channels=model.max_channels, rows=rows, mask_ratio=0.0, seed=0,
    )  # fmt: skip


def _chunks(m: Minutes, packer: Packer, rows: int) -> list[list[int]]:
    """Minutes in groups that fit one batch."""
    groups: list[list[int]] = []
    current: list[int] = []
    row, used = 0, 0
    for i, example in enumerate(m.examples):
        n = len(packer.tokens(example).slot)
        if used + n > packer.max_tokens:
            row, used = row + 1, 0
        if row == rows:
            groups.append(current)
            current, row, used = [], 0, 0
        current.append(i)
        used += n
    if current:
        groups.append(current)
    return groups


def _pooled(
    model: Encoder, m: Minutes, ids: list[int], packer: Packer, device: torch.device
) -> torch.Tensor:
    """The mean feature over the tokens of each labelled minute, [len(ids), width]."""
    batch, where = packer.pack([m.examples[i] for i in ids], trim=True)
    x = batch.inputs.to(device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = model.features(x).float()
    out = []
    for i, (row, document) in zip(ids, where, strict=True):
        start, stop = m.in_minute[i]
        t = x.time_s[row]
        keep = (x.document[row] == document) & (t >= start) & (t < stop)
        out.append(features[row][keep].mean(0))
    return torch.stack(out)


@torch.no_grad()
def embed(
    model: Encoder, m: Minutes, max_tokens: int, rows: int, device: torch.device, log: Log,
    name: str,
) -> np.ndarray:  # fmt: skip
    """The mean feature of every labelled minute; progress as `embed/<name>/...`."""
    model.eval()
    packer = _packer(model.config, max_tokens, rows)
    chunks = _chunks(m, packer, rows)
    parts, started = [], time.monotonic()
    for k, ids in enumerate(chunks, 1):
        parts.append(_pooled(model, m, ids, packer, device).cpu())
        if k % max(1, len(chunks) // 10) == 0 or k == len(chunks):
            log({f"embed/{name}/batches": float(k), f"embed/{name}/of": float(len(chunks)),
                 f"embed/{name}/seconds": time.monotonic() - started})  # fmt: skip
    return torch.cat(parts).numpy()


def _fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float, steps: int) -> torch.Tensor:
    """Weights [d + 1] of a logistic regression with an L2 penalty (bias not penalized)."""
    w = torch.zeros(x.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = x @ w[:-1] + w[-1]
        loss = F.binary_cross_entropy_with_logits(logits, y) + l2 * (w[:-1] ** 2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return w.detach()


def _scores(labels: np.ndarray, p: np.ndarray) -> dict[str, float]:
    predicted = p >= 0.5
    apnea, normal = labels == 1, labels == 0
    ranks = rankdata(p)  # ties get their average rank
    auc = (ranks[apnea].sum() - apnea.sum() * (apnea.sum() + 1) / 2) / (apnea.sum() * normal.sum())
    return {
        "accuracy": float((predicted == apnea).mean()),
        "sensitivity": float(predicted[apnea].mean()),
        "specificity": float((~predicted[normal]).mean()),
        "auc": float(auc),
    }


def probe(
    learn: np.ndarray, learn_labels: np.ndarray, groups: np.ndarray,
    test: np.ndarray, test_labels: np.ndarray, s: ApneaSettings, log: Log, prefix: str,
) -> dict[str, float]:  # fmt: skip
    """Logistic regression on standardized features: L2 by cross-validation on the
    learning set (folds by record), then one fit on all of it and one score on test. Each
    L2 tried is logged with its cross-validated accuracy, under `prefix`."""
    mean, std = learn.mean(0), learn.std(0) + 1e-8
    xl = torch.from_numpy((learn - mean) / std).double()
    xt = torch.from_numpy((test - mean) / std).double()
    yl = torch.from_numpy(learn_labels).double()
    records = [int(r) for r in np.unique(groups)]
    random.Random(s.seed).shuffle(records)  # noqa: S311 - not secret
    folds = [set(records[k :: s.probe_folds]) for k in range(s.probe_folds)]
    best, best_accuracy = s.probe_l2[0], -1.0
    for l2 in s.probe_l2:
        correct = 0
        for fold in folds:
            held = np.isin(groups, list(fold))
            w = _fit_logistic(xl[~held], yl[~held], l2, s.probe_steps)
            p = torch.sigmoid(xl[held] @ w[:-1] + w[-1]).numpy()
            correct += int(((p >= 0.5) == (learn_labels[held] == 1)).sum())
        log({f"{prefix}/l2": l2, f"{prefix}/cv_accuracy": correct / len(learn_labels)})
        if correct / len(learn_labels) > best_accuracy:
            best, best_accuracy = l2, correct / len(learn_labels)
    w = _fit_logistic(xl, yl, best, s.probe_steps)
    p = torch.sigmoid(xt @ w[:-1] + w[-1]).numpy()
    return _scores(test_labels, p) | {"l2": best, "cv_accuracy": best_accuracy}


def end_to_end(
    config: ModelConfig, learn: Minutes, test: Minutes, max_tokens: int,
    s: ApneaSettings, device: torch.device, start: dict[str, torch.Tensor] | None, log: Log,
    prefix: str,
) -> dict[str, float]:  # fmt: skip
    """The encoder and a linear head, every weight trained on the learning set with the
    labels, scored once on the test set. `start`: the pretrained weights (fine-tuning),
    or None for random weights (supervised from zero); the rest is the same. Logged under
    `prefix`: the training loss every LOG_EVERY_STEPS steps, and per epoch its mean, time
    and speed (the protocol has no validation set)."""
    torch.manual_seed(s.seed)
    model = Encoder(config).to(device)
    if start is not None:
        model.load_state_dict(start)
    head = torch.nn.Linear(config.width, 1).to(device)
    parameters = [*model.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=s.supervised_learning_rate, weight_decay=s.supervised_weight_decay
    )
    packer = _packer(config, max_tokens, s.rows)
    order = list(range(len(learn.examples)))
    rng = random.Random(s.seed)  # noqa: S311 - not secret
    step = 0
    for epoch in range(1, s.supervised_epochs + 1):
        model.train()
        rng.shuffle(order)
        started, losses, window = time.monotonic(), [], []
        for k in range(0, len(order), s.supervised_examples_per_step):
            ids = order[k : k + s.supervised_examples_per_step]
            pooled = _pooled(model, learn, ids, packer, device)
            logits = head(pooled).squeeze(-1)
            y = torch.from_numpy(learn.labels[ids]).float().to(device)
            loss = F.binary_cross_entropy_with_logits(logits, y)
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
        log({f"{prefix}/epoch": float(epoch), f"{prefix}/epoch_train_loss": float(np.mean(losses)),
             f"{prefix}/epoch_seconds": seconds,
             f"{prefix}/examples_per_s": len(order) / seconds})  # fmt: skip
    model.eval()
    p = []
    with torch.no_grad():
        for ids in _chunks(test, packer, s.rows):
            p.append(torch.sigmoid(head(_pooled(model, test, ids, packer, device))).squeeze(-1))
    return _scores(test.labels, torch.cat(p).cpu().numpy())


def run(
    store: Store,
    checkpoint: dict[str, Any],
    settings: ApneaSettings,
    device: torch.device,
    log: Log,
) -> dict[str, float]:
    """All scores of one checkpoint, as a flat dict for logging."""
    from biosignal_model.train import TrainConfig  # noqa: PLC0415 - avoids a cycle

    train_config = TrainConfig.from_checkpoint(checkpoint["config"])
    config = train_config.model_settings()
    max_tokens = train_config.tokens.max_tokens
    dataset = StandardDataset.open(store, DATASET, "latest")
    learn = minutes(dataset, LEARNING, settings.context_s)
    test = minutes(dataset, TEST, settings.context_s)
    results: dict[str, float] = {
        "minutes/learning": float(len(learn.labels)),
        "minutes/test": float(len(test.labels)),
        "minutes/test_apnea_fraction": float(test.labels.mean()),
    }
    pretrained = Encoder(config).to(device)
    pretrained.load_state_dict(checkpoint["model"])
    torch.manual_seed(settings.seed)
    untrained = Encoder(config).to(device)
    for name, model in (("probe", pretrained), ("random", untrained)):
        scores = probe(
            embed(model, learn, max_tokens, settings.rows, device, log, f"{name}/learning"),
            learn.labels, learn.record_of,
            embed(model, test, max_tokens, settings.rows, device, log, f"{name}/test"),
            test.labels, settings, log, name,
        )  # fmt: skip
        results |= {f"{name}/{k}": v for k, v in scores.items()}
        log({f"{name}/auc": scores["auc"]})
    for name, start in (("finetune", checkpoint["model"]), ("supervised", None)):
        scores = end_to_end(config, learn, test, max_tokens, settings, device, start, log, name)
        results |= {f"{name}/{k}": v for k, v in scores.items()}
        log({f"{name}/auc": scores["auc"]})
    return results

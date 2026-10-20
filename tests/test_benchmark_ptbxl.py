from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.benchmarks import ptbxl  # noqa: E402
from biosignal_model.model import Encoder, ModelConfig  # noqa: E402

STATEMENTS = [  # a few rows of scp_statements.csv
    {"": "NORM", "diagnostic": "1.0", "form": "", "rhythm": "", "diagnostic_class": "NORM",
     "diagnostic_subclass": "NORM"},
    {"": "IMI", "diagnostic": "1.0", "form": "", "rhythm": "", "diagnostic_class": "MI",
     "diagnostic_subclass": "IMI"},
    {"": "LVOLT", "diagnostic": "", "form": "1.0", "rhythm": "", "diagnostic_class": "",
     "diagnostic_subclass": ""},
    {"": "SR", "diagnostic": "", "form": "", "rhythm": "1.0", "diagnostic_class": "",
     "diagnostic_subclass": ""},
    {"": "AFIB", "diagnostic": "", "form": "", "rhythm": "1.0", "diagnostic_class": "",
     "diagnostic_subclass": ""},
]  # fmt: skip


def _records(codes: list[list[str]], folds: list[int]) -> ptbxl.Records:
    rng = np.random.default_rng(0)
    examples = []
    for i in range(len(codes)):
        signals = tuple(
            Signal(rng.normal(size=1000).astype(np.float32), 100.0, 0.0) for _ in range(2)
        )
        channels = tuple(
            Channel("100Hz", k, 100.0, 1.0, 0.0, None, f"L{k}", "ecg", "mV") for k in range(2)
        )
        examples.append(Example(signals, Origin(ptbxl.DATASET, f"r{i}", f"p{i}", 0.0, channels)))
    return ptbxl.Records([f"r{i}" for i in range(len(codes))], examples, np.array(folds), codes)


def test_labels_follow_the_authors_aggregation() -> None:
    # record 0: {'NORM': 100, 'LVOLT': 0, 'SR': 0}: a key counts whatever its likelihood
    records = _records([["NORM", "LVOLT", "SR"], ["IMI", "AFIB"], ["SR"]], [1, 9, 10])
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "all")
    assert classes == ["AFIB", "IMI", "LVOLT", "NORM", "SR"] and keep.all()
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "superdiagnostic")
    assert classes == ["MI", "NORM"] and keep.tolist() == [True, True, False]
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "rhythm")
    assert classes == ["AFIB", "SR"] and y.tolist() == [[0, 1], [1, 0], [0, 1]]
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "form")
    assert classes == ["LVOLT"] and keep.tolist() == [True, False, False]


def test_macro_auc_gives_ties_their_average_rank_and_skips_one_class_labels() -> None:
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p = np.array([[0.5, 0.1], [0.5, 0.2], [0.9, 0.3], [0.1, 0.4]])
    auc, n = ptbxl.macro_auc(y, p)
    # label 0: pairs (pos, neg) = (0.5,0.5) tie -> 0.5, (0.5,0.1) 1, (0.9,0.5) 1, (0.9,0.1) 1
    assert n == 1 and auc == pytest.approx(3.5 / 4)


def test_the_probe_chooses_on_fold_9_and_scores_fold_10() -> None:
    rng = np.random.default_rng(2)
    folds = np.array([1] * 300 + [9] * 100 + [10] * 100)
    y = rng.integers(0, 2, (500, 2)).astype(np.float32)
    features = rng.normal(size=(500, 6))
    features[:, 0] += 3 * y[:, 0]
    features[:, 1] += 3 * y[:, 1]
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["all"], rows=2, read_threads=1, probe_l2=[1e-3, 1.0], probe_steps=50,
        finetune_epochs=1, finetune_learning_rate=1e-3, finetune_weight_decay=0.0,
        finetune_examples_per_step=8, seed=0,
    )  # fmt: skip
    scores = ptbxl.probe(features, y, folds, settings)
    assert scores["macro_auc"] > 0.95 and scores["labels"] == 2 and scores["l2"] in (1e-3, 1.0)


def test_fine_tuning_runs_end_to_end_on_a_small_model() -> None:
    config = ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
    )  # fmt: skip
    codes = [["NORM"], ["IMI"]] * 12
    records = _records(codes, [1] * 16 + [9] * 4 + [10] * 4)
    y, _, _ = ptbxl.labels(records, STATEMENTS, "diagnostic")
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["diagnostic"], rows=2, read_threads=1, probe_l2=[1.0], probe_steps=10,
        finetune_epochs=2, finetune_learning_rate=1e-3, finetune_weight_decay=0.0,
        finetune_examples_per_step=4, seed=0,
    )  # fmt: skip
    device = torch.device("cpu")
    for start in (None, Encoder(config).state_dict()):
        scores = ptbxl.end_to_end(config, records, y, 256, settings, device, start)
        assert 0.0 <= scores["macro_auc"] <= 1.0 and scores["epoch"] in (1.0, 2.0)
    features = ptbxl.embed(Encoder(config), records.examples, 256, 2, device)
    assert features.shape == (24, 32) and np.isfinite(features).all()

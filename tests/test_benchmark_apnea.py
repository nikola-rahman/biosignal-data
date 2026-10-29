from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_model.benchmarks import apnea  # noqa: E402
from biosignal_model.model import Encoder, ModelConfig  # noqa: E402

SETTINGS = apnea.ApneaSettings(
    context_s=30.0, rows=4,
    linear_probing_l2=[1e-3, 1e-1], linear_probing_folds=2, linear_probing_steps=50,
    fine_tuning_epochs=2, fine_tuning_learning_rate=1e-3, fine_tuning_weight_decay=0.0,
    fine_tuning_examples_per_step=4, seed=0,
)  # fmt: skip


class _Dataset:
    """What `minutes` reads of a StandardDataset: events, and one 100 Hz ECG per record."""

    def __init__(self, records: dict[str, list[str]]) -> None:
        rng = np.random.default_rng(0)
        self.ecg = {r: rng.normal(size=len(labels) * 6000) for r, labels in records.items()}
        rows = [
            {"recording_id": r, "annotator": "apn", "onset_s": 60.0 * i, "label": label}
            for r, labels in records.items()
            for i, label in enumerate(labels)
        ]
        self.events = SimpleNamespace(to_pylist=lambda: rows)

    def read(self, record: str, array: str, *, physical: bool) -> np.ndarray:
        assert (array, physical) == ("100Hz", True)
        return self.ecg[record][None]


def test_scores_match_hand_counts() -> None:
    labels = np.array([1, 1, 0, 0, 1])
    p = np.array([0.9, 0.4, 0.2, 0.6, 0.8])
    s = apnea._scores(labels, p)
    assert s["accuracy"] == pytest.approx(3 / 5)
    assert s["sensitivity"] == pytest.approx(2 / 3)
    assert s["specificity"] == pytest.approx(1 / 2)
    assert s["auc"] == pytest.approx(5 / 6)  # 5 of the 6 apnea-normal pairs in order


def test_minutes_carry_their_context_and_label() -> None:
    m = apnea.minutes(_Dataset({"a01": ["N", "A", "A"]}), ["a01"], context_s=30.0)  # type: ignore[arg-type]
    assert m.labels.tolist() == [0, 1, 1]
    first, middle, last = (e.signals[0].values.size for e in m.examples)
    assert (first, middle, last) == (9000, 12000, 9000)  # no context past the ends
    assert m.in_minute == [(0.0, 60.0), (30.0, 90.0), (30.0, 90.0)]
    with pytest.raises(ValueError, match="no apn labels"):
        apnea.minutes(_Dataset({"x01": []}), ["x01"], context_s=30.0)  # type: ignore[arg-type]


def test_the_probe_finds_a_separable_feature() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    x = rng.normal(size=(400, 8))
    x[:, 3] += 4 * y  # one feature tells the classes apart
    groups = np.arange(400) // 50
    logged: list[dict[str, float]] = []
    s = apnea.linear_probing(
        x[:300], y[:300], groups[:300], x[300:], y[300:], SETTINGS, logged.append, "p"
    )
    assert [r["p/l2"] for r in logged] == SETTINGS.linear_probing_l2  # every L2 tried is logged
    best = ((x[300:, 3] > 2) == (y[300:] == 1)).mean()  # the Bayes rule on these minutes
    assert s["accuracy"] >= best - 0.03 and s["l2"] in SETTINGS.linear_probing_l2


def test_the_benchmark_runs_end_to_end_on_a_small_model() -> None:
    config = ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
    )  # fmt: skip
    data = _Dataset({"a01": ["N", "A"] * 4, "x01": ["A", "N"] * 3})
    learn = apnea.minutes(data, ["a01"], SETTINGS.context_s)  # type: ignore[arg-type]
    test = apnea.minutes(data, ["x01"], SETTINGS.context_s)  # type: ignore[arg-type]
    device = torch.device("cpu")
    logged: list[dict[str, float]] = []
    features = apnea.embed(Encoder(config), learn, 512, SETTINGS.rows, device, logged.append, "e")
    assert features.shape == (8, 32) and np.isfinite(features).all()
    assert logged[-1]["embed/e/batches"] == logged[-1]["embed/e/of"]  # the last batch is logged
    for start in (None, Encoder(config).state_dict()):  # from scratch, and fine-tuning
        logged.clear()
        scores = apnea.end_to_end(
            config, learn, test, 512, SETTINGS, device, start, logged.append, "f"
        )
        assert set(scores) == {"accuracy", "sensitivity", "specificity", "auc"}
        epochs = [r for r in logged if "f/epoch" in r]
        assert [r["f/epoch"] for r in epochs] == list(range(1, SETTINGS.fine_tuning_epochs + 1))
        assert all(np.isfinite(r["f/epoch_train_loss"]) for r in epochs)

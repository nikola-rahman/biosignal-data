from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel  # noqa: E402
from biosignal_model.benchmarks import isruc  # noqa: E402
from biosignal_model.model import Encoder, ModelConfig  # noqa: E402

CONFIG = ModelConfig(
    samples_per_token=50, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
    min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
)  # fmt: skip


def _settings(**changes: object) -> isruc.IsrucSettings:
    base = dict(
        channels="all", sequence_epochs=2, rows=4, read_threads=1, workers=0,
        linear_probing=True, linear_probing_l2=[1e-3, 1.0], linear_probing_steps=10,
        methods=["fine_tuning", "from_scratch"], fine_tuning_epochs=2, fine_tuning_patience=None,
        fine_tuning_encoder_learning_rate=1e-4, fine_tuning_head_learning_rate=1e-3,
        fine_tuning_min_learning_rate=1e-6, fine_tuning_weight_decay=5e-2,
        fine_tuning_sequences_per_step=3, fine_tuning_label_smoothing=0.1,
        fine_tuning_grad_clip=1.0, head_width=16, head_layers=1, head_heads=2,
        head_feed_forward=32, head_dropout=0.1, seed=0,
    )  # fmt: skip
    return isruc.IsrucSettings.model_validate(base | changes)


def _night(subject: int, epochs: int, rng: np.random.Generator) -> isruc.Night:
    """Two channels: EEG at 200 Hz (int16, scaled) and SpO2 at 12.5 Hz."""
    channels = [
        Channel("200Hz", 0, 200.0, 0.1, 0.0, None, "C4-A1", "eeg", "uV"),
        Channel("12.5Hz", 0, 12.5, 1.0, 0.0, None, "SaO2", "other", "%"),
    ]
    samples = [rng.integers(-500, 500, epochs * 6000).astype(np.int16),
               rng.integers(90, 100, epochs * 375).astype(np.int16)]  # fmt: skip
    return isruc.Night(subject, f"{subject}/{subject}", channels, samples,
                       rng.integers(0, 5, epochs), 2)  # fmt: skip


def test_stages_map_as_cbramod_and_other_values_are_refused() -> None:
    assert isruc.stages("0\n1\n\n2\n3\n5\n", "1/1").tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="stage '4'"):
        isruc.stages("0\n4\n", "1/1")


def test_scores_are_sklearns_balanced_accuracy_kappa_and_weighted_f1() -> None:
    truth, predicted = np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 2])
    s = isruc.scores(truth, predicted)
    assert s["balanced_accuracy"] == pytest.approx((0.5 + 1 + 1) / 3)
    assert s["kappa"] == pytest.approx((0.8 - 9 / 25) / (1 - 9 / 25))
    assert s["weighted_f1"] == pytest.approx((2 * 2 / 3 + 2 * 0.8 + 1) / 5)


def test_an_epoch_is_30_s_of_every_channel_in_physical_units() -> None:
    night = _night(1, 5, np.random.default_rng(0))
    example = night.epoch(2)
    eeg, spo2 = example.signals
    assert len(eeg.values) == 6000 and len(spo2.values) == 375
    np.testing.assert_allclose(eeg.values, night.samples[0][12000:18000] * 0.1, rtol=1e-6)
    assert night.sequences == 2  # 5 epochs: two whole sequences of 2, the last epoch unused


def test_fine_tuning_and_linear_probing_run_on_a_small_model() -> None:
    rng = np.random.default_rng(1)
    nights = [_night(n, 6, rng) for n in (1, 2, 81, 91)]  # train, train, validation, test
    s = _settings()
    logged: list[dict[str, float]] = []
    for start in (None, Encoder(CONFIG).state_dict()):
        logged.clear()
        scored = isruc.end_to_end(CONFIG, nights, 512, s, torch.device("cpu"), start,
                                  logged.append, "f")  # fmt: skip
        assert 0.0 <= scored["balanced_accuracy"] <= 1.0 and scored["epoch"] in (1.0, 2.0)
        assert [r["f/epoch"] for r in logged if "f/epoch" in r] == [1.0, 2.0]
    packer = isruc._packer(CONFIG, 512, s.rows)
    per_pass = isruc._per_pass(nights, packer, s)
    features, labels, subjects = isruc.embed(
        Encoder(CONFIG), nights, packer, per_pass, s, torch.device("cpu"), logged.append, "e"
    )
    assert features.shape == (4 * 6, 32) and subjects.tolist().count(91) == 6
    scored = isruc.linear_probing(features, labels, subjects, s, logged.append, "p",
                                  torch.device("cpu"))  # fmt: skip
    assert scored["l2"] in (1e-3, 1.0) and -1.0 <= scored["kappa"] <= 1.0


def test_cbramods_columns_2_to_7_come_from_the_files_channel_order() -> None:
    labels = ["LOC-A2", "ROC-A1", "C3-A2", "O1-A2", "C4-A1", "O2-A1", "X1", "X2", "SaO2"]
    header = " " * 252 + f"{len(labels):<4}" + "".join(f"{x:<16}" for x in labels) + "x" * 80
    assert isruc.edf_labels(header) == labels
    assert isruc.edf_labels(header)[isruc.CBRAMOD_COLUMNS] == labels[2:8]  # subject 8's layout

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel  # noqa: E402
from biosignal_model.batch import SamplesTokenizer  # noqa: E402
from biosignal_model.benchmarks import dalia  # noqa: E402
from biosignal_model.model import Encoder, ModelConfig  # noqa: E402

CONFIG = ModelConfig(
    samples_per_token=32, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
    min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
    tokenizer=SamplesTokenizer(kind="samples"), dropout=0.0, drop_path=0.0,
)  # fmt: skip
CHANNELS = (
    Channel("64Hz", 0, 64.0, 1.0, 0.0, None, "BVP", "ppg", ""),
    *(Channel("32Hz", k, 32.0, 1.0, 0.0, None, f"ACC_{a}", "accel", "1/64 g")
      for k, a in enumerate("xyz")),
)  # fmt: skip


def _settings(**changes: object) -> dalia.DaliaSettings:
    base: dict[str, object] = {
        "inputs": "ppg+acc",
        "folds": [{"test": ["S1"], "validation": ["S2"]}],
        "rows": 2,
        "read_threads": 1,
        "workers": 0,
        "linear_probing": True,
        "linear_probing_l2": [1e-3, 10.0],
        "methods": ["fine_tuning", "from_scratch"],
        "fine_tuning_epochs": 2,
        "fine_tuning_patience": 1,
        "fine_tuning_learning_rate": 1e-3,
        "fine_tuning_weight_decay": 1e-3,
        "fine_tuning_examples_per_step": 8,
        "fine_tuning_discriminative_factor": 0.1,
        "fine_tuning_lower_blocks_fraction": 0.25,
        "seed": 0,
    }
    return dalia.DaliaSettings.model_validate(base | changes)


def _subject(name: str, seconds: int, rng: np.random.Generator) -> dalia.Subject:
    windows = int((seconds - 8) // 2 + 1)
    return dalia.Subject(name, f"PPG_FieldStudy/{name}/{name}", rng.normal(size=seconds * 64),
                         rng.normal(size=(3, seconds * 32)), rng.uniform(60, 120, windows),
                         CHANNELS)  # fmt: skip


def test_a_window_is_8_s_from_2i_s_with_or_without_the_accelerometer() -> None:
    sub = _subject("S1", 20, np.random.default_rng(0))
    example = sub.window(3, "ppg+acc")
    assert [len(x.values) for x in example.signals] == [512, 256, 256, 256]
    np.testing.assert_allclose(example.signals[0].values, sub.bvp[384:896], rtol=1e-6)
    np.testing.assert_allclose(example.signals[1].values, sub.acc[0, 192:448], rtol=1e-6)
    assert len(sub.window(3, "ppg").signals) == 1 and example.origin.start_s == 6.0


def test_the_score_is_the_mean_of_the_subjects_maes() -> None:
    per_fold = [{"mae/S1": 2.0, "validation_mae": 3.0, "epoch": 4.0},
                {"mae/S2": 6.0, "validation_mae": 5.0, "epoch": 2.0}]  # fmt: skip
    out = dalia._summary(per_fold)
    assert out["mae"] == 4.0 and out["mean_validation_mae"] == 4.0 and out["mean_epoch"] == 3.0


def test_fine_tuning_and_linear_probing_run_on_a_small_model() -> None:
    rng = np.random.default_rng(1)
    subjects = [_subject(n, 30, rng) for n in ("S1", "S2", "S3", "S4")]
    s = _settings()
    fold = s.folds[0]
    logged: list[dict[str, float]] = []
    for start in (None, Encoder(CONFIG).state_dict()):
        scored = dalia.end_to_end(CONFIG, subjects, fold, 256, s, torch.device("cpu"), start,
                                  logged.append, "f")  # fmt: skip
        assert {k for k in scored if k.startswith("mae/")} == {"mae/S1"}
        assert scored["epoch"] >= 1.0 and scored["mae/S1"] >= 0.0
    packer = dalia._packer(CONFIG, 256, s.rows)
    items, features = dalia.embed(Encoder(CONFIG), subjects, packer, s, torch.device("cpu"),
                                  logged.append, "e")  # fmt: skip
    assert features.shape == (4 * 12, 32)
    # a feature that is the heart rate itself: ridge finds it, the MAE is about 0
    features = np.hstack([features, dalia._targets(subjects, items)[:, None]])
    scored = dalia.linear_probing(subjects, items, features, fold, s, logged.append, "p")
    assert scored["mae/S1"] < 0.5 and scored["l2"] == 1e-3

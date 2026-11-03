"""The scale table and the value transform (biosignal_model.scales)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biosignal_data.storage.stores import LocalStore, StoreFactory
from biosignal_data.training.loader import Channel
from biosignal_model.scales import (
    ScaleSettings,
    ScaleTable,
    build,
    centre_of,
    key_of,
    log_level,
    robust_spread,
    transform,
    write,
)
from tests.test_training_loader import _index


def _channel(kind: str, unit: str | None) -> Channel:
    return Channel("a", 0, 100.0, 1.0, 0.0, None, "x", kind, unit)


def test_keys_by_type_and_unit_or_by_dataset_without_a_unit() -> None:
    assert key_of("physionet/ptb-xl", _channel("ecg", "mV")) == "ecg mV"
    assert key_of("physionet/i-care", _channel("eeg", "nu")) == "physionet/i-care eeg"
    assert key_of("d/x", _channel("ppg", None)) == "d/x ppg"
    assert key_of("d/x", _channel("ppg", " ")) == "d/x ppg"


def test_the_robust_spread_is_the_sd_of_a_normal_and_ignores_outliers() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, 200_000).astype(np.float32)
    assert robust_spread(x) == pytest.approx(2.0, rel=0.01)
    x[:2000] = 1e6  # 1 % of saturated samples barely move it
    x[2000:2100] = np.nan
    assert robust_spread(x) == pytest.approx(2.0, rel=0.03)
    assert robust_spread(np.full(10, 1.0, dtype=np.float32)) is None  # too few samples


def test_the_transform_is_linear_near_the_centre_and_logarithmic_far_from_it() -> None:
    x = np.array([1.0, 1.1, 1.0 + 100.0], dtype=np.float32)
    y = transform(x, scale=0.1, knee=3.0, centre=1.0)
    assert y[0] == 0.0 and y[1] == pytest.approx(1.0, rel=0.02)  # (x - c) / s
    assert y[2] == pytest.approx(3.0 * np.arcsinh(100.0 / 0.3), rel=1e-5)  # far: log
    assert y[2] < 30  # 1000 scales away comes in at about 20, not 1000


def test_the_centre_comes_from_the_selected_samples_only() -> None:
    x = np.array([1.0, 2.0, 3.0, 100.0, np.nan], dtype=np.float32)
    visible = np.array([True, True, True, False, True])
    assert centre_of(x, visible) == 2.0  # the hidden 100 does not move it
    assert centre_of(x, np.zeros(5, dtype=bool)) == 0.0
    assert log_level(0.0, 1.0) == 0.0


def test_the_table_from_an_index_has_a_scale_per_type_and_unit(
    store: LocalStore, tmp_path: Path
) -> None:
    # 100 Hz C3/C4 eeg = 0.5 x uniform ints in [-1000, 1000) + 1; ECG 0.002 x the same;
    # 10 Hz resp 1 x the same; all in uV. MAD of a uniform of half-width w is w / 2.
    _index(store, {f"r{i}": 60.0 for i in range(6)})
    settings = ScaleSettings(
        store=str(store.root), index="i1", split="train", signal_types=["eeg", "ecg", "resp"],
        windows_per_dataset=50, max_samples=2000, max_tokens=500, token_s=0.04,
        max_channels=8, subject_hours_per_epoch=1.0, block_s=10.0,
        cache_directory=tmp_path / "cache", cache_bytes=10**8, seed=0,
    )  # fmt: skip
    table = build(settings, StoreFactory(str(store.root), None), "abc")
    assert set(table.scales) == {"eeg uV", "ecg uV", "resp uV"}
    assert table.scales["eeg uV"].scale == pytest.approx(1.4826 * 250, rel=0.05)
    assert table.scales["ecg uV"].scale == pytest.approx(1.4826 * 1.0, rel=0.05)
    assert table.scales["resp uV"].scale == pytest.approx(1.4826 * 500, rel=0.1)
    assert all(s.windows > 0 and s.datasets == ["d/x"] for s in table.scales.values())
    write(table, tmp_path / "t.yaml")
    again = ScaleTable.load(tmp_path / "t.yaml")
    assert again == table
    assert again.scale("d/x", _channel("ecg", "uV")) == table.scales["ecg uV"].scale
    with pytest.raises(KeyError, match="no scale"):
        again.scale("d/x", _channel("bp", "mmHg"))

"""The scale of each kind of channel, from the corpus (docs/architecture.md section 2).

The model's value transform is x' = (a / s) asinh((x - c) / a), a = knee x s, with s one
scale per (signal type, unit) for channels with a physical unit, and per (dataset, signal type)
for channels without one (their numbers mean something only within a dataset). No scale is set
by hand: s is the median, over many windows of the training split, of each window's robust
spread (1.4826 x the median absolute deviation from its median, the standard deviation of a
normal distribution). Flat windows (spread 0) are counted and left out of the median.

The windows are read per dataset (at most `windows_per_dataset` each, cut as training cuts
them, a cap of hours per subject), so a small dataset gets its scales too. The table is written
as YAML with what made it, and a training run names the table it uses.
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from biosignal_data.storage.stores import StoreFactory
from biosignal_data.training.loader import (
    Channel,
    ExampleSource,
    SampleAndTokenLimit,
    Selection,
)

MAD_TO_SD = 1.4826  # the median absolute deviation of a normal distribution x this = its sd
# Units that are not physical units (the dataset says the numbers have none): i-care's EEG
# header unit "nu" ("the recordings do not have units", its documentation).
NOT_A_UNIT = {"", "nu"}
MIN_SAMPLES = 16  # valid samples a channel needs in a window to count


def key_of(dataset_id: str, channel: Channel) -> str:
    """The table key of a channel: `<type> <unit>`, or `<dataset> <type>` without a unit."""
    unit = (channel.unit or "").strip()
    if unit in NOT_A_UNIT:
        return f"{dataset_id} {channel.signal_type}"
    return f"{channel.signal_type} {unit}"


class ScaleSettings(BaseModel):
    """What the table is made from; none of it has a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store: str
    index: str
    split: str
    signal_types: list[str]  # the types a training run selects
    windows_per_dataset: int
    max_samples: int  # a window's samples over its channels (the training span's limit)
    max_tokens: int
    token_s: float
    max_channels: int
    subject_hours_per_epoch: float
    block_s: float
    cache_directory: Path
    cache_bytes: int
    seed: int


class Scale(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float  # s: the median robust spread of the windows
    windows: int  # windows with a spread above 0
    flat_windows: int  # windows with spread 0 (left out)
    datasets: list[str]
    p10: float  # of the window spreads: how much they vary around s
    p90: float


class ScaleTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: ScaleSettings
    built_at: str
    code_commit: str
    scales: dict[str, Scale]

    @classmethod
    def load(cls, path: Path) -> ScaleTable:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def scale(self, dataset_id: str, channel: Channel) -> float:
        """s of a channel; a channel of a key the table does not have is an error (the table
        must be made from the index the run reads)."""
        key = key_of(dataset_id, channel)
        if key not in self.scales:
            raise KeyError(f"no scale for {key!r} in the table of index {self.settings.index}")
        return self.scales[key].scale


def robust_spread(values: np.ndarray) -> float | None:
    """1.4826 x the median absolute deviation of the finite values, or None if too few."""
    v = values[np.isfinite(values)]
    if v.size < MIN_SAMPLES:
        return None
    return float(MAD_TO_SD * np.median(np.abs(v - np.median(v))))


def build(settings: ScaleSettings, factory: StoreFactory, code_commit: str) -> ScaleTable:
    store = factory()
    every = Selection.load(store, settings.index, split=settings.split,
                           signal_types=set(settings.signal_types))  # fmt: skip
    spreads: dict[str, list[float]] = defaultdict(list)
    flat: dict[str, int] = defaultdict(int)
    sources: dict[str, set[str]] = defaultdict(set)
    for dataset_id in sorted({r.dataset_id for r in every.recordings}):
        selection = Selection.load(store, settings.index, split=settings.split,
                                   signal_types=set(settings.signal_types),
                                   datasets={dataset_id})  # fmt: skip
        source = ExampleSource(
            selection, factory, cache_directory=settings.cache_directory,
            cache_bytes=settings.cache_bytes,
            limit=SampleAndTokenLimit(settings.max_samples, settings.max_tokens,
                                      settings.token_s),
            spans="per_group", subject_hours_per_epoch=settings.subject_hours_per_epoch,
            max_channels=settings.max_channels, block_s=settings.block_s, shuffle_buffer=4096,
            reads_in_flight=8, seed=settings.seed, passes=None, flat_runs=None,
        )  # fmt: skip
        examples = source.examples(0, 0, 1)
        for n, example in enumerate(examples):
            if n >= settings.windows_per_dataset:
                examples.close()
                break
            for signal, channel in zip(example.signals, example.origin.channels, strict=True):
                spread = robust_spread(signal.values)
                if spread is None:
                    continue
                key = key_of(dataset_id, channel)
                sources[key].add(dataset_id)
                if spread > 0:
                    spreads[key].append(spread)
                else:
                    flat[key] += 1
    scales = {}
    for key in sorted(set(spreads) | set(flat)):
        s = np.array(spreads.get(key, []))
        if s.size == 0:  # only flat windows: no scale (the channels hold no signal)
            continue
        scales[key] = Scale(
            scale=float(np.median(s)), windows=int(s.size), flat_windows=flat.get(key, 0),
            datasets=sorted(sources[key]), p10=float(np.percentile(s, 10)),
            p90=float(np.percentile(s, 90)),
        )  # fmt: skip
    return ScaleTable(settings=settings, built_at=dt.datetime.now(dt.UTC).isoformat(),
                      code_commit=code_commit, scales=scales)  # fmt: skip


def write(table: ScaleTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = table.model_dump(mode="json")
    path.write_text(yaml.safe_dump(data, sort_keys=False, width=100))


def transform(
    values: np.ndarray, scale: float, knee: float, centre: float
) -> np.ndarray:  # fmt: skip
    """x' = (a / s) asinh((x - c) / a) with a = knee x s: about (x - c) / s near the centre,
    logarithmic far from it, so a large value (a pacing spike, a saturated lead) does not
    dominate while small ones keep their linear scale."""
    a = knee * scale
    return ((a / scale) * np.arcsinh((values - centre) / a)).astype(np.float32)


def centre_of(values: np.ndarray, use: np.ndarray) -> float:
    """c: the median of the samples `use` selects (finite only); 0 if there are none."""
    v = values[use & np.isfinite(values)]
    return float(np.median(v)) if v.size else 0.0


def log_level(centre: float, scale: float) -> float:
    """The level c as a model input: asinh(c / s), a signed log of the level in scales."""
    return math.asinh(centre / scale)

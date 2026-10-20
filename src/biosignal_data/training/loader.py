"""Windows for training, read from the standard layer through an index.

from biosignal_data.training import Selection, WindowSource
selection = Selection.load(store, "2026-10-20", split="train", signal_types={"ecg", "eeg"})
source = WindowSource(
    selection, store_factory, cache_directory=Path("/data/cache"), cache_bytes=200 * 2**30,
    window_s=30.0, min_window_s=10.0, shuffle_buffer=4096, reads_in_flight=8, seed=0,
)
for window in source.windows(epoch=0, part=0, parts=1):
    window.signals  # float32 (channels, samples), physical units, NaN where invalid

Each window is one array (one sampling rate) of one recording, at its native rate: nothing
is resampled, and nothing is normalized. The experiment decides both.

How it reads: a unit of work is one array of one recording. Its shards are read whole
(64 MiB at most) through a local disk cache, and cut into windows of `window_s` from a
random offset; the windows go through a shuffle buffer. Several units are read at the same
time (`reads_in_flight`), because one read from the S3 storage takes 80-240 ms. Memory is
bounded: at most two buffers of `shuffle_buffer` windows, and one shard per read.

`parts`: the units are dealt round-robin over the parts (data loader workers times
distributed ranks), after a shuffle that depends on `seed` and `epoch` only.
"""

from __future__ import annotations

import io
import itertools
import queue
import random
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

from biosignal_data.index.build import ROOT as INDEX_ROOT
from biosignal_data.storage.layout import SUCCESS
from biosignal_data.storage.stores import Store
from biosignal_data.training.cache import WholeObjectCache


@dataclass(frozen=True)
class Channel:
    index: int  # in the array
    name: str
    signal_type: str
    unit: str | None
    scale: float | None  # None: not calibrated, the window holds NaN
    offset: float | None
    invalid_value: int | None


@dataclass(frozen=True)
class Unit:
    """One array of one recording, with the channels selected from it."""

    dataset_id: str
    recording_id: str
    subject_id: str
    signals: str  # the key of the recording's Zarr group, from the store root
    array: str
    sampling_rate: float
    channels: tuple[Channel, ...]


@dataclass(frozen=True)
class Window:
    signals: np.ndarray  # float32 (channels, samples), physical units, NaN where invalid
    sampling_rate: float
    start_s: float  # from the start of the recording
    channel_names: tuple[str, ...]
    signal_types: tuple[str, ...]
    units: tuple[str | None, ...]
    dataset_id: str
    recording_id: str
    subject_id: str


class ReadError(RuntimeError):
    """A unit could not be read; the message names it, the cause is chained."""


@dataclass(frozen=True)
class _Done:
    unit: Unit
    error: Exception | None


@dataclass(frozen=True)
class Selection:
    """The units of one split of an index, with only the channels of `signal_types`."""

    index: str
    split: str
    units: tuple[Unit, ...]

    @classmethod
    def load(
        cls,
        store: Store,
        index: str,
        *,
        split: str,
        signal_types: set[str],
        datasets: set[str] | None = None,
    ) -> Selection:
        """`datasets`: only these dataset ids; None for all in the index."""
        prefix = f"{INDEX_ROOT}/{index}"
        if not store.exists(f"{prefix}/{SUCCESS}"):
            raise FileNotFoundError(f"{store.uri}/{prefix} is not a complete index")
        recordings = {
            (r["dataset_id"], r["recording_id"]): r
            for r in _table(store, f"{prefix}/recordings.parquet")
            if r["split"] == split and (datasets is None or r["dataset_id"] in datasets)
        }
        channels: dict[tuple[str, str, str], list[Channel]] = {}
        rates: dict[tuple[str, str, str], float] = {}
        for c in _table(store, f"{prefix}/channels.parquet"):
            key = (c["dataset_id"], c["recording_id"])
            if key not in recordings or c["signal_type"] not in signal_types:
                continue
            unit_key = (*key, c["array"])
            rates[unit_key] = c["sampling_rate"]
            channels.setdefault(unit_key, []).append(
                Channel(
                    c["index"], c["name"], c["signal_type"], c["unit"], c["scale"],
                    c["offset"], c["invalid_value"],
                )
            )  # fmt: skip
        units = tuple(
            Unit(
                dataset_id, recording_id, recordings[dataset_id, recording_id]["subject_id"],
                recordings[dataset_id, recording_id]["signals"], array,
                rates[dataset_id, recording_id, array],
                tuple(sorted(found, key=lambda c: c.index)),
            )
            for (dataset_id, recording_id, array), found in sorted(channels.items())
        )  # fmt: skip
        return cls(index, split, units)


class WindowSource:
    """Windows of the units of a selection, shuffled. `store_factory` makes the store in
    the process that reads (a data loader worker), because a store does not cross
    processes. Every argument is the experiment's choice; none has a default."""

    def __init__(
        self,
        selection: Selection,
        store_factory: Callable[[], Store],
        *,
        cache_directory: Path,
        cache_bytes: int,
        window_s: float,
        min_window_s: float,
        shuffle_buffer: int,
        reads_in_flight: int,
        seed: int,
    ) -> None:
        if not 0 < min_window_s <= window_s:
            raise ValueError("need 0 < min_window_s <= window_s")
        if shuffle_buffer < 1 or reads_in_flight < 1:
            raise ValueError("shuffle_buffer and reads_in_flight are at least 1")
        self.selection = selection
        self._store_factory = store_factory
        self._store: Store | None = None
        self.cache_directory = cache_directory
        self.cache_bytes = cache_bytes
        self.window_s = window_s
        self.min_window_s = min_window_s
        self.shuffle_buffer = shuffle_buffer
        self.reads_in_flight = reads_in_flight
        self.seed = seed

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None  # each process makes its own
        return state

    def units_of(self, epoch: int, part: int, parts: int) -> list[Unit]:
        """The units that `part` of `parts` reads in `epoch`, in reading order."""
        if not 0 <= part < parts:
            raise ValueError(f"part {part} is not in 0..{parts - 1}")
        order = list(self.selection.units)
        random.Random(f"{self.seed}/{epoch}").shuffle(order)  # noqa: S311 - not secret
        return order[part::parts]

    def windows(self, epoch: int, part: int, parts: int) -> Iterator[Window]:
        """The windows of `units_of(epoch, part, parts)`, shuffled. Which windows come is
        fixed by `seed` and `epoch`; their order also depends on which read ends first."""
        units = self.units_of(epoch, part, parts)
        rng = random.Random(f"{self.seed}/{epoch}/{part}")  # noqa: S311 - not secret
        phases = [rng.random() for _ in units]
        out: queue.Queue[Window | _Done] = queue.Queue(maxsize=self.shuffle_buffer)
        stop = threading.Event()

        def put(item: Window | _Done) -> bool:
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.1)
                except queue.Full:
                    continue
                return True
            return False

        def read(unit: Unit, phase: float) -> None:
            try:
                for window in self._cut(unit, phase):
                    if not put(window):
                        return
            except Exception as exc:  # noqa: BLE001 - raised again in the reading thread
                put(_Done(unit, exc))
                return
            put(_Done(unit, None))

        pool = ThreadPoolExecutor(self.reads_in_flight)
        buffer: list[Window] = []
        try:
            for unit, phase in zip(units, phases, strict=True):
                pool.submit(read, unit, phase)
            done = 0
            while done < len(units):
                item = out.get()
                if isinstance(item, _Done):
                    if item.error is not None:
                        raise ReadError(f"{item.unit.signals}/{item.unit.array}") from item.error
                    done += 1
                    continue
                buffer.append(item)
                if len(buffer) > self.shuffle_buffer:
                    yield _take(buffer, rng)
            while buffer:
                yield _take(buffer, rng)
        finally:  # also when the consumer stops early
            stop.set()
            pool.shutdown(wait=True, cancel_futures=True)

    def _cut(self, unit: Unit, phase: float) -> Iterator[Window]:
        """The windows of one unit. Cuts at `phase` times one window and every window from
        there; a piece shorter than `min_window_s` (at the ends) is left out, and a unit
        shorter than one window is one piece. Reads a whole shard at a time."""
        array = self._array(unit)
        rate = unit.sampling_rate
        length = round(self.window_s * rate)
        shortest = max(1, round(self.min_window_s * rate))
        total = int(array.shape[1])
        shard = int(array.shards[1]) if array.shards is not None else total
        rows = [c.index for c in unit.channels]
        first_cut = round(phase * length) % length
        if total < length:  # shorter than one window: all of it, if long enough
            cuts = [0, total]
        else:
            cuts = [0, *range(first_cut or length, total, length), total]
        block_start, block = 0, None
        for begin, end in itertools.pairwise(cuts):
            if end - begin < shortest:
                continue
            if block is None or end > block_start + block.shape[1] or begin < block_start:
                block_end = min(total, -(-end // shard) * shard)  # to the end of a shard
                block_start = begin
                block = np.asarray(array[:, begin:block_end])[rows]
            stored = block[:, begin - block_start : end - block_start]
            yield self._window(unit, stored, begin / rate)

    def _window(self, unit: Unit, stored: np.ndarray, start_s: float) -> Window:
        scale = np.array([np.nan if c.scale is None else c.scale for c in unit.channels])
        offset = np.array([np.nan if c.offset is None else c.offset for c in unit.channels])
        values = (stored * scale[:, None] + offset[:, None]).astype(np.float32)
        for i, c in enumerate(unit.channels):
            if c.invalid_value is not None:
                values[i, stored[i] == c.invalid_value] = np.nan
        return Window(
            values, unit.sampling_rate, start_s,
            tuple(c.name for c in unit.channels), tuple(c.signal_type for c in unit.channels),
            tuple(c.unit for c in unit.channels), unit.dataset_id, unit.recording_id,
            unit.subject_id,
        )  # fmt: skip

    def _array(self, unit: Unit) -> zarr.Array[Any]:
        if self._store is None:
            self._store = self._store_factory()
        cached = WholeObjectCache(
            self._store.zarr_store(unit.signals, read_only=True),
            self.cache_directory, unit.signals, self.cache_bytes,
        )  # fmt: skip
        found = zarr.open_group(store=cached, mode="r")[unit.array]
        if not isinstance(found, zarr.Array):
            raise TypeError(f"{unit.signals}/{unit.array} is not an array")
        return found


def _take(buffer: list[Window], rng: random.Random) -> Window:
    """Remove a random window from the buffer (swap with the last: no shifting)."""
    i = rng.randrange(len(buffer))
    buffer[i], buffer[-1] = buffer[-1], buffer[i]
    return buffer.pop()


def _table(store: Store, key: str) -> list[dict[str, Any]]:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    rows: list[dict[str, Any]] = pq.read_table(io.BytesIO(data)).to_pylist()
    return rows

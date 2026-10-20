"""Examples for training, read from the standard layer through an index.

from biosignal_data.training import ExampleSource, Selection
selection = Selection.load(store, "2026-10-20", split="train", signal_types=None)
source = ExampleSource(
    selection, store_factory, cache_directory=Path("/data/cache"), cache_bytes=200 * 2**30,
    max_tokens=8192, samples_per_token=64, max_channels=32, block_s=1800.0, shuffle_buffer=4096,
    reads_in_flight=8, seed=0,
)
for example in source.examples(epoch=0, part=0, parts=1):
    example.signals  # the model input: values, rate and start time of each channel
    example.origin   # where it comes from: for analysis only, never for the model

The model gets the samples and their times only. Each signal of an example is one channel:
its values (float32, physical values; stored integers if the channel has no calibration;
NaN where a sample is invalid), its sampling rate, and the time of its first sample in
seconds from the start of the example. All signals of an example are from one recording
and one time span, in random order. Names, types, units, datasets and the position in the
recording are in `origin`, which a data loader for the model must drop.

Float32 keeps each value to within 1/2**24 of itself: far finer than one step of the
recorder (the `scale`) for every value a 16-bit or 24-bit recorder makes.

The unit of work is a block of `block_s` seconds of one recording; the blocks of all
recordings are shuffled together each epoch, so that every phase of an epoch has the
same mix (reading whole recordings put the long ones at the end of each epoch, and the
model forgot the short ones meanwhile). An example never spans two blocks, so `block_s`
also bounds the longest example.

How a block becomes examples, for one epoch: a time cursor moves from the start to the
end of the block. At each step the selected channels are put in random order and
cut into groups of random size (1 to `max_channels`); each group is one example over the
same time span. The span is as long as the group with the most samples per second allows:
at most `max_tokens` tokens of `samples_per_token` samples in each example. The first span
is a random part of a full one, so the cuts differ from epoch to epoch. Every sample of
every selected channel is in exactly one example per epoch; nothing is left out.

How it reads: a recording is read through a local disk cache of whole shards (64 MiB at
most), several recordings at the same time (`reads_in_flight`), because one read from the
S3 storage takes 80-240 ms. Examples go through a shuffle buffer. Memory is bounded: two
buffers of `shuffle_buffer` examples, and one span per read.

`parts`: the blocks are dealt round-robin over the parts (data loader workers times
distributed ranks), after a shuffle that depends on `seed` and `epoch` only.
"""

from __future__ import annotations

import io
import logging
import math
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

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Channel:
    """A channel as the index describes it. Only `array`, `index`, `sampling_rate` and the
    calibration are used to read; the rest is for selection and analysis."""

    array: str
    index: int  # in the array
    sampling_rate: float
    scale: float | None  # None: no calibration, the values are the stored integers
    offset: float | None
    invalid_value: int | None
    name: str
    signal_type: str
    unit: str | None


@dataclass(frozen=True)
class Recording:
    dataset_id: str
    recording_id: str
    subject_id: str
    signals: str  # the key of the recording's Zarr group, from the store root
    channels: tuple[Channel, ...]  # the selected ones
    duration_s: float  # from the index: how many blocks the recording has


@dataclass(frozen=True)
class Block:
    """The unit of work: [index * block_s, (index + 1) * block_s) of one recording; the
    last block of a recording runs to its end."""

    recording: Recording
    index: int
    last: bool


@dataclass(frozen=True)
class Signal:
    """One channel of an example: all the model gets of it."""

    values: np.ndarray  # float32 (samples,)
    sampling_rate: float
    start_s: float  # time of the first sample, from the start of the example: [0, 1/rate)


@dataclass(frozen=True)
class Origin:
    """Where an example comes from, signal by signal. Never an input of the model."""

    dataset_id: str
    recording_id: str
    subject_id: str
    start_s: float  # the start of the example, from the start of the recording
    channels: tuple[Channel, ...]  # in the order of `Example.signals`


@dataclass(frozen=True)
class Example:
    signals: tuple[Signal, ...]
    origin: Origin


class ReadError(RuntimeError):
    """A recording could not be read; the message names it, the cause is chained."""


@dataclass(frozen=True)
class _Done:
    recording: Recording
    error: Exception | None


@dataclass(frozen=True)
class Selection:
    """The recordings of one split of an index, with the channels selected from them."""

    index: str
    split: str
    recordings: tuple[Recording, ...]

    @classmethod
    def load(
        cls,
        store: Store,
        index: str,
        *,
        split: str,
        signal_types: set[str] | None,
        datasets: set[str] | None = None,
    ) -> Selection:
        """`signal_types`: keep only channels of these types (from the index), or None for
        all channels. `datasets`: only these dataset ids, or None for all. Both select
        data for an experiment; neither is ever shown to the model."""
        prefix = f"{INDEX_ROOT}/{index}"
        if not store.exists(f"{prefix}/{SUCCESS}"):
            raise FileNotFoundError(f"{store.uri}/{prefix} is not a complete index")
        recordings = {
            (r["dataset_id"], r["recording_id"]): r
            for r in _table(store, f"{prefix}/recordings.parquet")
            if r["split"] == split and (datasets is None or r["dataset_id"] in datasets)
        }
        channels: dict[tuple[str, str], list[Channel]] = {}
        for c in _table(store, f"{prefix}/channels.parquet"):
            key = (c["dataset_id"], c["recording_id"])
            if key not in recordings:
                continue
            if signal_types is not None and c["signal_type"] not in signal_types:
                continue
            channels.setdefault(key, []).append(
                Channel(
                    c["array"], c["index"], c["sampling_rate"], c["scale"], c["offset"],
                    c["invalid_value"], c["name"], c["signal_type"], c["unit"],
                )
            )  # fmt: skip
        found = tuple(
            Recording(
                dataset_id, recording_id, recordings[dataset_id, recording_id]["subject_id"],
                recordings[dataset_id, recording_id]["signals"],
                tuple(sorted(chosen, key=lambda c: (c.array, c.index))),
                _duration(recordings[dataset_id, recording_id]),
            )
            for (dataset_id, recording_id), chosen in sorted(channels.items())
        )  # fmt: skip
        return cls(index, split, found)


class ExampleSource:
    """Examples of the recordings of a selection, shuffled. `store_factory` makes the store
    in the process that reads (a data loader worker), because a store does not cross
    processes. Every argument is the experiment's choice; none has a default."""

    def __init__(
        self,
        selection: Selection,
        store_factory: Callable[[], Store],
        *,
        cache_directory: Path,
        cache_bytes: int,
        max_tokens: int,
        samples_per_token: int,
        max_channels: int,
        block_s: float,
        shuffle_buffer: int,
        reads_in_flight: int,
        seed: int,
    ) -> None:
        if samples_per_token < 1 or max_channels < 1:
            raise ValueError("samples_per_token and max_channels are at least 1")
        if max_tokens <= 2 * max_channels:
            raise ValueError("max_tokens must be more than twice max_channels")
        if shuffle_buffer < 1 or reads_in_flight < 1:
            raise ValueError("shuffle_buffer and reads_in_flight are at least 1")
        self.selection = selection
        self._store_factory = store_factory
        self._store: Store | None = None
        self.cache_directory = cache_directory
        self.cache_bytes = cache_bytes
        self.max_tokens = max_tokens
        self.samples_per_token = samples_per_token
        self.max_channels = max_channels
        if block_s <= 0:
            raise ValueError("block_s is more than 0")
        self.block_s = block_s
        self.shuffle_buffer = shuffle_buffer
        self.reads_in_flight = reads_in_flight
        self.seed = seed

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None  # each process makes its own
        return state

    def blocks_of(self, epoch: int, part: int, parts: int) -> list[Block]:
        """The blocks that `part` of `parts` reads in `epoch`, in reading order: the blocks
        of all recordings, shuffled together, so that every phase of an epoch has the same
        mix of datasets (a long recording is spread over the whole epoch, not read at its
        end)."""
        if not 0 <= part < parts:
            raise ValueError(f"part {part} is not in 0..{parts - 1}")
        order = []
        for recording in self.selection.recordings:
            n = max(1, math.ceil(recording.duration_s / self.block_s))
            order += [Block(recording, i, i == n - 1) for i in range(n)]
        random.Random(f"{self.seed}/{epoch}").shuffle(order)  # noqa: S311 - not secret
        return order[part::parts]

    def examples(self, epoch: int, part: int, parts: int) -> Iterator[Example]:
        """The examples of `blocks_of(epoch, part, parts)`, shuffled. Which examples
        come is fixed by `seed` and `epoch`; their order also depends on which read ends
        first."""
        blocks = self.blocks_of(epoch, part, parts)
        rng = random.Random(f"{self.seed}/{epoch}/{part}")  # noqa: S311 - not secret
        out: queue.Queue[Example | _Done] = queue.Queue(maxsize=self.shuffle_buffer)
        stop = threading.Event()

        def put(item: Example | _Done) -> bool:
            while not stop.is_set():
                try:
                    out.put(item, timeout=0.1)
                except queue.Full:
                    continue
                return True
            return False

        def read(block: Block) -> None:
            try:
                for example in self.cut(block, epoch):
                    if not put(example):
                        return
            except Exception as exc:  # noqa: BLE001 - raised again in the reading thread
                put(_Done(block.recording, exc))
                return
            put(_Done(block.recording, None))

        pool = ThreadPoolExecutor(self.reads_in_flight)
        buffer: list[Example] = []
        try:
            for block in blocks:
                pool.submit(read, block)
            done = 0
            while done < len(blocks):
                item = out.get()
                if isinstance(item, _Done):
                    if item.error is not None:
                        raise ReadError(item.recording.signals) from item.error
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

    def cut(self, block: Block, epoch: int) -> Iterator[Example]:
        """All examples of one block in `epoch`, in time order."""
        recording = block.recording
        rng = random.Random(  # noqa: S311 - not secret
            f"{self.seed}/{epoch}/{recording.dataset_id}/{recording.recording_id}/{block.index}"
        )
        root = self._group(recording)
        arrays = {name: _array(root, name) for name in {c.array for c in recording.channels}}
        lengths = {c: int(arrays[c.array].shape[1]) for c in recording.channels}
        recording_end = max(lengths[c] / c.sampling_rate for c in recording.channels)
        t = block.index * self.block_s
        end_s = recording_end if block.last else min(recording_end, t + self.block_s)
        first = True
        while t < end_s:
            channels = [c for c in recording.channels if lengths[c] > _first_sample(t, c)]
            if not channels:  # the last samples start before end_s but after t
                break
            rng.shuffle(channels)
            groups = _random_groups(channels, self.max_channels, rng)
            duration = min(self._longest(g) for g in groups)
            if first:  # a random part of a span: the cuts differ from epoch to epoch
                duration *= 1.0 - rng.random()
                first = False
            stop_s = min(t + duration, end_s)
            read = self._read(arrays, channels, t, stop_s)
            for group in groups:
                chosen = [c for c in group if read[c].values.size]  # a span can be too short
                if not chosen:
                    continue
                signals = tuple(read[c] for c in chosen)
                origin = Origin(
                    recording.dataset_id, recording.recording_id, recording.subject_id, t,
                    tuple(chosen),
                )  # fmt: skip
                yield Example(signals, origin)
            t = stop_s

    def _longest(self, channels: list[Channel]) -> float:
        """The longest span (s) in which `channels` fit `max_tokens`. A span of d seconds
        holds at most rate * d + 1 samples of a channel, which take at most
        (rate * d + 1) / samples_per_token + 1 tokens: two more tokens a channel at most."""
        samples_per_s = sum(c.sampling_rate for c in channels)
        tokens = self.max_tokens - 2 * len(channels)
        return tokens * self.samples_per_token / samples_per_s

    def _read(
        self,
        arrays: dict[str, zarr.Array[Any]],
        channels: list[Channel],
        start_s: float,
        stop_s: float,
    ) -> dict[Channel, Signal]:
        """The samples of [start_s, stop_s) of each channel, one read per array."""
        found: dict[Channel, Signal] = {}
        for name in {c.array for c in channels}:
            mine = [c for c in channels if c.array == name]
            rate = mine[0].sampling_rate
            first = _first_sample(start_s, mine[0])
            last = min(int(arrays[name].shape[1]), _first_sample(stop_s, mine[0]))
            stored = _read_with_retries(arrays[name], first, last)
            for c in mine:
                found[c] = Signal(_values(stored[c.index], c), rate, first / rate - start_s)
        return found

    def _group(self, recording: Recording) -> zarr.Group:
        if self._store is None:
            self._store = self._store_factory()
        cached = WholeObjectCache(
            self._store.zarr_store(recording.signals, read_only=True),
            self.cache_directory, recording.signals, self.cache_bytes,
        )  # fmt: skip
        return zarr.open_group(store=cached, mode="r")


def _array(root: zarr.Group, name: str) -> zarr.Array[Any]:
    found = root[name]
    if not isinstance(found, zarr.Array):
        raise TypeError(f"{name} is not an array")
    return found


READ_ATTEMPTS = 3


def _read_with_retries(array: zarr.Array[Any], first: int, last: int) -> np.ndarray:
    """Samples [first, last) of every row; a read that times out or fails in transport is
    tried again, up to READ_ATTEMPTS times (see storage.stores: ZARR_CALL_TIMEOUT_S)."""
    for attempt in range(1, READ_ATTEMPTS + 1):
        try:
            return np.asarray(array[:, first:last])
        except (TimeoutError, OSError) as exc:
            if attempt == READ_ATTEMPTS:
                raise
            log.warning(
                "read of %s [%d:%d] failed (%s); trying again", array.path, first, last, exc
            )
    raise AssertionError("unreachable")


def _duration(row: dict[str, Any]) -> float:
    if row["duration_s"] is None:
        raise ValueError(f"{row['dataset_id']}: {row['recording_id']} has no duration in the index")
    return float(row["duration_s"])


def _first_sample(t: float, channel: Channel) -> int:
    """The index of the first sample at or after `t` seconds."""
    return math.ceil(round(t * channel.sampling_rate, 9))


def _values(stored: np.ndarray, channel: Channel) -> np.ndarray:
    """Physical values as float32; NaN where a sample is invalid, and where the physical
    value is not a finite float32. The latter comes from a broken calibration in a source
    header (e.g. autonomic-aging-cardiovascular 0065 NIBP: a gain of 2.78e-152, so a scale
    of 3.6e151), never from a real measurement."""
    if channel.scale is None:  # no calibration: the stored integers
        values = stored.astype(np.float32)
    else:
        physical = stored * np.float64(channel.scale) + np.float64(channel.offset or 0.0)
        finite = np.isfinite(physical) & (np.abs(physical) <= np.finfo(np.float32).max)
        values = np.where(finite, physical, np.nan).astype(np.float32)
    if channel.invalid_value is not None:
        values[stored == channel.invalid_value] = np.nan
    return values


def _random_groups(
    channels: list[Channel], max_channels: int, rng: random.Random
) -> list[list[Channel]]:
    """`channels` (already in random order) cut into groups of random size."""
    groups, i = [], 0
    while i < len(channels):
        size = rng.randint(1, min(max_channels, len(channels) - i))
        groups.append(channels[i : i + size])
        i += size
    return groups


def _take(buffer: list[Example], rng: random.Random) -> Example:
    """Remove a random example from the buffer (swap with the last: no shifting)."""
    i = rng.randrange(len(buffer))
    buffer[i], buffer[-1] = buffer[-1], buffer[i]
    return buffer.pop()


def _table(store: Store, key: str) -> list[dict[str, Any]]:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    rows: list[dict[str, Any]] = pq.read_table(io.BytesIO(data)).to_pylist()
    return rows

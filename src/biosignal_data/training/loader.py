"""Examples for training, read from the standard layer through an index.

from biosignal_data.training import ExampleSource, Selection
selection = Selection.load(store, "2026-10-20", split="train", signal_types=None)
source = ExampleSource(
    selection, store_factory, cache_directory=Path("/data/cache"), cache_bytes=200 * 2**30,
    limit=SampleAndTokenLimit(max_samples=400_000, max_tokens=8192, token_s=0.04),
    spans="per_group", subject_hours_per_epoch=4.0, max_channels=32, block_s=1800.0,
    shuffle_buffer=4096, reads_in_flight=8, seed=0, passes=None,
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
cut into groups of random size (1 to `max_channels`). The longest span a group may have
comes from `limit`:

- `SampleTokenLimit` (v0-v3): at most `max_tokens` tokens of `samples_per_token` samples;
- `SampleAndTokenLimit` (v4, docs/architecture.md): at most `max_samples` local samples (what
  the local stage reads) and at most `max_tokens` global tokens (cells of `token_s` seconds
  on one grid from the example's start that hold at least one sample of the channel).

`spans`: `shared` (v0-v3): every group is one example over the same span, as long as the
group with the most samples per second allows (the minimum over the groups). `per_group`:
each group cuts its own spans, each as long as its own limit allows, up to the step's end
(the longest of the groups' spans), where all groups are made anew; a slow group is then not
cut to the span of a fast one. The first step is a random part of a full one, so the cuts
differ from epoch to epoch. Every sample of every selected channel of every block read is in
exactly one example per reading of the block; nothing is left out.

`subject_hours_per_epoch` (None: no cap): a subject gives at most this many hours of blocks
to one pass over its dataset, as random blocks drawn anew in each epoch and pass (i-care has
~607 patients and more than half of all samples; docs/architecture.md section 1).

How it reads: the block index of each standard version is loaded once (blocks.parquet);
a span is one ranged request per block it touches (blocks hold at most 8 MiB), through a
local disk cache of blocks, several recordings at the same time (`reads_in_flight`),
because one read from the S3 storage takes 80-240 ms. Examples go through a shuffle
buffer. Memory is bounded: two buffers of `shuffle_buffer` examples, and one read: one
span (`shared`), or one group over one step (`per_group`; at most one block).

`parts`: the blocks are dealt round-robin over the parts (data loader workers times
distributed ranks), after a shuffle that depends on `seed` and `epoch` only.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import queue
import random
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
from botocore.exceptions import BotoCoreError

from biosignal_data.index.build import ROOT as INDEX_ROOT
from biosignal_data.standard import blocks
from biosignal_data.storage.layout import SUCCESS
from biosignal_data.storage.stores import Store
from biosignal_data.training.cache import BlockCache

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
    standard: str  # the standard version folder, from the store root
    channels: tuple[Channel, ...]  # the selected ones
    duration_s: float  # from the index: how many blocks the recording has
    arrays: tuple[tuple[str, blocks.ArrayBlocks], ...]  # the stored blocks of each array

    def array_blocks(self, array: str) -> blocks.ArrayBlocks:
        for name, found in self.arrays:
            if name == array:
                return found
        raise KeyError(f"{self.dataset_id}/{self.recording_id}: no array {array!r}")

    @property
    def name(self) -> str:
        return f"{self.dataset_id}/{self.recording_id}"


@dataclass(frozen=True)
class Block:
    """The unit of work: [index * block_s, (index + 1) * block_s) of one recording; the
    last block of a recording runs to its end."""

    recording: Recording
    index: int
    last: bool
    copy: int = 0  # the block's n-th reading in an epoch (a mixture repeats blocks)


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


@dataclass(frozen=True)
class SampleTokenLimit:
    """v0-v3: a span holds at most `max_tokens` tokens of `samples_per_token` samples."""

    max_tokens: int
    samples_per_token: int

    def tokens_per_second(self, channel: Channel) -> float:
        """The tokens one second of `channel` gives (for the mixture's sizes)."""
        return channel.sampling_rate / self.samples_per_token


@dataclass(frozen=True)
class SampleAndTokenLimit:
    """v4: a span holds at most `max_samples` samples (over its channels) and at most
    `max_tokens` global tokens: cells of `token_s` seconds on one grid from the example's
    start that hold at least one sample of their channel (biosignal_model.channel_batch)."""

    max_samples: int
    max_tokens: int
    token_s: float

    def tokens_per_second(self, channel: Channel) -> float:
        """The global tokens one second of `channel` gives at most: a cell holds at least
        one sample, so a channel slower than one sample a cell has one token a sample."""
        return min(1.0 / self.token_s, channel.sampling_rate)


SpanLimit = SampleTokenLimit | SampleAndTokenLimit
Spans = Literal["shared", "per_group"]


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
        indexes = {
            folder: blocks.index(
                pq.read_table(io.BytesIO(_bytes(store, f"{folder}/{blocks.TABLE}.parquet")))
            )
            for folder in sorted({recordings[key]["standard"] for key in channels})
        }

        def make(key: tuple[str, str], chosen: list[Channel]) -> Recording:
            row = recordings[key]
            arrays = sorted({c.array for c in chosen})
            return Recording(
                key[0], key[1], row["subject_id"], row["standard"],
                tuple(sorted(chosen, key=lambda c: (c.array, c.index))), _duration(row),
                tuple((a, indexes[row["standard"]][key[1], a]) for a in arrays),
            )  # fmt: skip

        return cls(index, split, tuple(make(k, chosen) for k, chosen in sorted(channels.items())))


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
        limit: SpanLimit,
        spans: Spans,
        subject_hours_per_epoch: float | None,
        max_channels: int,
        block_s: float,
        shuffle_buffer: int,
        reads_in_flight: int,
        seed: int,
        passes: Mapping[str, float] | None,
    ) -> None:
        """`limit`, `spans`, `subject_hours_per_epoch`: see the module docstring. `passes`:
        readings of each dataset's blocks in one epoch (training.mixture), or None: every
        block once."""
        if max_channels < 1:
            raise ValueError("max_channels is at least 1")
        if isinstance(limit, SampleTokenLimit):
            if limit.samples_per_token < 1:
                raise ValueError("samples_per_token is at least 1")
        elif limit.token_s <= 0 or limit.max_samples <= 2 * max_channels:
            raise ValueError("token_s is more than 0; max_samples more than twice max_channels")
        if limit.max_tokens <= 2 * max_channels:
            raise ValueError("max_tokens must be more than twice max_channels")
        if spans not in ("shared", "per_group"):
            raise ValueError(f"spans is shared or per_group, not {spans!r}")
        if shuffle_buffer < 1 or reads_in_flight < 1:
            raise ValueError("shuffle_buffer and reads_in_flight are at least 1")
        self.selection = selection
        self._store_factory = store_factory
        self._store: Store | None = None
        self._cache: BlockCache | None = None
        self.cache_directory = cache_directory
        self.cache_bytes = cache_bytes
        self.limit = limit
        self.spans = spans
        self.max_channels = max_channels
        if block_s <= 0:
            raise ValueError("block_s is more than 0")
        if subject_hours_per_epoch is not None and subject_hours_per_epoch * 3600.0 < block_s:
            raise ValueError("subject_hours_per_epoch must hold at least one block (block_s)")
        self.subject_hours_per_epoch = subject_hours_per_epoch
        self.block_s = block_s
        self.shuffle_buffer = shuffle_buffer
        self.reads_in_flight = reads_in_flight
        self.seed = seed
        if passes is not None:
            missing = {r.dataset_id for r in selection.recordings} - set(passes)
            if missing or any(v < 0 for v in passes.values()):
                raise ValueError(f"passes: none or negative for {sorted(missing) or passes}")
        self.passes = dict(passes) if passes is not None else None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None  # each process makes its own
        state["_cache"] = None
        return state

    def blocks_of(self, epoch: int, part: int, parts: int) -> list[Block]:
        """The blocks that `part` of `parts` reads in `epoch`, in reading order: the blocks
        of all recordings, shuffled together, so that every phase of an epoch has the same
        mix of datasets (a long recording is spread over the whole epoch, not read at its
        end). The order of all parts is made as arrays (12 bytes a block), and only this
        part's blocks become objects: every data loader worker makes the order, and with
        a mixture there are millions of blocks (v2-unimax: 22 workers of ~5 GB each ran
        the training out of memory with one object per block in each)."""
        if not 0 <= part < parts:
            raise ValueError(f"part {part} is not in 0..{parts - 1}")
        recordings = self.selection.recordings
        counts = np.array(
            [max(1, math.ceil(r.duration_s / self.block_s)) for r in recordings], dtype=np.int64
        )
        recording = np.repeat(np.arange(len(recordings), dtype=np.int64), counts)
        index = np.arange(recording.size, dtype=np.int64) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        recording, index, copy = self._readings(recording, index, epoch)
        order = _rng(f"{self.seed}/{epoch}").permutation(recording.size)[part::parts]
        return [
            Block(recordings[r], int(i), int(i) == counts[r] - 1, int(c))
            for r, i, c in zip(recording[order], index[order], copy[order], strict=True)
        ]

    def _readings(
        self, recording: np.ndarray, index: np.ndarray, epoch: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The blocks read in `epoch`, as (recording, index, copy) arrays: each dataset's
        blocks `passes` times (whole passes, then a random part of its blocks for the
        fraction, different in each epoch), or once without a mixture; in each pass a
        subject gives at most `subject_hours_per_epoch` hours of random blocks (drawn anew
        in each pass). A repeated block is cut afresh (its `copy` is in the seed of its
        cuts)."""
        everything = np.arange(recording.size, dtype=np.int64)
        if self.passes is None:
            chosen = self._capped(everything, recording, index, _rng(f"{self.seed}/{epoch}/cap/0"))
            return recording[chosen], index[chosen], np.zeros(chosen.size, dtype=np.int64)
        names = sorted({r.dataset_id for r in self.selection.recordings})
        number = {name: k for k, name in enumerate(names)}
        dataset_of = np.array([number[r.dataset_id] for r in self.selection.recordings])
        dataset = dataset_of[recording]
        rng = _rng(f"{self.seed}/{epoch}/passes")
        picked, copies = [], []
        for k, name in enumerate(names):
            found = np.flatnonzero(dataset == k)
            whole, fraction = divmod(self.passes[name], 1.0)
            for c in range(int(whole) + 1):
                kept = self._capped(found, recording, index, _rng(f"{self.seed}/{epoch}/cap/{c}"))
                if c == int(whole):  # the fraction: a random part of one more pass
                    kept = rng.choice(kept, round(fraction * kept.size), replace=False)
                picked.append(kept)
                copies.append(np.full(kept.size, c, dtype=np.int64))
        return (recording[np.concatenate(picked)], index[np.concatenate(picked)],
                np.concatenate(copies))  # fmt: skip

    def _capped(
        self, blocks: np.ndarray, recording: np.ndarray, index: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:  # fmt: skip
        """The positions `blocks` (into `recording`, `index`) left after the subject cap: of
        each subject, random blocks while their seconds stay within the cap; all of them
        without a cap. In the order of `blocks`."""
        if self.subject_hours_per_epoch is None or blocks.size == 0:
            return blocks
        recordings = self.selection.recordings
        subjects = {s: k for k, s in enumerate(sorted({r.subject_id for r in recordings}))}
        subject_of = np.array([subjects[r.subject_id] for r in recordings], dtype=np.int64)
        duration_of = np.array([r.duration_s for r in recordings], dtype=np.float64)
        r = recording[blocks]
        seconds = np.clip(duration_of[r] - index[blocks] * self.block_s, 0.0, self.block_s)
        shuffled = rng.permutation(blocks.size)
        subject = subject_of[r][shuffled]
        by_subject = np.argsort(subject, kind="stable")  # random order within each subject
        ordered = shuffled[by_subject]
        total = np.cumsum(seconds[ordered])
        first_of = np.searchsorted(subject[by_subject], subject[by_subject], side="left")
        before = np.where(first_of > 0, total[first_of - 1], 0.0)
        keep = (total - before) <= self.subject_hours_per_epoch * 3600.0 + 1e-9
        return blocks[np.sort(ordered[keep])]

    def examples(
        self, epoch: int, part: int, parts: int, piece: int = 0, pieces: int = 1
    ) -> Iterator[Example]:
        """The examples of `blocks_of(epoch, part, parts)`, shuffled; with `pieces` > 1
        only piece `piece` of them (every pieces-th block: the pieces of an epoch together
        read each block once). Which examples come is fixed by `seed` and `epoch`; their
        order also depends on which read ends first."""
        if not 0 <= piece < pieces:
            raise ValueError(f"piece {piece} is not in 0..{pieces - 1}")
        blocks = self.blocks_of(epoch, part, parts)[piece::pieces]
        key = f"{self.seed}/{epoch}/{part}" + (f"/{piece}" if pieces > 1 else "")
        rng = random.Random(key)  # noqa: S311 - not secret
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
                        raise ReadError(item.recording.name) from item.error
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
        """All examples of one block in `epoch`, in time order (of their steps)."""
        recording = block.recording
        rng = random.Random(  # noqa: S311 - not secret
            f"{self.seed}/{epoch}/{recording.dataset_id}/{recording.recording_id}/{block.index}"
            + (f"/{block.copy}" if block.copy else "")
        )
        names = {c.array for c in recording.channels}
        arrays = {name: recording.array_blocks(name) for name in names}
        lengths = {c: arrays[c.array].length for c in recording.channels}
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
            longest = [self._longest(g) for g in groups]
            step = min(longest) if self.spans == "shared" else max(longest)
            if first:  # a random part of a step: the cuts differ from epoch to epoch
                step *= 1.0 - rng.random()
                first = False
            stop_s = min(t + step, end_s)
            if self.spans == "shared":
                read = self._read(recording, arrays, channels, t, stop_s)
                for group in groups:
                    yield from self._example(recording, group, read, t)
            else:
                for group, span in zip(groups, longest, strict=True):
                    read = self._read(recording, arrays, group, t, stop_s)
                    at = t
                    while at < stop_s:
                        until = min(at + span, stop_s)
                        part = {c: _piece(read[c], t, at, until) for c in group}
                        yield from self._example(recording, group, part, at)
                        at = until
            t = stop_s

    def _example(
        self, recording: Recording, group: list[Channel], read: dict[Channel, Signal],
        start_s: float,
    ) -> Iterator[Example]:  # fmt: skip
        """The example of `group` from `read` (signals from `start_s`); none if no channel
        has a sample (a span can be too short)."""
        chosen = [c for c in group if read[c].values.size]
        if chosen:
            origin = Origin(
                recording.dataset_id, recording.recording_id, recording.subject_id, start_s,
                tuple(chosen),
            )  # fmt: skip
            yield Example(tuple(read[c] for c in chosen), origin)

    def _longest(self, channels: list[Channel]) -> float:
        """The longest span (s) in which `channels` fit the limit. A span of d seconds holds
        at most rate * d + 1 samples of a channel. Sample tokens: at most
        (rate * d + 1) / samples_per_token + 1 tokens, two more a channel at most. Global
        tokens: a channel's cells with a sample are at most ceil(d / token_s) and at most its
        samples, so at most min(1 / token_s, rate) * d + 1."""
        limit = self.limit
        if isinstance(limit, SampleTokenLimit):
            samples_per_s = sum(c.sampling_rate for c in channels)
            tokens = limit.max_tokens - 2 * len(channels)
            return tokens * limit.samples_per_token / samples_per_s
        by_samples = (limit.max_samples - len(channels)) / sum(c.sampling_rate for c in channels)
        by_tokens = (limit.max_tokens - len(channels)) / sum(
            limit.tokens_per_second(c) for c in channels
        )
        return min(by_samples, by_tokens)

    def _read(
        self,
        recording: Recording,
        arrays: dict[str, blocks.ArrayBlocks],
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
            last = min(arrays[name].length, _first_sample(stop_s, mine[0]))
            stored = blocks.span(arrays[name], first, last, self._getter(recording))
            for c in mine:
                found[c] = Signal(_values(stored[c.index], c), rate, first / rate - start_s)
        return found

    def _getter(self, recording: Recording) -> Callable[[blocks.BlockRef], bytes]:
        """A block's compressed bytes, through the local cache."""
        if self._store is None:
            self._store = self._store_factory()
        if self._cache is None:
            self._cache = BlockCache(self.cache_directory, self.cache_bytes)
        store, cache = self._store, self._cache

        def get(ref: blocks.BlockRef) -> bytes:
            key = f"{recording.standard}/{ref.object}"

            def download() -> bytes:
                return _read_with_retries(lambda: store.get_range(key, ref.offset, ref.nbytes), key)

            return cache.get(key, ref.offset, download)

        return get


READ_ATTEMPTS = 5
RETRY_WAIT_S = 1.0  # then 2, 4, 8 s

# Transport failures, tried again: a timeout, a socket error, and a response cut off in the
# middle. botocore raises the last as ResponseStreamingError (a BotoCoreError, not an
# OSError), and http.client as IncompleteRead (an HTTPException); v1-pretrain stopped at
# step ~201,000 on one such cut read from the S3 storage (2026-10-28). A missing object
# (NoSuchKey, a ClientError) is not tried again.
TRANSIENT = (TimeoutError, OSError, BotoCoreError, HTTPException)


def _read_with_retries(read: Callable[[], bytes], what: str) -> bytes:
    """One ranged read; one that times out or fails in transport is tried again, up to
    READ_ATTEMPTS times, with growing waits (botocore also retries each request; see
    storage.stores)."""
    for attempt in range(1, READ_ATTEMPTS + 1):
        try:
            return read()
        except TRANSIENT as exc:
            if attempt == READ_ATTEMPTS:
                raise
            log.warning("read of %s failed (%s); trying again", what, exc)
            time.sleep(RETRY_WAIT_S * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _rng(key: str) -> np.random.Generator:
    """A random generator seeded by a text (the run's seed, the epoch, ...)."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


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


def _piece(signal: Signal, read_s: float, start_s: float, stop_s: float) -> Signal:
    """The samples of `signal` (read from `read_s`) in [start_s, stop_s), timed from
    `start_s`."""
    rate = signal.sampling_rate
    offset = _sample_at(read_s + signal.start_s, rate)  # the read's first sample
    first = max(0, math.ceil(round(start_s * rate, 9)) - offset)
    stop = max(first, math.ceil(round(stop_s * rate, 9)) - offset)
    values = signal.values[first:stop]
    return Signal(values, rate, (offset + first) / rate - start_s)


def _sample_at(t: float, rate: float) -> int:
    """The index of the sample at `t` seconds (t is a sample time)."""
    return round(t * rate)


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


def _bytes(store: Store, key: str) -> bytes:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(key)
    return data


def _table(store: Store, key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = pq.read_table(io.BytesIO(_bytes(store, key))).to_pylist()
    return rows

"""The v4 batch format (docs/architecture.md, sections 3, 4 and 7): every sample of every
channel, a channel table and a token table, with the masks applied to the samples.

A batch holds the channels of packed examples:

- one flat sample buffer: the samples of every channel, one channel after the other, at the
  channel's own rate, with a flag per sample (observed in the source and not hidden) and a
  hidden flag;
- a channel table: per channel its row and document (example within the row), rate, the time
  of its first sample from the example's start, its offset and length in the buffer, its signal
  type, and the optional descriptor fields (docs/architecture.md section 5; an empty placeholder
  until the index has them);
- a token table: per global token its channel, its time span [t0, t1) and centre in seconds
  (float64) from the example's start, whether it is hidden, and its place in its row of global
  tokens; and per sample the token that holds it.

Global tokens (section 4): one grid of `token_s` seconds from the example's start, the same for
every channel and rate; a channel's token is a cell of the grid that holds at least one of its
samples (a channel slower than one sample a cell has one token a sample, not empty cells). The
token table is what the fixed chunker gives (biosignal_model.chunking); a learned chunker will
place the boundaries in the model instead.

Masks (section 7), for masked pretraining: a set ratio of each example's global tokens is
hidden, in four kinds with set shares: cycle combs, time spans over all channels together,
channel groups over a time window, and single tokens. Span lengths are log-uniform between two
global bounds in seconds; nothing depends on the signal type. A cycle comb hides the same phase
of every cycle of the example (all channels together): with a repeating signal (the beats of an
ECG) a span leaves the same phase visible in the other cycles, and a copy from there solves
most of the task (2026-11-04: a copy of the beat before explains 88 % of a PTB-XL QRS); a comb
leaves only the cycle's own other phases. The period comes from the autocorrelation of the
example's own energy envelope (`cycle_period`); without a clear period the comb's share goes to
time spans. The mask then depends on the values through the period only. A token is in a span
when its centre is. The
hidden tokens are then applied to the SAMPLES: every sample of a hidden token is set to 0 and
flagged hidden, before any part of the model, so no model part (the local stage sees the whole
crop) can read them. The values before masking are kept apart, in `ChannelTargets`, for the
loss only.

Packing: examples are put whole, in order, into rows of at most `max_tokens` global tokens and
`max_samples` samples (the two limits of the loader's spans); `rows` rows make a batch.

Values (as v0-v2): per channel of an example, x' = asinh((x - m) / s) with m and s the mean
and standard deviation of the channel's VISIBLE valid samples (s = 1 if they are constant), so
nothing about a hidden sample enters the input through them; asinh(m) and log(s) go to the model
as the channel's level and scale, so the amplitude is not lost. NaN (invalid) -> 0 and not
observed. The teacher of the latent objective reads the same transform of every valid sample
(`ChannelBatch.teacher_inputs`: nothing hidden).

Each channel of an example gets a random slot in [0, max_channels), distinct within the
example (as v0-v2): it tells the channels of one example apart, and says nothing else.
"""

from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, fields
from typing import Any, Literal, TypeVar

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict

from biosignal_data.index.signals import SignalType
from biosignal_data.training.loader import Example, Signal

SIGNAL_TYPES = [t.value for t in SignalType]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MaskSpec(_Spec):
    """Which global tokens of an example are hidden; none of it has a default."""

    ratio: float  # of the example's global tokens, in [0, 1); at least one token stays visible
    # shares of the hidden tokens by kind; they sum to 1
    cycles: float  # cycle combs: the same phase of every cycle, over all channels together
    time_spans: float  # time spans hidden over all channels of the example together
    channel_groups: float  # a group of channels hidden over a time window
    single_tokens: float  # single tokens at random
    time_span_s: tuple[float, float]  # bounds of the log-uniform span length (s)
    group_span_s: tuple[float, float]  # bounds of the log-uniform window of a channel group (s)
    cycle_period_s: tuple[float, float]  # the periods a comb looks for (s)
    cycle_tooth: tuple[float, float]  # bounds of a tooth's uniform width, as a part of the period
    cycle_strength: float  # least autocorrelation at the period for a comb, in (0, 1)


class ChannelTokens(_Spec):
    """The v4 batch format (`tokens.format: channels` in a train config)."""

    format: Literal["channels"]
    token_s: float  # the global token duration (s), the same for every channel and rate
    max_tokens: int  # global tokens per row and per example
    max_samples: int  # samples per row and per example (what the local stage reads)
    max_channels: int  # channels per example
    rows: int  # rows in one step (over all GPUs)
    mask: MaskSpec | None  # None: nothing hidden


@dataclass
class ChannelTable:
    """Per channel of the batch, [C]."""

    row: torch.Tensor  # int64
    document: torch.Tensor  # int64: the example within its row
    rate_hz: torch.Tensor  # float64
    start_s: torch.Tensor  # float64: the first sample's time from the example's start
    offset: torch.Tensor  # int64: the first sample in the flat buffer
    length: torch.Tensor  # int64: samples
    signal_type: torch.Tensor  # int64: index in SIGNAL_TYPES (a descriptor field)
    slot: torch.Tensor  # int64: a random slot, distinct within the example
    level: torch.Tensor  # float32: asinh of the mean of the visible valid samples
    log_scale: torch.Tensor  # float32: log of their standard deviation (0 if constant)
    # Optional descriptor fields (section 5): name -> values [C, ...] and present [C] (bool).
    # Empty until the index has them; an absent field adds nothing.
    descriptors: dict[str, torch.Tensor] = field(default_factory=dict)
    present: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class TokenTable:
    """Per global token of the batch, [T], in the order of the rows; and per sample its
    token, [S]."""

    channel: torch.Tensor  # int64: index in the channel table
    t0: torch.Tensor  # float64: start of the token's span (s, from the example's start)
    t1: torch.Tensor  # float64: its end
    centre: torch.Tensor  # float64
    hidden: torch.Tensor  # bool
    row: torch.Tensor  # int64
    position: torch.Tensor  # int64: the token's place in its row
    sample_token: torch.Tensor  # int64 [S]: the token of each sample


@dataclass
class ChannelInputs:
    """Everything the model may read."""

    values: torch.Tensor  # float32 [S]: 0 where hidden or not observed
    observed: torch.Tensor  # bool [S]: observed in the source and not hidden
    hidden: torch.Tensor  # bool [S]
    channels: ChannelTable
    tokens: TokenTable
    rows: int
    max_tokens: int  # global tokens a row can hold

    def to(self, device: torch.device) -> ChannelInputs:
        return ChannelInputs(
            values=self.values.to(device, non_blocking=True),
            observed=self.observed.to(device, non_blocking=True),
            hidden=self.hidden.to(device, non_blocking=True),
            channels=_moved(self.channels, device),
            tokens=_moved(self.tokens, device),
            rows=self.rows,
            max_tokens=self.max_tokens,
        )


@dataclass
class ChannelTargets:
    """For the loss and the teacher only; never an input of the student."""

    values: torch.Tensor  # float32 [S]: the samples before masking; 0 where not observed
    observed: torch.Tensor  # bool [S]: observed in the source
    # The teacher's input: the transform of every valid sample with the mean and standard
    # deviation of ALL the channel's valid samples, and its level and log scale [C]. (The
    # student's come from its visible samples, so nothing hidden reaches it; the teacher sees
    # everything, so with the student's statistics its targets changed with the mask: by up to
    # 7.7 on unit-variance targets between two masks of one example, 2026-11-04.)
    normalized: torch.Tensor  # float32 [S]
    level: torch.Tensor  # float32 [C]
    log_scale: torch.Tensor  # float32 [C]


@dataclass
class ChannelAnalysis:
    """Per channel, for reports only: never an input of the model."""

    dataset: torch.Tensor  # int64 [C]: index in `ChannelPacker.datasets`


@dataclass
class ChannelBatch:
    inputs: ChannelInputs
    targets: ChannelTargets
    analysis: ChannelAnalysis
    examples: int
    samples: int  # observed samples in the batch

    def to(self, device: torch.device) -> ChannelBatch:
        t = self.targets
        targets = ChannelTargets(
            values=t.values.to(device, non_blocking=True),
            observed=t.observed.to(device, non_blocking=True),
            normalized=t.normalized.to(device, non_blocking=True),
            level=t.level.to(device, non_blocking=True),
            log_scale=t.log_scale.to(device, non_blocking=True),
        )
        analysis = ChannelAnalysis(self.analysis.dataset.to(device, non_blocking=True))
        return ChannelBatch(self.inputs.to(device), targets, analysis, self.examples, self.samples)

    def teacher_inputs(self) -> ChannelInputs:
        """The input of the latent objective's teacher: every valid sample, nothing hidden,
        transformed with the statistics of all of them (the tables are the student's but for
        the levels and scales). It does not depend on the mask."""
        x, t = self.inputs, self.targets
        return ChannelInputs(
            values=t.normalized, observed=t.observed, hidden=torch.zeros_like(x.hidden),
            channels=dataclasses.replace(x.channels, level=t.level, log_scale=t.log_scale),
            tokens=dataclasses.replace(x.tokens, hidden=torch.zeros_like(x.tokens.hidden)),
            rows=x.rows, max_tokens=x.max_tokens,
        )  # fmt: skip


def _moved(table: _Table, device: torch.device) -> _Table:
    out: dict[str, Any] = {}
    for f in fields(table):
        v = getattr(table, f.name)
        out[f.name] = (
            {k: t.to(device, non_blocking=True) for k, t in v.items()}
            if isinstance(v, dict)
            else v.to(device, non_blocking=True)
        )
    return type(table)(**out)


_Table = TypeVar("_Table", "ChannelTable", "TokenTable")


def apply_sample_mask(
    values: torch.Tensor, valid: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's input from the samples: (values, observed) with every hidden or invalid
    sample set to 0 and not observed. The one place where the mask meets the samples (the
    packer and the leak tests use it)."""
    observed = valid & ~hidden
    return torch.where(observed, values, torch.zeros_like(values)), observed


def normalize(values: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, float, float]:
    """asinh((x - m) / s) of a channel's samples (0 where not finite), with m and s the mean
    and standard deviation of the `visible` finite samples (s = 1 when they are constant or
    there are none); and the channel's level asinh(m) and log(s)."""
    finite = np.isfinite(values)
    seen = values[visible & finite].astype(np.float64)
    mean = float(seen.mean()) if seen.size else 0.0
    std = float(seen.std()) if seen.size else 0.0
    scale = std if std > 0 else 1.0
    z = np.where(finite, np.arcsinh((values - mean) / scale), 0.0).astype(np.float32)
    return z, math.asinh(mean), math.log(scale)


def token_cells(start_s: float, rate: float, length: int, token_s: float) -> np.ndarray:
    """The grid cell of each sample of a channel: floor(time / token_s), the time from the
    example's start in float64 (rounded to 1e-9 of a cell, so a sample on a boundary is in
    the cell it starts)."""
    times = start_s + np.arange(length, dtype=np.float64) / rate
    return np.floor(np.round(times / token_s, 9)).astype(np.int64)


@dataclass
class _Example:
    """One example, before packing."""

    values: list[np.ndarray]  # per channel, float32, NaN where invalid
    rates: list[float]
    starts: list[float]
    types: list[int]
    dataset: int
    cells: list[np.ndarray]  # per channel: the cell of each token (sorted, unique)
    sample_token: list[np.ndarray]  # per channel: the token (index in its cells) of each sample
    hidden: list[np.ndarray]  # per channel: hidden flag of each token
    slots: list[int]  # per channel: its random slot

    @property
    def tokens(self) -> int:
        return sum(c.size for c in self.cells)

    @property
    def samples(self) -> int:
        return sum(v.size for v in self.values)


class ChannelPacker:
    """Turns a stream of examples into ChannelBatches. Every argument is the experiment's
    choice; none has a default."""

    def __init__(
        self,
        *,
        datasets: list[str],
        token_s: float,
        max_tokens: int,
        max_samples: int,
        max_channels: int,
        rows: int,
        mask: MaskSpec | None,
        seed: int,
    ) -> None:
        if token_s <= 0 or max_tokens < 1 or max_samples < 1 or rows < 1:
            raise ValueError("token_s, max_tokens, max_samples and rows are positive")
        if mask is not None:
            _check(mask)
        self.datasets = datasets
        self._dataset_index = {d: i for i, d in enumerate(datasets)}
        self.token_s = token_s
        self.max_tokens = max_tokens
        self.max_samples = max_samples
        self.max_channels = max_channels
        self.rows = rows
        self.mask = mask
        self._rng = random.Random(seed)  # noqa: S311 - not secret

    def reseed(self, seed: int) -> None:
        """Start the random choices (hidden tokens) again from `seed`."""
        self._rng = random.Random(seed)  # noqa: S311 - not secret

    def example(self, example: Example) -> _Example:
        """The tokens and the mask of one example."""
        if len(example.signals) > self.max_channels:
            raise ValueError(
                f"{len(example.signals)} channels; max_channels is {self.max_channels}"
            )
        cells, sample_token = [], []
        for s in example.signals:
            c = token_cells(s.start_s, s.sampling_rate, s.values.size, self.token_s)
            unique, inverse = np.unique(c, return_inverse=True)
            cells.append(unique)
            sample_token.append(inverse.astype(np.int64))
        hidden = [np.zeros(c.size, dtype=bool) for c in cells]
        if self.mask is not None:
            rng = np.random.default_rng(self._rng.getrandbits(64))
            m = self.mask
            period = (
                cycle_period(example.signals, m.cycle_period_s, m.cycle_strength, self.token_s)
                if m.cycles > 0 else None
            )  # fmt: skip
            hidden = hide(cells, self.token_s, m, rng, groups=None, period_s=period)
        slots = self._rng.sample(range(self.max_channels), len(example.signals))
        return _Example(
            values=[s.values for s in example.signals],
            rates=[s.sampling_rate for s in example.signals],
            starts=[s.start_s for s in example.signals],
            types=[SIGNAL_TYPES.index(c.signal_type) for c in example.origin.channels],
            dataset=self._dataset_index[example.origin.dataset_id],
            cells=cells, sample_token=sample_token, hidden=hidden, slots=slots,
        )  # fmt: skip

    def _fits(self, e: _Example) -> None:
        if e.tokens > self.max_tokens or e.samples > self.max_samples:
            raise ValueError(
                f"an example of {e.tokens} tokens and {e.samples} samples is longer than "
                f"max_tokens {self.max_tokens} or max_samples {self.max_samples}"
            )

    def batches(self, examples: Iterable[Example]) -> Iterator[ChannelBatch]:
        rows: list[list[_Example]] = [[]]
        tokens = samples = count = 0
        for example in examples:
            e = self.example(example)
            self._fits(e)
            if tokens + e.tokens > self.max_tokens or samples + e.samples > self.max_samples:
                if len(rows) == self.rows:
                    yield self._batch(rows, count)
                    rows, count = [], 0
                rows.append([])
                tokens = samples = 0
            rows[-1].append(e)
            tokens, samples, count = tokens + e.tokens, samples + e.samples, count + 1
        if count:
            yield self._batch(rows, count)

    def pack(
        self, examples: list[Example], *, trim: bool = True
    ) -> tuple[ChannelBatch, list[tuple[int, int]]]:
        """All `examples` in one batch, in order, and the (row, document) of each. Raises if
        they do not fit `rows` rows. `trim`: as Packer.pack's (a channel batch is flat, it has
        no padding to trim: the same batch either way)."""
        rows: list[list[_Example]] = [[]]
        where: list[tuple[int, int]] = []
        tokens = samples = 0
        for example in examples:
            e = self.example(example)
            self._fits(e)
            if tokens + e.tokens > self.max_tokens or samples + e.samples > self.max_samples:
                rows.append([])
                tokens = samples = 0
            if len(rows) > self.rows:
                raise ValueError(f"{len(examples)} examples do not fit {self.rows} rows")
            where.append((len(rows) - 1, len(rows[-1])))
            rows[-1].append(e)
            tokens, samples = tokens + e.tokens, samples + e.samples
        return self._batch(rows, len(examples)), where

    def _batch(self, rows: list[list[_Example]], count: int) -> ChannelBatch:
        ch: dict[str, list[float | int]] = {
            k: [] for k in ("row", "document", "rate", "start", "offset", "length", "type",
                            "dataset", "slot", "level", "log_scale", "t_level", "t_log_scale")
        }  # fmt: skip
        values: list[np.ndarray] = []
        normalized: list[np.ndarray] = []
        teacher: list[np.ndarray] = []
        tok: dict[str, list[np.ndarray]] = {
            k: [] for k in ("channel", "cell", "hidden", "row", "position")
        }
        sample_token, sample_hidden = [], []
        offset = channel = token = 0
        for r, row in enumerate(rows):
            position = 0
            for d, e in enumerate(row):
                for k, v in enumerate(e.values):
                    visible_hidden = e.hidden[k][e.sample_token[k]]
                    z, level, log_scale = normalize(v, ~visible_hidden)
                    normalized.append(z)
                    zt, t_level, t_log_scale = normalize(v, np.ones(v.size, dtype=bool))
                    teacher.append(zt)
                    for name, value in (("row", r), ("document", d), ("rate", e.rates[k]),
                                        ("start", e.starts[k]), ("offset", offset),
                                        ("length", v.size), ("type", e.types[k]),
                                        ("dataset", e.dataset), ("slot", e.slots[k]),
                                        ("level", level), ("log_scale", log_scale),
                                        ("t_level", t_level),
                                        ("t_log_scale", t_log_scale)):  # fmt: skip
                        ch[name].append(value)
                    n = e.cells[k].size
                    tok["channel"].append(np.full(n, channel, dtype=np.int64))
                    tok["cell"].append(e.cells[k])
                    tok["hidden"].append(e.hidden[k])
                    tok["row"].append(np.full(n, r, dtype=np.int64))
                    tok["position"].append(position + np.arange(n, dtype=np.int64))
                    sample_token.append(token + e.sample_token[k])
                    sample_hidden.append(visible_hidden)
                    values.append(v)
                    offset += v.size
                    channel += 1
                    token += n
                    position += n
        raw = np.concatenate(values).astype(np.float32) if values else np.zeros(0, np.float32)
        valid = torch.from_numpy(np.isfinite(raw))
        raw_t = torch.from_numpy(np.where(np.isfinite(raw), raw, 0.0).astype(np.float32))
        normalized_t = torch.from_numpy(_cat(normalized, np.float32))
        hidden = torch.from_numpy(_cat(sample_hidden, bool))
        inputs_values, observed = apply_sample_mask(normalized_t, valid, hidden)
        cell = _cat(tok["cell"], np.int64)
        t0 = cell.astype(np.float64) * self.token_s

        def i64(name: str) -> torch.Tensor:
            return torch.tensor(ch[name], dtype=torch.int64)

        channels = ChannelTable(
            row=i64("row"), document=i64("document"),
            rate_hz=torch.tensor(ch["rate"], dtype=torch.float64),
            start_s=torch.tensor(ch["start"], dtype=torch.float64),
            offset=i64("offset"), length=i64("length"), signal_type=i64("type"),
            slot=i64("slot"), level=torch.tensor(ch["level"], dtype=torch.float32),
            log_scale=torch.tensor(ch["log_scale"], dtype=torch.float32),
        )  # fmt: skip
        tokens = TokenTable(
            channel=torch.from_numpy(_cat(tok["channel"], np.int64)),
            t0=torch.from_numpy(t0), t1=torch.from_numpy(t0 + self.token_s),
            centre=torch.from_numpy(t0 + self.token_s / 2),
            hidden=torch.from_numpy(_cat(tok["hidden"], bool)),
            row=torch.from_numpy(_cat(tok["row"], np.int64)),
            position=torch.from_numpy(_cat(tok["position"], np.int64)),
            sample_token=torch.from_numpy(_cat(sample_token, np.int64)),
        )  # fmt: skip
        inputs = ChannelInputs(
            values=inputs_values, observed=observed, hidden=hidden, channels=channels,
            tokens=tokens, rows=len(rows), max_tokens=self.max_tokens,
        )  # fmt: skip
        return ChannelBatch(
            inputs=inputs,
            targets=ChannelTargets(
                values=raw_t, observed=valid,
                normalized=torch.from_numpy(_cat(teacher, np.float32)),
                level=torch.tensor(ch["t_level"], dtype=torch.float32),
                log_scale=torch.tensor(ch["t_log_scale"], dtype=torch.float32),
            ),
            analysis=ChannelAnalysis(dataset=i64("dataset")), examples=count,
            samples=int(valid.sum()),
        )  # fmt: skip


def _cat(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)


def concat(batches: list[ChannelBatch]) -> ChannelBatch:
    """One batch of the rows of `batches`, in order (each batch's rows after the last's)."""
    if len(batches) == 1:
        return batches[0]
    rows = offset = channels = tokens = 0
    parts: dict[str, list[torch.Tensor]] = {}

    def add(name: str, value: torch.Tensor) -> None:
        parts.setdefault(name, []).append(value)

    for b in batches:
        x, c, t = b.inputs, b.inputs.channels, b.inputs.tokens
        for name in ("values", "observed", "hidden"):
            add(name, getattr(x, name))
        add("target_values", b.targets.values)
        add("target_observed", b.targets.observed)
        add("target_normalized", b.targets.normalized)
        add("target_level", b.targets.level)
        add("target_log_scale", b.targets.log_scale)
        add("dataset", b.analysis.dataset)
        add("c.row", c.row + rows)
        add("c.offset", c.offset + offset)
        for name in ("document", "rate_hz", "start_s", "length", "signal_type", "slot", "level",
                     "log_scale"):  # fmt: skip
            add(f"c.{name}", getattr(c, name))
        add("t.channel", t.channel + channels)
        add("t.row", t.row + rows)
        add("t.sample_token", t.sample_token + tokens)
        for name in ("t0", "t1", "centre", "hidden", "position"):
            add(f"t.{name}", getattr(t, name))
        rows += x.rows
        offset += x.values.numel()
        channels += c.row.numel()
        tokens += t.channel.numel()
        if c.descriptors:
            raise NotImplementedError("joining batches with descriptor fields")
    joined: dict[str, Any] = {k: torch.cat(v) for k, v in parts.items()}
    table = ChannelTable(**{k[2:]: v for k, v in joined.items() if k.startswith("c.")})
    token_table = TokenTable(**{k[2:]: v for k, v in joined.items() if k.startswith("t.")})
    inputs = ChannelInputs(
        values=joined["values"], observed=joined["observed"], hidden=joined["hidden"],
        channels=table, tokens=token_table, rows=rows, max_tokens=batches[0].inputs.max_tokens,
    )  # fmt: skip
    return ChannelBatch(
        inputs=inputs,
        targets=ChannelTargets(
            joined["target_values"], joined["target_observed"], joined["target_normalized"],
            joined["target_level"], joined["target_log_scale"],
        ),
        analysis=ChannelAnalysis(joined["dataset"]),
        examples=sum(b.examples for b in batches), samples=sum(b.samples for b in batches),
    )  # fmt: skip


def _check(mask: MaskSpec) -> None:
    if not 0.0 <= mask.ratio < 1.0:
        raise ValueError("mask.ratio is in [0, 1)")
    shares = (mask.cycles, mask.time_spans, mask.channel_groups, mask.single_tokens)
    if min(shares) < 0 or abs(sum(shares) - 1.0) > 1e-9:
        raise ValueError("the mask shares are not negative and sum to 1")
    for low, high in (mask.time_span_s, mask.group_span_s, mask.cycle_period_s):
        if not 0 < low <= high:
            raise ValueError("span and period bounds: 0 < low <= high")
    if not 0 < mask.cycle_tooth[0] <= mask.cycle_tooth[1] < 1:
        raise ValueError("cycle_tooth: 0 < low <= high < 1")
    if not 0 < mask.cycle_strength < 1:
        raise ValueError("cycle_strength is in (0, 1)")


def cycle_period(
    signals: Iterable[Signal], bounds: tuple[float, float], strength: float, smooth_s: float
) -> float | None:
    """The period (s) of an example's repeating part, or None: the lag in `bounds` where the
    autocorrelation of the energy envelope (the absolute first difference, a moving mean over
    `smooth_s`: one global token), averaged over the channels on a 1 ms grid of lags, is
    highest; the first lag within 10 % of that peak is taken (so twice the period is not). None
    when that peak is below `strength`, or no channel holds two periods. Invalid (NaN) samples
    count as 0. On PTB-XL (folds 1-8, 500 Hz, 2026-11-04) the envelope found a period of
    strength >= 0.3 in 91 % of the records (non-AF 96 %), within 40 ms of the median RR in
    91 %; the autocorrelation of the first difference itself in 37 % (noise dominates it)."""
    grid = np.arange(bounds[0], bounds[1] + 5e-4, 1e-3)
    total, count = np.zeros(grid.size), 0
    for s in signals:
        x = np.abs(np.diff(np.nan_to_num(np.asarray(s.values, dtype=np.float64))))
        longest = math.ceil(bounds[1] * s.sampling_rate) + 1
        if x.size < 2 * longest or not np.any(x):
            continue
        width = max(1, round(smooth_s * s.sampling_rate))
        x = np.convolve(x, np.ones(width) / width, mode="same")
        x = x - x.mean()
        n = 1 << (2 * x.size - 1).bit_length()
        spectrum = np.fft.rfft(x, n)
        acf = np.fft.irfft(spectrum * np.conj(spectrum), n)[: longest + 1]
        if acf[0] <= 0:
            continue
        total += np.interp(grid * s.sampling_rate, np.arange(acf.size), acf / acf[0])
        count += 1
    if count == 0:
        return None
    mean = total / count
    peak = float(np.max(mean))
    if peak < strength:
        return None
    return float(grid[int(np.argmax(mean >= 0.9 * peak))])


# Windows drawn per kind at most (each draw hides at least one new token, or the kind ends:
# then single tokens fill what is left of the ratio).
MAX_WINDOWS = 10_000


def hide(
    cells: list[np.ndarray],
    token_s: float,
    mask: MaskSpec,
    rng: np.random.Generator,
    groups: list[object | None] | None,
    period_s: float | None = None,
) -> list[np.ndarray]:
    """Hidden flags of the tokens of one example (`cells`: per channel, the grid cell of each
    token). About round(ratio x tokens) tokens are hidden (at least one stays visible), in the
    kinds' shares: cycle combs, time spans over all channels, channel groups over a time
    window, single tokens. A comb (with the example's period `period_s`; None: the combs'
    share goes to the time spans) hides every token whose centre's phase, (centre - phase0)
    mod period, is below its tooth width (a uniform part of the period in cycle_tooth), with
    phase0 uniform in [0, period): the same phase of every cycle; combs are drawn until their
    share is reached. A span or window of length L starts uniformly in [-L, end), so every time is
    covered alike; a token is in it when its centre is. A span or window is hidden in whole
    grid cells: at a cell, all its channels (a time span: every channel of the example; a
    group window: every channel of the group) or none, so no hidden time is visible on another
    channel of the span. The window that reaches a kind's share ends at the cell that reaches
    it (a kind can pass its share by the channels of one cell). `groups`: per channel a group
    label (a descriptor field's value) or None; channels without a label, or `groups` None,
    are grouped at random (a random subset of 1..C-1 channels; the only channel if C = 1)."""
    total = sum(c.size for c in cells)
    hidden = [np.zeros(c.size, dtype=bool) for c in cells]
    if total < 2 or mask.ratio == 0.0:
        return hidden
    goal = min(total - 1, round(mask.ratio * total))
    h = _Hider(cells, token_s, rng)
    spans = round((mask.cycles + mask.time_spans) * goal)  # combs first; spans take the rest
    groups_budget = round(mask.channel_groups * goal)
    singles = goal - spans - groups_budget
    if period_s is not None and mask.cycles > 0:
        h.combs(round(mask.cycles * goal), period_s, mask.cycle_tooth)
    h.windows(spans - h.count(), mask.time_span_s, None)
    labels: list[object | None] = list(groups) if groups is not None else [None] * len(cells)
    h.windows(groups_budget, mask.group_span_s, lambda: _random_group(labels, rng))
    if mask.single_tokens > 0:
        h.singles(min(singles, total - 1 - h.count()))
    at = 0
    for k, c in enumerate(cells):
        hidden[k] = h.flat[at : at + c.size].copy()
        at += c.size
    return hidden


def _random_group(labels: list[object | None], rng: np.random.Generator) -> np.ndarray:
    """The channels of a labelled group, or a random subset of 1..C-1 of the unlabelled ones
    (the only one if there is one)."""
    labelled = sorted({g for g in labels if g is not None}, key=repr)
    chosen = rng.integers(len(labelled) + 1) if labelled else len(labelled)
    if chosen < len(labelled):
        return np.array([k for k, g in enumerate(labels) if g == labelled[chosen]])
    free = [k for k, g in enumerate(labels) if g is None]
    if len(free) <= 1:
        return np.array(free)
    size = int(rng.integers(1, len(free)))
    return rng.choice(np.array(free), size, replace=False)


class _Hider:
    """One example's hidden flags while `hide` draws its kinds: every token of every channel
    in one flat order, with its channel and centre time."""

    def __init__(self, cells: list[np.ndarray], token_s: float, rng: np.random.Generator) -> None:
        self.rng = rng
        self.total = sum(c.size for c in cells)
        self.channel = np.concatenate([np.full(c.size, k) for k, c in enumerate(cells)])
        self.centre = (np.concatenate(cells).astype(np.float64) + 0.5) * token_s
        self.end = float(np.max(self.centre)) + token_s / 2
        self.flat = np.zeros(self.total, dtype=bool)

    def count(self) -> int:
        return int(self.flat.sum())

    def _take(self, inside: np.ndarray, done: int, budget: int) -> tuple[int, bool]:
        """Hides the tokens `inside` in whole cells in time order until `budget` (one token
        stays visible): the new count, and whether the kind is to end."""
        found = np.flatnonzero(inside & ~self.flat)
        found = found[np.argsort(self.centre[found], kind="stable")]
        cuts = np.flatnonzero(np.diff(self.centre[found])) + 1
        for part in np.split(found, cuts) if found.size else []:
            if done >= budget or self.count() + part.size > self.total - 1:
                return done, True
            self.flat[part] = True
            done += part.size
        return done, done >= budget

    def windows(
        self, budget: int, bounds: tuple[float, float], members: Callable[[], np.ndarray] | None
    ) -> None:
        """Spans (members None: every channel) or channel-group windows of log-uniform length
        in `bounds`, each from a uniform start in [-length, end), until `budget`."""
        done = 0
        for _ in range(MAX_WINDOWS):
            if done >= budget:
                return
            length = math.exp(self.rng.uniform(math.log(bounds[0]), math.log(bounds[1])))
            start = self.rng.uniform(-length, self.end)
            inside = (self.centre >= start) & (self.centre < start + length)
            if members is not None:
                inside &= np.isin(self.channel, members())
            done, stop = self._take(inside, done, budget)
            if stop:
                return

    def combs(self, budget: int, period: float, tooth: tuple[float, float]) -> None:
        """Whole combs only (a comb cut at the budget would leave its phase visible in the
        later cycles): a comb that passes the budget gets a narrower tooth at its phase. A
        cell is hidden on all its channels, so a comb hides whole cells."""
        done, cycles = 0, max(1, int(self.end / period))
        for _ in range(MAX_WINDOWS):
            left = budget - done
            if left < cycles:  # not one cell a cycle is left: the rest goes to time spans
                return
            width = self.rng.uniform(*tooth) * period
            phase = self.rng.uniform(0.0, period)
            inside = (np.mod(self.centre - phase, period) < width) & ~self.flat
            if int(inside.sum()) > left:
                width *= left / int(inside.sum())
                inside = (np.mod(self.centre - phase, period) < width) & ~self.flat
            n = int(inside.sum())
            if n == 0:
                continue
            if self.count() + n > self.total - 1:
                return
            self.flat[inside] = True
            done += n

    def singles(self, count: int) -> None:
        if count > 0:
            chosen = self.rng.choice(np.flatnonzero(~self.flat), count, replace=False)
            self.flat[chosen] = True

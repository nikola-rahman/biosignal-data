"""From examples to batches: tokens, masks and packing.

A token is `samples_per_token` consecutive samples of one channel (the last token of a
channel holds what is left, the rest is marked as not observed). For each token the model
gets, and nothing else:

- its samples in the scale of its channel in this example: asinh((x - mean) / std)
  (below), with a flag per sample: observed or not (invalid in the source, or past the
  end of the channel);
- the time of its centre and its length, in seconds from the start of the example;
- a channel slot: a number drawn at random per example, which only says "same channel"
  (or, with `slots`, a fixed number per channel name: for fine-tuning on one montage);
- the scale of its channel: the mean and standard deviation of the channel's visible
  samples, so nothing is lost: x = sinh(value) * std + mean.

Why the division: values are physical, in the units of each source, so they span many
orders of magnitude (EEG in V is 1e-5, CTG in bpm is 1e2). Dividing by a channel's own
scale makes them comparable for the network, and giving the scale to the network as an
input keeps every value recoverable. It is the network's input encoding, not a change of
the data: the loader and the stored data stay as they are. The statistics come from the
visible tokens only, so masked samples do not leak into them; at inference nothing is
hidden, so they come from all tokens.

Why asinh after the division: a channel that is flat in its visible tokens (SpO2 at 97 %
for minutes, a heart rate, a position) has a standard deviation near 0, so a change in a
hidden token becomes 1e6 after the division, and its square ruled the loss of a whole
batch (seen on 2026-10-20, capslpdb SpO2 and HR: 1e12 against about 1 for other types).
asinh is the identity near 0 and grows like a logarithm beyond: 1e6 becomes 14.5. It can
be reversed exactly, and it needs no metadata and no hidden data. Its cost: the loss
weighs errors in the size of large changes about on a log scale. The same encoding is
used in training and at inference: it is part of the model, not of the data.

Masking (for masked-token pretraining): in each channel with two or more tokens,
`mask_ratio` of the tokens are hidden at random, at least one token stays visible.

Packing: examples are put whole, in the order they come, into rows of `max_tokens`
tokens, as in LLM training; `document` says which example each token belongs to, and the
model attends only within one example. Padding at the end of a row has document -1.

`Analysis` holds what the model must not see (dataset, signal type), to split the loss
for reports.

Tokenizers (`Tokenizer`, docs/v3-tokenizer.md): `samples` is the above; `duration` and
`hierarchical` cut each channel into frames of a fixed duration in seconds (whole samples at
its rate), `frames_per_token` frames per token, stored as `[rows, tokens, frames, samples]`
with each token's rate, channel within the row and position within its channel.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, fields
from typing import Annotated, Literal, TypeVar

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from biosignal_data.index.signals import SignalType
from biosignal_data.training.loader import Example

SIGNAL_TYPES = [t.value for t in SignalType]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SamplesTokenizer(_Spec):
    """v0-v2: `samples_per_token` consecutive samples per token, one linear map."""

    kind: Literal["samples"]


class _StemSpec(_Spec):
    """The continuous-time stem (docs/v3-tokenizer.md): `stem_windows` Hann sub-windows per
    frame, `stem_channels` outputs each, kernels on `kernel_frequencies` sinusoids log-spaced
    from `kernel_min_hz` to `kernel_max_hz` (the ones at or above a channel's Nyquist frequency
    are left out)."""

    stem_windows: int
    stem_channels: int
    kernel_frequencies: int
    kernel_min_hz: float
    kernel_max_hz: float


class DurationTokenizer(_StemSpec):
    """D1: tokens of `token_s` every `hop_s` seconds, through the stem."""

    kind: Literal["duration"]
    token_s: float
    hop_s: float


class HierarchicalTokenizer(_StemSpec):
    """D2: frames of `frame_s` every `frame_hop_s` seconds through the stem, `local_layers`
    bidirectional S4D blocks (`local_width`, `local_states`) over each channel's frames,
    attention pooling (`pool_heads`) of `frames_per_token` frames into one token, and the
    token's frames pooled again (`residual_pool`) for the output residual."""

    kind: Literal["hierarchical"]
    frame_s: float
    frame_hop_s: float
    frames_per_token: int
    local_width: int
    local_layers: int
    local_states: int
    pool_heads: int
    # How the token's local frame features reach the output residual: their mean (the first
    # bench) or attention pooling (D2', 2026-10-29: the mean loses where in the token a feature is)
    residual_pool: Literal["mean", "attention"]


Tokenizer = Annotated[
    SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer, Field(discriminator="kind")
]


def frame_shape(
    tokenizer: DurationTokenizer | HierarchicalTokenizer, rate: float
) -> tuple[int, int, int]:
    """(samples per frame, hop in samples, frames per token) at `rate`: whole samples, at
    least one; the same in the packer and the model."""
    if isinstance(tokenizer, DurationTokenizer):
        length, hop, frames = tokenizer.token_s, tokenizer.hop_s, 1
    else:
        length, hop, frames = tokenizer.frame_s, tokenizer.frame_hop_s, tokenizer.frames_per_token
    n, h = max(1, round(length * rate)), max(1, round(hop * rate))
    if h > n:
        raise ValueError("a frame must be at least as long as its hop")
    return n, h, frames


@dataclass
class Inputs:
    """The model's input, [rows, max_tokens, ...]; nothing else goes into the model."""

    values: torch.Tensor  # float32 [R, L, P]: samples / channel scale; 0 where not observed
    observed: torch.Tensor  # bool [R, L, P]
    hidden: torch.Tensor  # bool [R, L]: masked token, the model does not see its samples
    # float64 [R, L]: centre of the token, from the example's start (float32 is off by 6e-5 s
    # at 30 min, too coarse for rotary angles at 10 ms wavelengths)
    time_s: torch.Tensor
    log_length_s: torch.Tensor  # float32 [R, L]: log of the token's length in seconds
    channel_scale: torch.Tensor  # float32 [R, L, 2]: asinh(mean), log(std) of the channel
    slot: torch.Tensor  # int64 [R, L]
    document: torch.Tensor  # int64 [R, L]: example within the row; -1 is padding
    # duration / hierarchical tokenizers only (values, observed are [R, L, F, P] there):
    rate_hz: torch.Tensor | None = None  # float32 [R, L]: the token's sampling rate; 0: padding
    channel_index: torch.Tensor | None = None  # int64 [R, L]: the channel within the row
    position: torch.Tensor | None = None  # int64 [R, L]: the token's order in its channel

    def to(self, device: torch.device) -> Inputs:
        moved = copy.copy(self)
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                setattr(moved, f.name, value.to(device, non_blocking=True))
        return moved


@dataclass
class Analysis:
    """Per token, for reports only: never an input of the model."""

    dataset: torch.Tensor  # int64 [R, L]: index in `Packer.datasets`; -1 is padding
    signal_type: torch.Tensor  # int64 [R, L]: index in SIGNAL_TYPES; -1 is padding

    def to(self, device: torch.device) -> Analysis:
        return Analysis(**{k: v.to(device, non_blocking=True) for k, v in vars(self).items()})


@dataclass
class Batch:
    inputs: Inputs
    analysis: Analysis
    examples: int
    samples: int  # observed samples in the batch


def concat(batches: list[Batch]) -> Batch:
    """One batch of the rows of `batches`, in order (rows of one length: not trimmed)."""

    def joined(k: str) -> torch.Tensor | None:
        parts = [vars(b.inputs)[k] for b in batches]
        return None if parts[0] is None else torch.cat(parts)

    inputs = copy.copy(batches[0].inputs)
    for f in fields(inputs):
        setattr(inputs, f.name, joined(f.name))
    return Batch(
        inputs=inputs,
        analysis=Analysis(**{k: torch.cat([vars(b.analysis)[k] for b in batches])
                             for k in vars(batches[0].analysis)}),
        examples=sum(b.examples for b in batches),
        samples=sum(b.samples for b in batches),
    )  # fmt: skip


@dataclass
class _Tokens:
    """The tokens of one example, before packing."""

    values: np.ndarray  # [n, P]
    observed: np.ndarray
    hidden: np.ndarray
    time_s: np.ndarray
    log_length_s: np.ndarray
    channel_scale: np.ndarray  # [n, 2]
    slot: np.ndarray
    dataset: np.ndarray
    signal_type: np.ndarray


@dataclass
class _FrameTokens:
    """The tokens of one example for the duration / hierarchical tokenizers."""

    values: np.ndarray  # [n, F, P] (P: this example's largest frame)
    observed: np.ndarray
    hidden: np.ndarray
    time_s: np.ndarray
    log_length_s: np.ndarray
    channel_scale: np.ndarray  # [n, 2]
    slot: np.ndarray
    dataset: np.ndarray
    signal_type: np.ndarray
    rate_hz: np.ndarray
    channel: np.ndarray  # the channel within the example
    position: np.ndarray


_Kind = TypeVar("_Kind", _Tokens, _FrameTokens)


def _rows_of(rows: list[list[_Tokens | _FrameTokens]], kind: type[_Kind]) -> list[list[_Kind]]:
    """The rows, each token of the type `kind` (one tokenizer makes every token of a batch)."""
    out: list[list[_Kind]] = []
    for row in rows:
        kept = [t for t in row if isinstance(t, kind)]
        if len(kept) != len(row):
            raise TypeError(f"a batch of {kind.__name__} holds tokens of another tokenizer")
        out.append(kept)
    return out


# A trimmed batch's tokens per row are a multiple of this: the block size of flex attention,
# so that its block mask needs no padding of its own, and few distinct shapes are compiled.
TRIM_TOKENS = 128


class Packer:
    """Turns a stream of examples into batches of `rows` x `max_tokens` tokens. Every
    argument is the experiment's choice; none has a default."""

    def __init__(
        self,
        *,
        datasets: list[str],
        max_tokens: int,
        samples_per_token: int,
        max_channels: int,
        rows: int,
        mask_ratio: float,
        seed: int,
        slots: dict[str, int] | None,
        tokenizer: SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer,
    ) -> None:
        """`slots`: None, a random slot per channel and example (pretraining: a slot only says
        "same channel"); or the slot of each channel name, the same in every example, so a
        task head can learn which channel is which (e.g. the 12 ECG leads). `tokenizer`: how
        channels become tokens (`samples_per_token` is used by `samples` only)."""
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio is in [0, 1)")
        if slots is not None and not all(0 <= k < max_channels for k in slots.values()):
            raise ValueError(f"every slot is in [0, {max_channels})")
        self.slots = slots
        self.tokenizer = tokenizer
        self.datasets = datasets
        self._dataset_index = {d: i for i, d in enumerate(datasets)}
        self.max_tokens = max_tokens
        self.samples_per_token = samples_per_token
        self.max_channels = max_channels
        self.rows = rows
        self.mask_ratio = mask_ratio
        self._rng = random.Random(seed)  # noqa: S311 - not secret

    def reseed(self, seed: int) -> None:
        """Start the random choices (slots, masked tokens) again from `seed`: a batch made
        in a data loader worker is then the same whichever worker makes it."""
        self._rng = random.Random(seed)  # noqa: S311 - not secret

    def batches(self, examples: Iterable[Example]) -> Iterator[Batch]:
        rows: list[list[_Tokens | _FrameTokens]] = [[]]
        used = 0
        count = 0
        for example in examples:
            tokens = self.tokens(example)
            n = len(tokens.slot)
            if n > self.max_tokens:
                raise ValueError(f"an example of {n} tokens is longer than max_tokens")
            if used + n > self.max_tokens:  # the row is full: start the next one
                if len(rows) == self.rows:
                    yield self._batch(rows, count)
                    rows, count = [], 0
                rows.append([])
                used = 0
            rows[-1].append(tokens)
            used += n
            count += 1
        if count:
            yield self._batch(rows, count)

    def pack(self, examples: list[Example], *, trim: bool) -> tuple[Batch, list[tuple[int, int]]]:
        """All `examples` in one batch, in order, and the (row, document) of each: for
        reading the model's output of each example back (benchmarks). Raises if they do
        not fit `rows` x `max_tokens`. `trim`: the batch has only the rows it uses, and
        only as many tokens per row as the longest row holds, rounded up to a multiple of
        TRIM_TOKENS; else every batch is `rows` x `max_tokens`. Padding is masked either
        way, so the model's output for each example is the same."""
        rows: list[list[_Tokens | _FrameTokens]] = [[]]
        where: list[tuple[int, int]] = []
        used = 0
        for example in examples:
            tokens = self.tokens(example)
            n = len(tokens.slot)
            if n > self.max_tokens:
                raise ValueError(f"an example of {n} tokens is longer than max_tokens")
            if used + n > self.max_tokens:
                rows.append([])
                used = 0
            if len(rows) > self.rows:
                raise ValueError(f"{len(examples)} examples do not fit {self.rows} rows")
            where.append((len(rows) - 1, len(rows[-1])))
            rows[-1].append(tokens)
            used += n
        if not trim:
            return self._batch(rows, len(examples)), where
        longest = max(sum(len(tok.slot) for tok in row) for row in rows)
        length = min(self.max_tokens, -(-longest // TRIM_TOKENS) * TRIM_TOKENS)
        return self._batch(rows, len(examples), len(rows), length), where

    def tokens(self, example: Example) -> _Tokens | _FrameTokens:
        if len(example.signals) > self.max_channels:
            raise ValueError(
                f"{len(example.signals)} channels; max_channels is {self.max_channels}"
            )
        if self.slots is None:
            slots = self._rng.sample(range(self.max_channels), len(example.signals))
        else:
            slots = [self.slots[channel.name] for channel in example.origin.channels]
        if not isinstance(self.tokenizer, SamplesTokenizer):
            return self._frame_tokens(example, slots, self.tokenizer)
        p = self.samples_per_token
        parts: list[_Tokens] = []
        for signal, channel, slot in zip(
            example.signals, example.origin.channels, slots, strict=True
        ):
            n = math.ceil(signal.values.size / p)
            padded = np.full(n * p, np.nan, dtype=np.float32)
            padded[: signal.values.size] = signal.values
            values = padded.reshape(n, p)
            observed = np.isfinite(values)  # NaN: invalid or past the end; inf never counts
            hidden = np.zeros(n, dtype=bool)
            if n >= 2 and self.mask_ratio > 0:
                k = min(n - 1, max(1, round(n * self.mask_ratio)))
                hidden[self._rng.sample(range(n), k)] = True
            visible = values[~hidden][observed[~hidden]]
            mean = float(visible.mean()) if visible.size else 0.0
            std = float(visible.std()) if visible.size else 0.0
            scale = std if std > 0 else 1.0
            length_s = p / signal.sampling_rate
            starts = signal.start_s + np.arange(n) * length_s
            parts.append(
                _Tokens(
                    values=np.where(observed, np.arcsinh((values - mean) / scale), 0.0).astype(
                        np.float32
                    ),
                    observed=observed,
                    hidden=hidden,
                    time_s=(starts + length_s / 2).astype(np.float64),
                    log_length_s=np.full(n, math.log(length_s), dtype=np.float32),
                    channel_scale=np.tile(
                        np.array([math.asinh(mean), math.log(scale)], dtype=np.float32), (n, 1)
                    ),
                    slot=np.full(n, slot, dtype=np.int64),
                    dataset=np.full(
                        n, self._dataset_index[example.origin.dataset_id], dtype=np.int64
                    ),
                    signal_type=np.full(n, SIGNAL_TYPES.index(channel.signal_type), dtype=np.int64),
                )
            )
        return _Tokens(
            **{
                name: np.concatenate([getattr(part, name) for part in parts])
                for name in _Tokens.__dataclass_fields__
            }
        )

    def _frame_tokens(
        self, example: Example, slots: list[int], spec: DurationTokenizer | HierarchicalTokenizer
    ) -> _FrameTokens:
        """Frames of a fixed duration (docs/v3-tokenizer.md): frame k of a channel covers
        samples [k h - (n - h) // 2, + n), centred on its core [k h, (k + 1) h); F frames make
        one token; samples outside the channel are not observed."""
        parts: list[_FrameTokens] = []
        largest = max(frame_shape(spec, s.sampling_rate)[0] for s in example.signals)
        for c, (signal, channel, slot) in enumerate(
            zip(example.signals, example.origin.channels, slots, strict=True)
        ):
            rate = signal.sampling_rate
            n, h, f = frame_shape(spec, rate)
            size = signal.values.size
            count = max(1, math.ceil(size / h))  # frames with a core in the channel
            tokens = math.ceil(count / f)
            starts = np.arange(tokens * f) * h - (n - h) // 2
            index = starts[:, None] + np.arange(n)[None, :]
            inside = (index >= 0) & (index < size) & (np.arange(tokens * f)[:, None] < count)
            raw = np.where(inside, signal.values[np.clip(index, 0, max(0, size - 1))], np.nan)
            observed = np.isfinite(raw)  # NaN: invalid, outside the channel or a padding frame
            hidden = np.zeros(tokens, dtype=bool)
            if tokens >= 2 and self.mask_ratio > 0:
                k = min(tokens - 1, max(1, round(tokens * self.mask_ratio)))
                hidden[self._rng.sample(range(tokens), k)] = True
            # the scale from the samples whose token (by the frames' cores) is not hidden
            core = np.repeat(hidden, f * h)[:size]
            visible = signal.values[~np.pad(core, (0, size - core.size))]
            visible = visible[np.isfinite(visible)]
            mean = float(visible.mean()) if visible.size else 0.0
            std = float(visible.std()) if visible.size else 0.0
            scale = std if std > 0 else 1.0
            values = np.zeros((tokens, f, largest), dtype=np.float32)
            values[:, :, :n] = np.where(observed, np.arcsinh((raw - mean) / scale), 0.0).reshape(
                tokens, f, n
            )
            seen = np.zeros((tokens, f, largest), dtype=bool)
            seen[:, :, :n] = observed.reshape(tokens, f, n)
            centre = signal.start_s + (np.arange(tokens) * f + f / 2) * h / rate
            parts.append(
                _FrameTokens(
                    values=values, observed=seen, hidden=hidden,
                    time_s=centre.astype(np.float64),
                    log_length_s=np.full(tokens, math.log(f * h / rate), dtype=np.float32),
                    channel_scale=np.tile(
                        np.array([math.asinh(mean), math.log(scale)], dtype=np.float32),
                        (tokens, 1),
                    ),
                    slot=np.full(tokens, slot, dtype=np.int64),
                    dataset=np.full(
                        tokens, self._dataset_index[example.origin.dataset_id], dtype=np.int64
                    ),
                    signal_type=np.full(
                        tokens, SIGNAL_TYPES.index(channel.signal_type), dtype=np.int64
                    ),
                    rate_hz=np.full(tokens, rate, dtype=np.float32),
                    channel=np.full(tokens, c, dtype=np.int64),
                    position=np.arange(tokens, dtype=np.int64),
                )
            )  # fmt: skip
        return _FrameTokens(
            **{name: np.concatenate([getattr(part, name) for part in parts])
               for name in _FrameTokens.__dataclass_fields__}
        )  # fmt: skip

    def _batch(
        self, rows: list[list[_Tokens | _FrameTokens]], count: int,
        r: int | None = None, length: int | None = None,
    ) -> Batch:  # fmt: skip
        r = self.rows if r is None else r
        length = self.max_tokens if length is None else length
        if not isinstance(self.tokenizer, SamplesTokenizer):
            return self._frame_batch(_rows_of(rows, _FrameTokens), count, r, length)
        sample_rows = _rows_of(rows, _Tokens)
        p = self.samples_per_token
        values = np.zeros((r, length, p), dtype=np.float32)
        observed = np.zeros((r, length, p), dtype=bool)
        hidden = np.zeros((r, length), dtype=bool)
        time_s = np.zeros((r, length), dtype=np.float64)
        log_length_s = np.zeros((r, length), dtype=np.float32)
        channel_scale = np.zeros((r, length, 2), dtype=np.float32)
        slot = np.zeros((r, length), dtype=np.int64)
        document = np.full((r, length), -1, dtype=np.int64)
        dataset = np.full((r, length), -1, dtype=np.int64)
        signal_type = np.full((r, length), -1, dtype=np.int64)
        for i, row in enumerate(sample_rows):
            at = 0
            for d, t in enumerate(row):
                n = len(t.slot)
                s = slice(at, at + n)
                values[i, s], observed[i, s], hidden[i, s] = t.values, t.observed, t.hidden
                time_s[i, s], log_length_s[i, s] = t.time_s, t.log_length_s
                channel_scale[i, s], slot[i, s], document[i, s] = t.channel_scale, t.slot, d
                dataset[i, s], signal_type[i, s] = t.dataset, t.signal_type
                at += n
        inputs = Inputs(
            *(torch.from_numpy(a) for a in (values, observed, hidden, time_s, log_length_s,
                                            channel_scale, slot, document))
        )  # fmt: skip
        analysis = Analysis(torch.from_numpy(dataset), torch.from_numpy(signal_type))
        return Batch(inputs, analysis, count, int(observed.sum()))

    def _frame_batch(
        self, rows: list[list[_FrameTokens]], count: int, r: int, length: int
    ) -> Batch:
        tokens = [t for row in rows for t in row]
        f = tokens[0].values.shape[1] if tokens else 1
        p = max((t.values.shape[2] for t in tokens), default=1)
        values = np.zeros((r, length, f, p), dtype=np.float32)
        observed = np.zeros((r, length, f, p), dtype=bool)
        hidden = np.zeros((r, length), dtype=bool)
        time_s = np.zeros((r, length), dtype=np.float64)
        log_length_s = np.zeros((r, length), dtype=np.float32)
        channel_scale = np.zeros((r, length, 2), dtype=np.float32)
        slot = np.zeros((r, length), dtype=np.int64)
        document = np.full((r, length), -1, dtype=np.int64)
        dataset = np.full((r, length), -1, dtype=np.int64)
        signal_type = np.full((r, length), -1, dtype=np.int64)
        rate_hz = np.zeros((r, length), dtype=np.float32)
        channel_index = np.full((r, length), -1, dtype=np.int64)
        position = np.zeros((r, length), dtype=np.int64)
        for i, row in enumerate(rows):
            at, channels = 0, 0
            for d, t in enumerate(row):
                n, q = len(t.slot), t.values.shape[2]
                s = slice(at, at + n)
                values[i, s, :, :q], observed[i, s, :, :q] = t.values, t.observed
                hidden[i, s], time_s[i, s] = t.hidden, t.time_s
                log_length_s[i, s], channel_scale[i, s] = t.log_length_s, t.channel_scale
                slot[i, s], document[i, s] = t.slot, d
                dataset[i, s], signal_type[i, s] = t.dataset, t.signal_type
                rate_hz[i, s], channel_index[i, s] = t.rate_hz, t.channel + channels
                position[i, s] = t.position
                channels += int(t.channel.max()) + 1
                at += n
        inputs = Inputs(
            *(torch.from_numpy(a) for a in (values, observed, hidden, time_s, log_length_s,
                                            channel_scale, slot, document, rate_hz,
                                            channel_index, position))
        )  # fmt: skip
        analysis = Analysis(torch.from_numpy(dataset), torch.from_numpy(signal_type))
        return Batch(inputs, analysis, count, int(observed.sum()))

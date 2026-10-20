"""From examples to batches: tokens, masks and packing.

A token is `samples_per_token` consecutive samples of one channel (the last token of a
channel holds what is left, the rest is marked as not observed). For each token the model
gets, and nothing else:

- its samples, divided by the scale of its channel in this example (below), with a flag
  per sample: observed or not (invalid in the source, or past the end of the channel);
- the time of its centre and its length, in seconds from the start of the example;
- a channel slot: a number drawn at random per example, which only says "same channel";
- the scale of its channel: the mean and standard deviation of the channel's visible
  samples, so the division loses nothing.

Why the division: values are physical, in the units of each source, so they span many
orders of magnitude (EEG in V is 1e-5, CTG in bpm is 1e2). Dividing by a channel's own
scale makes them comparable for the network, and giving the scale to the network as an
input keeps every value recoverable. It is the network's input encoding, not a change of
the data: the loader and the stored data stay as they are. The statistics come from the
visible tokens only, so masked samples do not leak into them.

Masking (for masked-token pretraining): in each channel with two or more tokens,
`mask_ratio` of the tokens are hidden at random, at least one token stays visible.

Packing: examples are put whole, in the order they come, into rows of `max_tokens`
tokens, as in LLM training; `document` says which example each token belongs to, and the
model attends only within one example. Padding at the end of a row has document -1.

`Analysis` holds what the model must not see (dataset, signal type), to split the loss
for reports.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import torch

from biosignal_data.index.signals import SignalType
from biosignal_data.training.loader import Example

SIGNAL_TYPES = [t.value for t in SignalType]


@dataclass
class Inputs:
    """The model's input, [rows, max_tokens, ...]; nothing else goes into the model."""

    values: torch.Tensor  # float32 [R, L, P]: samples / channel scale; 0 where not observed
    observed: torch.Tensor  # bool [R, L, P]
    hidden: torch.Tensor  # bool [R, L]: masked token, the model does not see its samples
    time_s: torch.Tensor  # float32 [R, L]: centre of the token, from the example's start
    log_length_s: torch.Tensor  # float32 [R, L]: log of the token's length in seconds
    channel_scale: torch.Tensor  # float32 [R, L, 2]: asinh(mean), log(std) of the channel
    slot: torch.Tensor  # int64 [R, L]
    document: torch.Tensor  # int64 [R, L]: example within the row; -1 is padding

    def to(self, device: torch.device) -> Inputs:
        return Inputs(**{k: v.to(device, non_blocking=True) for k, v in vars(self).items()})


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
    ) -> None:
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio is in [0, 1)")
        self.datasets = datasets
        self._dataset_index = {d: i for i, d in enumerate(datasets)}
        self.max_tokens = max_tokens
        self.samples_per_token = samples_per_token
        self.max_channels = max_channels
        self.rows = rows
        self.mask_ratio = mask_ratio
        self._rng = random.Random(seed)  # noqa: S311 - not secret

    def batches(self, examples: Iterable[Example]) -> Iterator[Batch]:
        rows: list[list[_Tokens]] = [[]]
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

    def tokens(self, example: Example) -> _Tokens:
        p = self.samples_per_token
        if len(example.signals) > self.max_channels:
            raise ValueError(
                f"{len(example.signals)} channels; max_channels is {self.max_channels}"
            )
        slots = self._rng.sample(range(self.max_channels), len(example.signals))
        parts: list[_Tokens] = []
        for signal, channel, slot in zip(
            example.signals, example.origin.channels, slots, strict=True
        ):
            n = math.ceil(signal.values.size / p)
            padded = np.full(n * p, np.nan, dtype=np.float32)
            padded[: signal.values.size] = signal.values
            values = padded.reshape(n, p)
            observed = ~np.isnan(values)
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
                    values=np.where(observed, (values - mean) / scale, 0.0).astype(np.float32),
                    observed=observed,
                    hidden=hidden,
                    time_s=(starts + length_s / 2).astype(np.float32),
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

    def _batch(self, rows: list[list[_Tokens]], count: int) -> Batch:
        r, length, p = self.rows, self.max_tokens, self.samples_per_token
        values = np.zeros((r, length, p), dtype=np.float32)
        observed = np.zeros((r, length, p), dtype=bool)
        hidden = np.zeros((r, length), dtype=bool)
        time_s = np.zeros((r, length), dtype=np.float32)
        log_length_s = np.zeros((r, length), dtype=np.float32)
        channel_scale = np.zeros((r, length, 2), dtype=np.float32)
        slot = np.zeros((r, length), dtype=np.int64)
        document = np.full((r, length), -1, dtype=np.int64)
        dataset = np.full((r, length), -1, dtype=np.int64)
        signal_type = np.full((r, length), -1, dtype=np.int64)
        for i, row in enumerate(rows):
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

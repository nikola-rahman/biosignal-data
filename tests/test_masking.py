"""The v4 masks (biosignal_model.channel_batch.hide and the packer): what is hidden, and that
nothing hidden reaches the model. docs/architecture.md sections 7 and 10."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.channel_batch import (  # noqa: E402
    ChannelPacker,
    MaskSpec,
    apply_sample_mask,
    hide,
    token_cells,
)

TOKEN_S = 0.04
SPANS = MaskSpec(
    ratio=0.6, time_spans=1.0, channel_groups=0.0, single_tokens=0.0,
    time_span_s=(0.2, 2.0), group_span_s=(0.2, 2.0),
)  # fmt: skip
RATES = [100.0, 250.0, 500.0, 173.61, 1.0]  # mixed rates, one slower than one sample a cell


def _example(values: list[np.ndarray], rates: list[float], starts: list[float]) -> Example:
    signals = tuple(Signal(v, r, s) for v, r, s in zip(values, rates, starts, strict=True))
    channels = tuple(
        Channel("a", i, r, 1.0, 0.0, None, f"c{i}", "ecg", "mV") for i, r in enumerate(rates)
    )
    return Example(signals, Origin("d/x", "r", "s", 0.0, channels))


def _signals(seconds: float, seed: int, rates: list[float] = RATES) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=max(1, round(seconds * r))).astype(np.float32) for r in rates]


def _packer(mask: MaskSpec | None, seed: int = 0) -> ChannelPacker:
    return ChannelPacker(
        datasets=["d/x"], token_s=TOKEN_S, max_tokens=100_000, max_samples=10_000_000,
        max_channels=8, rows=1, mask=mask, seed=seed,
    )  # fmt: skip


def _cells_of(seconds: float, rates: list[float], starts: list[float]) -> list[np.ndarray]:
    return [
        np.unique(token_cells(s, r, max(1, round(seconds * r)), TOKEN_S))
        for r, s in zip(rates, starts, strict=True)
    ]


def _goal(total: int, ratio: float) -> int:
    return min(total - 1, round(ratio * total))


# --- what is hidden -------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_a_hidden_time_is_hidden_on_every_channel_at_mixed_rates(seed: int) -> None:
    """Time spans hide whole grid cells: at every cell, all of the channels that have a token
    there are hidden, or none is (also the cell where the share is reached)."""
    starts = [0.0, 0.013, 0.0, 0.031, 0.4]
    cells = _cells_of(12.0, RATES, starts)
    hidden = hide(cells, TOKEN_S, SPANS, np.random.default_rng(seed), groups=None)
    state: dict[int, set[bool]] = {}
    for c, h in zip(cells, hidden, strict=True):
        for cell, flag in zip(c.tolist(), h.tolist(), strict=True):
            state.setdefault(cell, set()).add(flag)
    assert all(len(flags) == 1 for flags in state.values())


@pytest.mark.parametrize("tokens", [2, 3, 7, 50, 1000])
def test_the_ratio_is_reached_within_one_cell_and_one_token_stays_visible(tokens: int) -> None:
    channels = 4
    cells = [np.arange(math.ceil(tokens / channels)) for _ in range(channels)]
    total = sum(c.size for c in cells)
    for seed in range(20):
        for ratio in (0.1, 0.5, 0.9):
            spec = SPANS.model_copy(update={"ratio": ratio})
            hidden = np.concatenate(hide(cells, TOKEN_S, spec, np.random.default_rng(seed), None))
            goal = _goal(total, ratio)
            assert total - hidden.sum() >= 1  # something stays visible
            # whole cells of `channels` tokens: the goal, passed by less than one cell, or all
            # cells but the last one when the goal is in it
            assert goal <= hidden.sum() <= goal + channels - 1 or hidden.sum() == total - channels


def test_span_lengths_follow_the_bounds() -> None:
    """With a low ratio, spans rarely touch: the hidden runs have the lengths of the spans,
    log-uniform between the bounds (to one cell). Only the span that reaches the share is
    cut, so an example has at most one run shorter than the low bound (runs at the ends of
    the example are left out: a span may start before or end after it)."""
    spec = SPANS.model_copy(update={"ratio": 0.05})
    low, high = spec.time_span_s
    lengths = []
    for seed in range(300):
        cells = [np.arange(1500)]  # 60 s
        hidden = hide(cells, TOKEN_S, spec, np.random.default_rng(seed), None)[0].astype(int)
        edges = np.diff(np.concatenate([[0], hidden, [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        inner = [(b - a) * TOKEN_S for a, b in zip(starts, ends, strict=True)
                 if a > 0 and b < hidden.size]  # fmt: skip
        assert sum(length < low - TOKEN_S for length in inner) <= 1  # the cut span only
        lengths += inner
    lengths_s = np.array(lengths)
    assert lengths_s.size > 300
    assert (lengths_s <= high + TOKEN_S).mean() > 0.95  # two touching spans make one run
    # log-uniform: the median of the lengths is near the geometric mean of the bounds
    assert 0.7 < np.median(lengths_s) / math.sqrt(low * high) < 1.4


def test_every_time_is_hidden_alike_also_at_the_ends() -> None:
    """A span starts uniformly in [-L, end): the first and last cells are hidden as often as
    the middle (no edge bias the model could learn)."""
    cells = [np.arange(250)]  # 10 s
    counts = np.zeros(250)
    for seed in range(2000):
        counts += hide(cells, TOKEN_S, SPANS, np.random.default_rng(seed), None)[0]
    share = counts / 2000
    middle = share[100:150].mean()
    assert 0.5 < share[:5].mean() / middle < 1.5 and 0.5 < share[-5:].mean() / middle < 1.5


def test_the_mask_does_not_depend_on_the_values() -> None:
    """The same shapes and seed give the same mask whatever the samples are: the mask cannot
    carry anything about the content."""
    starts = [0.0] * len(RATES)
    a = _packer(SPANS).pack([_example(_signals(8.0, 1), RATES, starts)])[0]
    b = _packer(SPANS).pack([_example(_signals(8.0, 2), RATES, starts)])[0]
    assert torch.equal(a.inputs.tokens.hidden, b.inputs.tokens.hidden)
    assert torch.equal(a.inputs.hidden, b.inputs.hidden)


def test_the_same_seed_gives_the_same_masks_and_reseed_starts_again() -> None:
    starts = [0.0] * len(RATES)
    examples = [_example(_signals(5.0, s), RATES, starts) for s in range(3)]
    first = _packer(SPANS, seed=7)
    masks = [first.pack([e])[0].inputs.tokens.hidden for e in examples]
    first.reseed(7)
    again = [first.pack([e])[0].inputs.tokens.hidden for e in examples]
    assert all(torch.equal(m, n) for m, n in zip(masks, again, strict=True))
    assert not torch.equal(masks[0], _packer(SPANS, seed=8).pack([examples[0]])[0]
                           .inputs.tokens.hidden)  # fmt: skip


# --- nothing hidden reaches the model -------------------------------------------------------


def _hidden_sample_flags(seconds: float, starts: list[float]) -> list[np.ndarray]:
    """Per channel, whether each sample is in a hidden token (the packer's mask, seed 0)."""
    batch = _packer(SPANS).pack([_example(_signals(seconds, 0), RATES, starts)])[0]
    x = batch.inputs
    return [
        x.hidden[int(o) : int(o) + int(n)].numpy()
        for o, n in zip(x.channels.offset, x.channels.length, strict=True)
    ]


def test_changing_hidden_samples_changes_nothing_the_model_reads() -> None:
    """Two examples that differ only in their hidden samples (any values, also NaN and huge)
    give bit-identical model inputs: no model part can depend on a hidden sample."""
    starts = [0.0, 0.013, 0.0, 0.031, 0.4]
    original = _signals(10.0, 3)
    flags = _hidden_sample_flags(10.0, starts)
    rng = np.random.default_rng(9)
    changed = []
    for v, h in zip(original, flags, strict=True):
        w = v.copy()
        w[h] = rng.choice(np.array([np.nan, 1e6, -1e6, 0.0], dtype=np.float32), int(h.sum()))
        changed.append(w)
    assert sum(int(h.sum()) for h in flags) > 0
    a = _packer(SPANS).pack([_example(original, RATES, starts)])[0].inputs
    b = _packer(SPANS).pack([_example(changed, RATES, starts)])[0].inputs
    assert torch.equal(a.values, b.values) and torch.equal(a.observed, b.observed)
    assert torch.equal(a.hidden, b.hidden) and torch.equal(a.tokens.hidden, b.tokens.hidden)
    for name in ("level", "log_scale", "slot", "signal_type", "rate_hz", "start_s", "length"):
        assert torch.equal(getattr(a.channels, name), getattr(b.channels, name)), name


def test_a_sample_in_a_hidden_time_is_hidden_on_every_channel() -> None:
    """By time, not by token: a sample whose time is in a hidden cell of any channel is hidden,
    on every channel and rate; the hidden samples are 0 and not observed."""
    starts = [0.0, 0.013, 0.0, 0.031, 0.4]
    batch = _packer(SPANS).pack([_example(_signals(10.0, 4), RATES, starts)])[0]
    x = batch.inputs
    hidden_cells: set[int] = set()
    per_channel = []
    for k, (o, n) in enumerate(zip(x.channels.offset, x.channels.length, strict=True)):
        cells = token_cells(starts[k], RATES[k], int(n), TOKEN_S)
        flags = x.hidden[int(o) : int(o) + int(n)].numpy()
        hidden_cells |= set(cells[flags].tolist())
        per_channel.append((cells, flags))
    for cells, flags in per_channel:
        assert np.array_equal(flags, np.isin(cells, list(hidden_cells)))
    assert (x.values[x.hidden] == 0).all() and not x.observed[x.hidden].any()


def test_no_gradient_reaches_a_hidden_sample() -> None:
    values = torch.randn(50, dtype=torch.float32, requires_grad=True)
    valid = torch.ones(50, dtype=torch.bool)
    hidden = torch.zeros(50, dtype=torch.bool)
    hidden[10:30] = True
    out, _ = apply_sample_mask(values, valid, hidden)
    (out * torch.randn(50)).sum().backward()
    assert values.grad is not None
    assert (values.grad[hidden] == 0).all() and (values.grad[~hidden] != 0).all()


# --- the masks are not solvable by a trivial rule -------------------------------------------


def _ecg_like(seconds: float, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Narrow QRS-like pulses (80-120 ms) at a varying RR (0.6-1.0 s) on a small baseline."""
    t = np.arange(round(seconds * rate)) / rate
    beats = np.cumsum(rng.uniform(0.6, 1.0, size=int(seconds) + 2))
    x = 0.05 * np.sin(2 * np.pi * 0.3 * t)
    for b in beats:
        x += np.exp(-0.5 * ((t - b) / rng.uniform(0.016, 0.024)) ** 2)
    return x.astype(np.float32)


def test_linear_interpolation_across_a_hidden_span_fails_on_qrs_like_signals() -> None:
    """The trivial solver of a time span is interpolation from the visible samples at its two
    ends (no other channel is visible there). With the spans' bounds it must miss the beats:
    its error on the hidden samples is most of their variance."""
    rng = np.random.default_rng(0)
    errors, variances = [], []
    for seed in range(20):
        x = _ecg_like(20.0, 100.0, rng)
        cells = token_cells(0.0, 100.0, x.size, TOKEN_S)
        unique, inverse = np.unique(cells, return_inverse=True)
        hidden = hide([unique], TOKEN_S, SPANS, np.random.default_rng(seed), None)[0][inverse]
        index = np.arange(x.size)
        guess = np.interp(index[hidden], index[~hidden], x[~hidden])
        errors.append(np.mean((guess - x[hidden]) ** 2))
        variances.append(np.var(x[hidden]))
    assert np.mean(errors) / np.mean(variances) > 0.8


# --- the value transform --------------------------------------------------------------------


def test_the_level_and_scale_come_from_the_visible_samples_only() -> None:
    """m and s of a channel are those of its visible valid samples: x' = asinh((x - m) / s);
    the level asinh(m) and log(s) are the model's inputs; a constant channel gets s = 1."""
    rates = [100.0, 250.0]
    values = [np.full(1000, 3.0, dtype=np.float32), _signals(4.0, 5, [250.0])[0] * 7 + 2]
    batch = _packer(SPANS).pack([_example(values, rates, [0.0, 0.0])])[0]
    x, c = batch.inputs, batch.inputs.channels
    k = 1
    o, n = int(c.offset[k]), int(c.length[k])
    visible = ~x.hidden[o : o + n].numpy()
    seen = values[k][visible].astype(np.float64)
    assert float(c.level[k]) == pytest.approx(np.arcsinh(seen.mean()), rel=1e-5)
    assert float(c.log_scale[k]) == pytest.approx(np.log(seen.std()), rel=1e-5)
    expected = np.arcsinh((values[k] - seen.mean()) / seen.std()).astype(np.float32)
    np.testing.assert_allclose(batch.targets.normalized[o : o + n].numpy(), expected, atol=1e-5)
    assert float(c.log_scale[0]) == 0.0 and float(c.level[0]) == pytest.approx(np.arcsinh(3.0))
    assert (batch.targets.normalized[: int(c.length[0])] == 0).all()


def test_slots_are_distinct_within_an_example() -> None:
    examples = [_example(_signals(2.0, s), RATES, [0.0] * len(RATES)) for s in range(20)]
    for e in examples:
        slots = _packer(SPANS, seed=3).pack([e])[0].inputs.channels.slot.tolist()
        assert len(set(slots)) == len(RATES) and all(0 <= k < 8 for k in slots)

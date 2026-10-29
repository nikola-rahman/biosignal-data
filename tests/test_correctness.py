"""Correctness tests before pretraining (architecture-v2.md, section 10), for every tokenizer:
the v0-v2 sample tokens, D1 (duration) and D2 (hierarchical, both residual pools).

- Hidden samples reach no output: no gradient from the outputs of hidden tokens to the raw
  samples in their cores, and a change of those samples changes no output.
- Packing invariance: an example alone gives the same features as packed with others.
- Channel permutation invariance without channel identity (every channel in the same slot).
- Rotary precision at 1800 s: float64 times and angles; a shift of the whole example in time
  changes no feature.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.batch import (  # noqa: E402
    DurationTokenizer,
    HierarchicalTokenizer,
    Inputs,
    Packer,
    SamplesTokenizer,
    frame_shape,
)
from biosignal_model.model import Encoder, ModelConfig  # noqa: E402

AnyTokenizer = SamplesTokenizer | DurationTokenizer | HierarchicalTokenizer

SAMPLES = SamplesTokenizer(kind="samples")
D1 = DurationTokenizer(
    kind="duration", token_s=0.08, hop_s=0.04, stem_windows=4, stem_channels=8,
    kernel_frequencies=6, kernel_min_hz=1.0, kernel_max_hz=200.0,
)  # fmt: skip
D2 = HierarchicalTokenizer(
    kind="hierarchical", frame_s=0.04, frame_hop_s=0.02, frames_per_token=12, stem_windows=2,
    stem_channels=8, kernel_frequencies=6, kernel_min_hz=1.0, kernel_max_hz=200.0,
    local_width=16, local_layers=2, local_states=8, pool_heads=4, residual_pool="mean",
)  # fmt: skip
D2_ATTENTION = D2.model_copy(update={"residual_pool": "attention"})
ALL = [SAMPLES, D1, D2, D2_ATTENTION]
IDS = ["samples", "duration", "hierarchical", "hierarchical-attention"]

# Frames of D1 and D2 are longer than their hop: a frame reads (n - h) / 2 samples of each
# neighbour's core, so a visible neighbour carries samples of a hidden token (and the D2
# local stage then spreads them). Reported, not redesigned here (architecture-v2.md section 3:
# hidden samples are zeroed before the local stage in v4).
FRAME_LEAK = pytest.mark.xfail(
    reason="D1/D2 frames overlap the neighbours' cores (frame longer than hop): visible "
    "neighbour frames hold samples of a hidden token's core",
    strict=True,
)
LEAK_CASES = [
    SAMPLES,
    pytest.param(D1, marks=FRAME_LEAK),
    pytest.param(D2, marks=FRAME_LEAK),
    pytest.param(D2_ATTENTION, marks=FRAME_LEAK),
]


def _config(tokenizer: AnyTokenizer) -> ModelConfig:
    return ModelConfig(
        samples_per_token=8, max_channels=4, width=32, layers=2, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense", tokenizer=tokenizer,
        dropout=0.0, drop_path=0.0,
    )  # fmt: skip


def _example(rates: list[float], seconds: float, seed: int, name: str = "r") -> Example:
    rng = np.random.default_rng(seed)
    signals, channels = [], []
    for i, rate in enumerate(rates):
        values = rng.normal(size=round(seconds * rate)).astype(np.float32)
        signals.append(Signal(values, rate, 0.0))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", f"{name}{seed}", "s", 0.0, tuple(channels)))


def _packer(
    tokenizer: AnyTokenizer, rows: int, mask_ratio: float = 0.0,
    slots: dict[str, int] | None = None,
) -> Packer:  # fmt: skip
    return Packer(
        datasets=["d/x"], max_tokens=1024, samples_per_token=8, max_channels=4, rows=rows,
        mask_ratio=mask_ratio, seed=0,
        slots={f"c{i}": i for i in range(4)} if slots is None else slots, tokenizer=tokenizer,
    )  # fmt: skip


def _outputs(model: Encoder, x: Inputs) -> torch.Tensor:
    """What a hidden token's prediction is read from: the pretraining output where the
    tokenizer has one (samples, duration), else the features (hierarchical has no
    pretraining head yet)."""
    if isinstance(model.config.tokenizer, HierarchicalTokenizer):
        return model.features(x)
    out: torch.Tensor = model(x)
    return out


def _raw_positions(x: Inputs, tokenizer: AnyTokenizer, example: Example) -> dict[str, np.ndarray]:
    """For row 0, document 0 (one example): per entry of `x.values` the channel (index in the
    example, -1: none) and the raw sample index it holds, and per token its channel and the
    raw range of its core [first, stop)."""
    real = (x.document[0] == 0).numpy()
    tokens = int(real.sum())
    values_shape = tuple(x.values.shape[2:])
    channel_of = np.full((x.values.shape[1], *values_shape), -1)
    sample_of = np.full((x.values.shape[1], *values_shape), -1)
    token_channel = np.full(x.values.shape[1], -1)
    core = np.zeros((x.values.shape[1], 2), dtype=np.int64)
    if isinstance(tokenizer, SamplesTokenizer):
        slot_to_channel = {i: i for i in range(len(example.signals))}  # fixed slots c<i> -> i
        seen: dict[int, int] = defaultdict(int)
        p = x.values.shape[2]
        for t in range(tokens):
            c = slot_to_channel[int(x.slot[0, t])]
            pos = seen[c]
            seen[c] += 1
            token_channel[t], core[t] = c, (pos * p, (pos + 1) * p)
            channel_of[t] = c
            sample_of[t] = pos * p + np.arange(p)
    else:
        assert x.channel_index is not None and x.position is not None
        for t in range(tokens):
            c = int(x.channel_index[0, t])
            n, h, f = frame_shape(tokenizer, example.signals[c].sampling_rate)
            pos = int(x.position[0, t])
            token_channel[t], core[t] = c, (pos * f * h, (pos + 1) * f * h)
            frames = pos * f + np.arange(f)
            channel_of[t, :, :n] = c
            sample_of[t, :, :n] = frames[:, None] * h - (n - h) // 2 + np.arange(n)[None, :]
    return {"channel": channel_of, "sample": sample_of, "token_channel": token_channel,
            "core": core}  # fmt: skip


def _hidden_cores(x: Inputs, where: dict[str, np.ndarray], example: Example) -> list[np.ndarray]:
    """Per channel of the example, which raw samples are in the core of a hidden token."""
    hidden = [np.zeros(s.values.size, dtype=bool) for s in example.signals]
    for t in np.flatnonzero(x.hidden[0].numpy() & (x.document[0] == 0).numpy()):
        c = where["token_channel"][t]
        first, stop = where["core"][t]
        hidden[c][first : min(stop, hidden[c].size)] = True
    return hidden


@pytest.mark.parametrize("tokenizer", LEAK_CASES, ids=IDS)
def test_hidden_samples_get_no_gradient_from_the_hidden_predictions(
    tokenizer: AnyTokenizer,
) -> None:
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    example = _example([100.0, 500.0], 2.5, 0)
    batch, _ = _packer(tokenizer, rows=1, mask_ratio=0.3).pack([example], trim=True)
    x = batch.inputs
    x.values = x.values.clone().requires_grad_(True)
    out = _outputs(model, x)
    hidden = x.hidden & (x.document == 0)
    assert hidden.any() and (~hidden & (x.document == 0)).any()
    weight = torch.randn_like(out[hidden])  # every hidden output, none cancelling another
    (out[hidden] * weight).sum().backward()  # type: ignore[no-untyped-call]
    assert x.values.grad is not None
    grad = x.values.grad[0].numpy()
    where = _raw_positions(x, tokenizer, example)
    cores = _hidden_cores(x, where, example)
    observed = x.observed[0].numpy()
    leaks = []
    for c, core in enumerate(cores):
        entries = (where["channel"] == c) & observed & (where["sample"] >= 0)
        samples = where["sample"][entries]
        inside = (samples < core.size) & core[np.clip(samples, 0, core.size - 1)]
        touched = np.abs(grad[entries][inside]) > 0
        if touched.any():
            leaks.append(
                f"channel {c}: {np.unique(samples[inside][touched]).size} of {int(core.sum())} "
                f"hidden samples have a gradient (max {np.abs(grad[entries][inside]).max():.2e})"
            )
    assert not leaks, "; ".join(leaks)


@pytest.mark.parametrize("tokenizer", LEAK_CASES, ids=IDS)
def test_a_change_of_hidden_samples_changes_no_output(tokenizer: AnyTokenizer) -> None:
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    example = _example([100.0, 500.0], 2.5, 1)
    packer = _packer(tokenizer, rows=1, mask_ratio=0.3)
    packer.reseed(7)
    batch, _ = packer.pack([example], trim=True)
    cores = _hidden_cores(batch.inputs, _raw_positions(batch.inputs, tokenizer, example), example)
    assert any(c.any() for c in cores)
    changed = Example(
        tuple(
            Signal(np.where(core, s.values + 5.0, s.values).astype(np.float32),
                   s.sampling_rate, s.start_s)
            for s, core in zip(example.signals, cores, strict=True)
        ),
        example.origin,
    )  # fmt: skip
    packer.reseed(7)
    other, _ = packer.pack([changed], trim=True)
    assert torch.equal(other.inputs.hidden, batch.inputs.hidden)  # the same tokens hidden
    with torch.no_grad():
        a, b = _outputs(model, batch.inputs), _outputs(model, other.inputs)
    real = batch.inputs.document >= 0
    difference = (a[real] - b[real]).abs().max().item()
    assert difference == 0.0, f"largest change of an output: {difference:.2e}"


@pytest.mark.parametrize("tokenizer", ALL, ids=IDS)
def test_an_example_alone_has_the_features_it_has_packed_with_others(
    tokenizer: AnyTokenizer,
) -> None:
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    alone = _example([100.0, 500.0], 2.5, 2)
    before, after = _example([500.0, 100.0, 250.0], 1.7, 3), _example([250.0], 3.1, 4)
    single, _ = _packer(tokenizer, rows=1).pack([alone], trim=True)
    together, where = _packer(tokenizer, rows=2).pack([before, alone, after], trim=False)
    assert where[1] == (0, 1)  # second in its row, after another example
    with torch.no_grad():
        a = model.features(single.inputs)[0][single.inputs.document[0] == 0]
        r, d = where[1]
        b = model.features(together.inputs)[r][together.inputs.document[r] == d]
    assert a.shape == b.shape and a.shape[0] > 0
    torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


def _token_count(tokenizer: AnyTokenizer, signal: Signal) -> int:
    if isinstance(tokenizer, SamplesTokenizer):
        return math.ceil(signal.values.size / 8)
    _, h, f = frame_shape(tokenizer, signal.sampling_rate)
    return math.ceil(max(1, math.ceil(signal.values.size / h)) / f)


@pytest.mark.parametrize("tokenizer", ALL, ids=IDS)
def test_the_order_of_the_channels_changes_nothing_without_channel_identity(
    tokenizer: AnyTokenizer,
) -> None:
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    example = _example([100.0, 250.0, 500.0], 2.0, 5)
    order = [2, 0, 1]
    o = example.origin
    permuted = Example(
        tuple(example.signals[k] for k in order),
        Origin(o.dataset_id, o.recording_id, o.subject_id, o.start_s,
               tuple(o.channels[k] for k in order)),
    )  # fmt: skip
    same_slot = {f"c{i}": 0 for i in range(3)}  # no channel identity
    features = []
    for e in (example, permuted):
        batch, _ = _packer(tokenizer, rows=1, slots=same_slot).pack([e], trim=True)
        with torch.no_grad():
            h = model.features(batch.inputs)[0][batch.inputs.document[0] == 0]
        counts = [_token_count(tokenizer, s) for s in e.signals]
        assert sum(counts) == h.shape[0]
        parts = torch.split(h, counts)
        features.append({ch.name: p for ch, p in zip(e.origin.channels, parts, strict=True)})
    for name, h in features[0].items():
        torch.testing.assert_close(features[1][name], h, atol=2e-5, rtol=1e-4)


def test_token_times_are_float64_and_exact_after_30_minutes() -> None:
    example = _example([100.0], 1810.0, 6)
    for tokenizer in (SAMPLES, D1, D2):
        t = _packer(tokenizer, rows=1).tokens(example).time_s
        assert t.dtype == np.float64
        if isinstance(tokenizer, SamplesTokenizer):
            expected = (np.arange(t.size) + 0.5) * 8 / 100.0
        else:
            _, h, f = frame_shape(tokenizer, 100.0)
            expected = (np.arange(t.size) * f + f / 2) * h / 100.0
        assert expected[-1] > 1800.0
        np.testing.assert_array_equal(t, expected)
        batch, _ = _packer(tokenizer, rows=1).pack([_example([100.0], 1.0, 6)], trim=True)
        assert batch.inputs.time_s.dtype == torch.float64


def test_rotary_angles_are_precise_at_1800_s() -> None:
    model = Encoder(_config(SAMPLES))
    t = 1800.0 + torch.arange(64, dtype=torch.float64) / 500.0
    cos, sin = model.rotary(t[None])
    assert cos.dtype == sin.dtype == torch.float32 and cos.shape == (1, 1, 64, 4)
    half = 32 // 4 // 2
    wavelengths = np.logspace(math.log10(0.01), math.log10(1000.0), half)
    angles = t.numpy()[:, None] * (2 * np.pi / wavelengths)
    np.testing.assert_allclose(cos[0, 0].numpy(), np.cos(angles), atol=1e-6)
    np.testing.assert_allclose(sin[0, 0].numpy(), np.sin(angles), atol=1e-6)
    # the control: the same angles in float32 (time and product) are off by > 0.01
    old = (t.float()[:, None] * torch.tensor(2 * np.pi / wavelengths, dtype=torch.float32)).cos()
    assert np.abs(old.numpy() - np.cos(angles)).max() > 1e-2


@pytest.mark.parametrize("tokenizer", ALL, ids=IDS)
def test_tokens_at_1800_s_have_the_features_of_the_same_tokens_at_0_s(
    tokenizer: AnyTokenizer,
) -> None:
    """Attention depends on time differences only (rotary), so the tokens of a 2.5 s span at
    the end of a 30-min example (all times + 1800 s) have the features they have at 0 s."""
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    batch, _ = _packer(tokenizer, rows=1).pack([_example([100.0, 500.0], 2.5, 8)], trim=True)
    x = batch.inputs
    late = Inputs(**(vars(x) | {"time_s": x.time_s + 1800.0}))
    real = x.document[0] == 0
    with torch.no_grad():
        a, b = model.features(x)[0][real], model.features(late)[0][real]
    torch.testing.assert_close(b, a, atol=2e-5, rtol=1e-4)

"""The v4 local stage on the channel batch (docs/architecture.md sections 3 and 10): no leak
from hidden samples, packing and channel permutation invariance, rate invariance of the
continuous-time SSM, and its discretizations."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.channel_batch import (  # noqa: E402
    ChannelInputs,
    ChannelPacker,
    MaskSpec,
    apply_sample_mask,
)
from biosignal_model.local_stage import (  # noqa: E402
    S4D,
    IdentityLocalSpec,
    LocalStage,
    S4DLocalSpec,
    _plan,
    fast_size,
    local_stage,
)

MASK = MaskSpec(
    ratio=0.6, cycles=0.0, time_spans=0.5, channel_groups=0.3, single_tokens=0.2,
    time_span_s=(0.1, 0.5), group_span_s=(0.1, 0.5),
    cycle_period_s=(0.3, 2.0), cycle_tooth=(0.1, 0.25), cycle_strength=0.3,
)  # fmt: skip


def _spec(**changes: object) -> S4DLocalSpec:
    base = S4DLocalSpec(
        kind="s4d", width=16, layers=3, states=8, min_time_s=0.001, max_time_s=10.0,
        kernel_time_constants=5.0, discretization="zoh_centred", masked_modes="static_gain",
        mixing_layers=[0, 2], mixing_heads=4, dropout=0.0, checkpoint=False,
    )  # fmt: skip
    return base.model_copy(update=changes)


STAGES: dict[str, S4DLocalSpec | IdentityLocalSpec] = {
    "s4d": _spec(),  # the design: centred ZOH, static gains of masked modes
    "s4d-zoh-drop": _spec(discretization="zoh", masked_modes="drop"),
    "s4d-bilinear-checkpoint": _spec(discretization="bilinear", checkpoint=True),
    "identity": IdentityLocalSpec(kind="identity", width=16),
}


def _stage(name: str) -> LocalStage:
    torch.manual_seed(0)
    return local_stage(STAGES[name]).double().eval()


def _example(rates: list[float], seconds: float, seed: int, name: str = "r") -> Example:
    rng = np.random.default_rng(seed)
    signals, channels = [], []
    for i, rate in enumerate(rates):
        values = rng.normal(size=round(seconds * rate)).astype(np.float32)
        signals.append(Signal(values, rate, 0.0))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", f"{name}{seed}", "s", 0.0, tuple(channels)))


def _packer(mask: MaskSpec | None, rows: int, seed: int = 0) -> ChannelPacker:
    return ChannelPacker(
        datasets=["d/x"], token_s=0.04, max_tokens=4096, max_samples=50_000, max_channels=8,
        rows=rows, mask=mask, seed=seed,
    )  # fmt: skip


def _double(x: ChannelInputs) -> ChannelInputs:
    return replace(x, values=x.values.double())


# Rates 100 and 500 Hz each with two channels on one time base (mixed), and 250 Hz alone.
RATES = [100.0, 100.0, 500.0, 500.0, 250.0]


@pytest.mark.parametrize("name", list(STAGES))
def test_hidden_samples_get_no_gradient_through_the_mask_and_the_local_stage(name: str) -> None:
    stage = _stage(name)
    batch, _ = _packer(MASK, rows=1).pack([_example(RATES, 2.0, 0)])
    x, raw = batch.inputs, batch.targets.values.double().requires_grad_(True)
    assert x.hidden.any() and (~x.hidden).any()
    values, observed = apply_sample_mask(raw, batch.targets.observed, x.hidden)
    out = stage(replace(x, values=values, observed=observed))
    weight = torch.randn_like(out)  # every output, none cancelling another
    (out * weight).sum().backward()
    assert raw.grad is not None
    assert (raw.grad[x.hidden] == 0).all(), "a hidden sample reaches an output"
    assert (raw.grad[~x.hidden & batch.targets.observed] != 0).any()


@pytest.mark.parametrize("name", list(STAGES))
def test_a_change_of_hidden_samples_changes_no_output(name: str) -> None:
    stage = _stage(name)
    example = _example(RATES, 2.0, 1)
    batch, _ = _packer(MASK, rows=1, seed=7).pack([example])
    hidden = batch.inputs.hidden.numpy()
    changed, at = [], 0
    for s in example.signals:
        mine = hidden[at : at + s.values.size]
        changed.append(Signal(np.where(mine, s.values + 5.0, s.values).astype(np.float32),
                              s.sampling_rate, s.start_s))  # fmt: skip
        at += s.values.size
    other, _ = _packer(MASK, rows=1, seed=7).pack([Example(tuple(changed), example.origin)])
    assert torch.equal(other.inputs.hidden, batch.inputs.hidden)
    assert not torch.equal(other.targets.values, batch.targets.values)
    with torch.no_grad():
        a, b = stage(_double(batch.inputs)), stage(_double(other.inputs))
    assert torch.equal(a, b), f"largest change: {(a - b).abs().max().item():.2e}"


def _channel_outputs(stage: LocalStage, batch: object, row: int, document: int) -> list[object]:
    x = batch.inputs  # type: ignore[attr-defined]
    with torch.no_grad():
        out = stage(_double(x))
    c = x.channels
    return [
        out[int(c.offset[k]) : int(c.offset[k] + c.length[k])]
        for k in range(c.row.numel())
        if int(c.row[k]) == row and int(c.document[k]) == document
    ]


@pytest.mark.parametrize("name", list(STAGES))
def test_an_example_alone_has_the_features_it_has_packed_with_others(name: str) -> None:
    stage = _stage(name)
    alone = _example(RATES, 2.0, 2)
    before, after = _example([500.0, 100.0, 250.0], 3.7, 3), _example([100.0, 100.0], 1.3, 4)
    single, _ = _packer(None, rows=1).pack([alone])
    together, where = _packer(None, rows=2).pack([before, alone, after])
    assert where[1] == (0, 1)
    a = _channel_outputs(stage, single, 0, 0)
    b = _channel_outputs(stage, together, *where[1])
    assert len(a) == len(b) == len(RATES)
    for p, q in zip(a, b, strict=True):
        torch.testing.assert_close(q, p, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("name", list(STAGES))
def test_the_order_of_the_channels_changes_nothing_without_descriptors(name: str) -> None:
    stage = _stage(name)
    example = _example(RATES, 2.0, 5)
    order = [3, 0, 4, 2, 1]
    o = example.origin
    permuted = Example(
        tuple(example.signals[k] for k in order),
        Origin(o.dataset_id, o.recording_id, o.subject_id, o.start_s,
               tuple(o.channels[k] for k in order)),
    )  # fmt: skip
    a = _channel_outputs(stage, _packer(None, rows=1).pack([example])[0], 0, 0)
    b = _channel_outputs(stage, _packer(None, rows=1).pack([permuted])[0], 0, 0)
    for position, k in enumerate(order):
        torch.testing.assert_close(b[position], a[k], atol=1e-9, rtol=1e-9)


def test_fast_sizes_are_7_smooth() -> None:
    assert [fast_size(n) for n in (1, 11, 13, 17, 97, 1001, 4097)] == [
        1, 12, 14, 18, 98, 1008, 4116,
    ]  # fmt: skip


def test_modes_above_nyquist_are_masked_and_the_kernel_is_capped() -> None:
    torch.manual_seed(0)
    ssm = S4D(_spec())
    _, kept_100 = ssm.modes(100.0)
    _, kept_500 = ssm.modes(500.0)
    assert kept_100.sum() < kept_500.sum()  # more modes below Nyquist at 500 Hz
    assert (kept_500 | ~kept_100).all()  # a mode kept at 100 Hz is kept at 500 Hz
    with torch.no_grad():
        z, kept = ssm.modes(500.0)
    assert (z[kept].abs() <= math.pi).all() and (z[~kept].abs() > math.pi).all()
    slowest = float((1.0 / (-z.real)[kept]).max())  # decay time in samples
    assert ssm.length_at(500.0, 10**9) == math.ceil(5.0 * slowest)
    assert ssm.length_at(500.0, 300) == 300


def _response(spec: S4DLocalSpec, rate: float, frequencies: np.ndarray) -> np.ndarray:
    """The frequency response of one layer's forward kernel (all features) at `rate`, from
    the kernel itself (a long FFT), at `frequencies` (Hz)."""
    torch.manual_seed(0)
    ssm = S4D(spec)
    with torch.no_grad():
        k = ssm.kernel(rate, ssm.length_at(rate, 10**9))[0].double().numpy()
    n = fast_size(max(k.shape[-1], int(rate * 200)))
    spectrum = np.fft.rfft(k, n)
    grid = np.fft.rfftfreq(n, 1 / rate)
    out = np.stack([np.interp(frequencies, grid, spectrum[h].real)
                    + 1j * np.interp(frequencies, grid, spectrum[h].imag)
                    for h in range(k.shape[0])])  # fmt: skip
    return out


def test_zoh_keeps_the_response_across_rates_where_bilinear_warps_it() -> None:
    """The same continuous system at 100 and 500 Hz: with ZOH (poles exact) the magnitude
    response in 1-30 Hz agrees closely across the two rates; bilinear warps the frequency
    axis by (2 fs / w) tan(w / 2 fs), a few % at 30 Hz from 100 Hz."""
    frequencies = np.linspace(1.0, 30.0, 59)
    # time scales 0.1-1 s: every mode is below 50 Hz, so none is masked at 100 Hz and the
    # two rates have the same modes (the comparison is of the discretization alone)
    spec = _spec(width=8, states=8, min_time_s=0.1, max_time_s=1.0)
    errors = {}
    for kind in ("zoh", "zoh_centred", "bilinear"):
        s = spec.model_copy(update={"discretization": kind})
        low, high = (_response(s, rate, frequencies) for rate in (100.0, 500.0))
        errors[kind] = float(np.abs(np.abs(low) - np.abs(high)).max() / np.abs(high).max())
    assert errors["zoh"] < errors["bilinear"], errors
    assert errors["zoh_centred"] < errors["bilinear"], errors


@pytest.mark.parametrize("kind", ["zoh", "zoh_centred"])
def test_with_static_gains_every_rate_has_the_same_dc_gain(kind: str) -> None:
    """Time scales 1-50 ms: at 100 Hz most modes are masked, at 500 Hz few. With
    `static_gain` the DC gain (the sum of the kernel over both directions) is the same at both
    rates; with `drop` it is not."""
    gains = {}
    for masked in ("drop", "static_gain"):
        torch.manual_seed(0)
        ssm = S4D(_spec(min_time_s=0.001, max_time_s=0.05, discretization=kind,
                        masked_modes=masked))  # fmt: skip
        out = []
        for rate in (100.0, 500.0):
            with torch.no_grad():
                length = 6 * ssm.length_at(rate, 10**9)  # 30 decay times: e^-30
                out.append(ssm.kernel(rate, length).double().sum(dim=(0, 2)))
        gains[masked] = out
    low, high = gains["static_gain"]
    torch.testing.assert_close(low, high, rtol=1e-4, atol=1e-5)
    low, high = gains["drop"]
    assert (low - high).abs().max() > 0.1 * high.abs().max()


def _uneven(seed: int) -> Example:
    """Three 100 Hz channels on one time base, one of them shorter (a recording's end), and a
    500 Hz pair: the short group is gathered, the pair is a view."""
    rng = np.random.default_rng(seed)
    sizes = [(100.0, 200), (100.0, 150), (100.0, 200), (500.0, 1000), (500.0, 1000)]
    signals = tuple(Signal(rng.normal(size=n).astype(np.float32), r, 0.0) for r, n in sizes)
    channels = tuple(Channel("a", i, r, 1.0, 0.0, None, f"c{i}", "ecg", "mV")
                     for i, (r, _) in enumerate(sizes))  # fmt: skip
    return Example(signals, Origin("d/x", f"u{seed}", "s", 0.0, channels))


def test_the_stage_order_is_a_permutation_and_uneven_groups_are_gathered() -> None:
    batch, _ = _packer(None, rows=2).pack([_uneven(0), _example(RATES, 1.0, 1)])
    plan = _plan(batch.inputs)
    n = batch.inputs.values.numel()
    assert torch.equal(torch.sort(plan.order).values, torch.arange(n))
    assert torch.equal(plan.order[plan.inverse], torch.arange(n))
    assert sum(b.channels * b.length for b in plan.buckets) == n
    assert len(plan.gathered) == 1 and plan.gathered[0].index.shape == (200, 3)


@pytest.mark.parametrize("name", ["s4d", "s4d-zoh-drop"])
def test_uneven_groups_keep_packing_and_permutation_invariance(name: str) -> None:
    stage = _stage(name)
    example = _uneven(2)
    single, _ = _packer(None, rows=1).pack([example])
    together, where = _packer(None, rows=2).pack([_example(RATES, 1.5, 3), example])
    a = _channel_outputs(stage, single, 0, 0)
    b = _channel_outputs(stage, together, *where[1])
    for p, q in zip(a, b, strict=True):
        torch.testing.assert_close(q, p, atol=1e-9, rtol=1e-9)
    order = [4, 2, 0, 3, 1]
    o = example.origin
    permuted = Example(tuple(example.signals[k] for k in order),
                       Origin(o.dataset_id, o.recording_id, o.subject_id, o.start_s,
                              tuple(o.channels[k] for k in order)))  # fmt: skip
    c = _channel_outputs(stage, _packer(None, rows=1).pack([permuted])[0], 0, 0)
    for position, k in enumerate(order):
        torch.testing.assert_close(c[position], a[k], atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("kind", ["zoh", "zoh_centred", "bilinear"])
@pytest.mark.parametrize("masked", ["drop", "static_gain"])
def test_slow_channels_give_finite_outputs_and_gradients(kind: str, masked: str) -> None:
    """At 1 Hz nearly every mode of a 1 ms-10 s init is masked, some with |z| ~ 1e5."""
    torch.manual_seed(0)
    stage = local_stage(_spec(discretization=kind, masked_modes=masked)).train()
    batch, _ = _packer(None, rows=1).pack([_example([1.0, 1.0, 32.0, 256.0], 30.0, 6)])
    out = stage(batch.inputs)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in stage.parameters() if p.grad is not None)

"""The scan path of the local stage (biosignal_model.ssm_scan, GPU): the same outputs and
gradients as the FFT path, and the same leak, packing and permutation properties."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the scan runs on the GPU", allow_module_level=True)

from torch import Tensor  # noqa: E402

from biosignal_model.channel_batch import ChannelBatch, apply_sample_mask  # noqa: E402
from biosignal_model.local_stage import S4D, S4DLocalStage, fast_size  # noqa: E402
from biosignal_model.ssm_scan import ScanConv  # noqa: E402
from tests.test_local_stage import (  # noqa: E402
    MASK,
    RATES,
    _channel_outputs,
    _example,
    _packer,
    _spec,
    _uneven,
)

CUDA = torch.device("cuda")


def _relative(a: Tensor, b: Tensor) -> float:
    return float((a - b).abs().max() / b.abs().max().clamp(min=1e-30))


def _stages(**changes: object) -> tuple[S4DLocalStage, S4DLocalStage, S4DLocalStage]:
    """The same weights: FFT float32, scan float32, and FFT float64 (the reference)."""
    torch.manual_seed(0)
    fft = S4DLocalStage(_spec(width=32, states=16, **changes)).to(CUDA)
    scan = S4DLocalStage(_spec(width=32, states=16, convolution="scan", **changes)).to(CUDA)
    scan.load_state_dict(fft.state_dict())
    exact = S4DLocalStage(_spec(width=32, states=16, **changes)).to(CUDA).double()
    exact.load_state_dict(fft.state_dict())
    return fft, scan, exact


@pytest.mark.parametrize("kind", ["zoh", "zoh_centred", "bilinear"])
def test_the_scan_is_as_exact_as_the_fft_in_outputs_and_gradients(kind: str) -> None:
    """Both float32 paths against the float64 FFT path: the scan's error is at most the FFT's
    (x 3), or below 1e-6, in the outputs and in every weight's gradient."""
    stages = _stages(discretization=kind)
    examples = [_example(RATES, 2.0, 0), _uneven(1), _example([1.0, 32.0, 256.0], 20.0, 2)]
    batch, _ = _packer(MASK, rows=3).pack(examples)
    x = batch.inputs.to(CUDA)
    outs, grads = [], []
    for stage in stages:
        inputs = replace(x, values=x.values.to(next(stage.parameters()).dtype))
        out = stage(inputs)
        weight = torch.randn(out.shape, generator=torch.Generator(CUDA).manual_seed(1),
                             device=CUDA, dtype=out.dtype)  # fmt: skip
        (out * weight).sum().backward()
        outs.append(out.detach().double())
        grads.append({n: p.grad.double() for n, p in stage.named_parameters()
                      if p.grad is not None})  # fmt: skip
    fft_error, scan_error = (_relative(o, outs[2]) for o in outs[:2])
    assert scan_error <= max(3 * fft_error, 1e-6), (scan_error, fft_error)
    for name, exact in grads[2].items():
        fft_g, scan_g = _relative(grads[0][name], exact), _relative(grads[1][name], exact)
        assert scan_g <= max(3 * fft_g, 1e-6), (name, scan_g, fft_g)


@pytest.mark.parametrize("kind", ["zoh", "zoh_centred", "bilinear"])
def test_the_scan_gradients_of_one_layer_match_float64(kind: str) -> None:
    """ScanConv against the FFT convolution of S4D.kernel in float64: the outputs and the
    gradients to the input and to every S4D parameter (through the modes, not the kernel)."""
    fft, _, _ = _stages(discretization=kind)
    ssm = cast(S4D, cast(Any, fft.layers[0]).ssm)
    channels, length, width, rate = 3, 300, 32, 100.0
    u = torch.randn(channels, length, width, device=CUDA, requires_grad=True)
    dy = torch.randn(channels, length, width, device=CUDA)
    y = ScanConv.apply(u, *ssm.scan_parameters(rate))
    params = [p for p in ssm.parameters() if p is not ssm.d]
    grads = torch.autograd.grad(y, [u, *params], dy)
    exact = S4D(ssm.spec).to(CUDA).double()
    exact.load_state_dict(ssm.state_dict())
    u64 = u.detach().double().requires_grad_(True)
    kernel = exact.kernel(rate, length)
    size = fast_size(2 * length - 1)
    ref = exact(u64.transpose(1, 2).contiguous(), size, kernel).transpose(1, 2)
    params64 = [p for p in exact.parameters() if p is not exact.d]
    grads64 = torch.autograd.grad(ref, [u64, *params64], dy.double())
    assert _relative(y.double(), ref) < 1e-5
    for g, g64 in zip(grads, grads64, strict=True):
        assert _relative(g.double(), g64) < 1e-4


def test_the_scan_keeps_the_leak_packing_and_permutation_properties() -> None:
    _, stage, _ = _stages()
    batch, _ = _packer(MASK, rows=1).pack([_example(RATES, 2.0, 0)])
    x, raw = batch.inputs.to(CUDA), batch.targets.values.to(CUDA).requires_grad_(True)
    values, observed = apply_sample_mask(raw, batch.targets.observed.to(CUDA), x.hidden)
    out = stage(replace(x, values=values, observed=observed))
    (out * torch.randn_like(out)).sum().backward()
    assert raw.grad is not None and (raw.grad[x.hidden] == 0).all()
    alone = _uneven(2)
    single, _ = _packer(None, rows=1).pack([alone])
    together, where = _packer(None, rows=2).pack([_example(RATES, 1.5, 3), alone])

    def outputs(b: ChannelBatch, row: int, document: int) -> list[Tensor]:
        moved = replace(b, inputs=b.inputs.to(CUDA))
        return [cast(Tensor, o).cpu() for o in _channel_outputs(stage, moved, row, document)]

    for p, q in zip(outputs(single, 0, 0), outputs(together, *where[1]), strict=True):
        torch.testing.assert_close(q, p, atol=1e-5, rtol=1e-5)
    order = [4, 2, 0, 3, 1]
    o = alone.origin
    permuted = type(alone)(tuple(alone.signals[k] for k in order),
                           type(o)(o.dataset_id, o.recording_id, o.subject_id, o.start_s,
                                   tuple(o.channels[k] for k in order)))  # fmt: skip
    a = outputs(single, 0, 0)
    c = outputs(_packer(None, rows=1).pack([permuted])[0], 0, 0)
    for position, k in enumerate(order):
        torch.testing.assert_close(c[position], a[k], atol=1e-5, rtol=1e-5)
    assert np.isfinite(out.detach().cpu().numpy()).all()

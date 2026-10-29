"""The v3 tokenizers (docs/v3-tokenizer.md): frames of a fixed duration, the continuous-time
stem, the hierarchical S4D stage; one weight set at 100 and 500 Hz; padding and other examples
never reach pooling or the loss."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.batch import (  # noqa: E402
    DurationTokenizer,
    HierarchicalTokenizer,
    Packer,
    SamplesTokenizer,
    frame_shape,
)
from biosignal_model.benchmarks import ptbxl  # noqa: E402
from biosignal_model.model import Encoder, ModelConfig, S4DBlock, masked_loss  # noqa: E402

D1 = DurationTokenizer(
    kind="duration", token_s=0.08, hop_s=0.04, stem_windows=4, stem_channels=8,
    kernel_frequencies=6, kernel_min_hz=1.0, kernel_max_hz=200.0,
)  # fmt: skip
D2 = HierarchicalTokenizer(
    kind="hierarchical", frame_s=0.04, frame_hop_s=0.02, frames_per_token=12, stem_windows=2,
    stem_channels=8, kernel_frequencies=6, kernel_min_hz=1.0, kernel_max_hz=200.0,
    local_width=16, local_layers=2, local_states=8, pool_heads=4, residual_pool="mean",
)  # fmt: skip
# D2': the token's frames reach the output residual by attention pooling, not their mean
D2_ATTENTION = D2.model_copy(update={"residual_pool": "attention"})


def _config(tokenizer: DurationTokenizer | HierarchicalTokenizer | SamplesTokenizer) -> ModelConfig:
    return ModelConfig(
        samples_per_token=8, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense", tokenizer=tokenizer,
        dropout=0.0, drop_path=0.0,
    )  # fmt: skip


def _example(rates: list[float], seconds: float, seed: int, slow: bool = False) -> Example:
    rng = np.random.default_rng(seed)
    signals, channels = [], []
    for i, rate in enumerate(rates):
        t = np.arange(round(seconds * rate)) / rate
        values = (np.sin(2 * np.pi * 2 * t) + 0.5 * np.cos(2 * np.pi * 3 * t)
                  if slow else rng.normal(size=t.size))  # fmt: skip
        signals.append(Signal(values.astype(np.float32), rate, 0.0))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", f"r{seed}", "s", 0.0, tuple(channels)))


def _packer(tokenizer: DurationTokenizer | HierarchicalTokenizer, rows: int = 2) -> Packer:
    return Packer(
        datasets=["d/x"], max_tokens=1024, samples_per_token=8, max_channels=4, rows=rows,
        mask_ratio=0.0, seed=0, slots={f"c{i}": i for i in range(4)}, tokenizer=tokenizer,
    )  # fmt: skip


def test_frames_have_a_fixed_duration_at_every_rate() -> None:
    assert frame_shape(D1, 100.0) == (8, 4, 1) and frame_shape(D1, 500.0) == (40, 20, 1)
    assert frame_shape(D2, 100.0) == (4, 2, 12) and frame_shape(D2, 500.0) == (20, 10, 12)
    for rate in (100.0, 500.0):
        batch, _ = _packer(D1).pack([_example([rate], 2.5, 0)], trim=True)
        x, n = batch.inputs, frame_shape(D1, rate)[0]
        real = x.document[0] >= 0
        assert int(real.sum()) == 63 and x.values.shape[2:] == (1, n)  # ceil(2.5 s / 0.04 s)
        assert x.time_s[0, :3].tolist() == pytest.approx([0.02, 0.06, 0.10])  # core centres
        # the first frame starts (n - h) / 2 samples before the channel: not observed there
        assert x.observed[0, 0, 0].tolist() == [False] * (n // 4) + [True] * (n - n // 4)
        assert x.rate_hz[0, real].unique().tolist() == [rate]
    batch, _ = _packer(D2).pack([_example([100.0, 500.0], 2.5, 0)], trim=True)
    x = batch.inputs
    real = x.document[0] >= 0
    assert int(real.sum()) == 22 and x.values.shape[2:] == (12, 20)  # 2 x ceil(125 / 12)
    assert x.channel_index[0, real].tolist() == [0] * 11 + [1] * 11
    assert x.position[0, real].tolist() == list(range(11)) * 2
    assert not x.observed[0, :11, :, 4:].any()  # 100 Hz frames hold 4 samples
    assert not x.observed[0, 10, 5:].any() and x.observed[0, 10, 4].any()  # 125 frames


def test_the_stem_leaves_out_frequencies_above_nyquist() -> None:
    stem = Encoder(_config(D1)).stem
    basis = stem.basis(torch.linspace(-0.04, 0.04, 8), 100.0)
    above = (stem.frequencies >= 50.0).nonzero().flatten()
    assert above.numel() and (basis[:, 1 + above] == 0).all()
    assert (basis[:, 1 + 6 + above] == 0).all() and basis[:, 0].eq(1).all()
    assert (stem.basis(torch.linspace(-0.04, 0.04, 40), 500.0)[:, 1:] != 0).any(0).all()


@pytest.mark.parametrize(
    "tokenizer", [D1, D2, D2_ATTENTION], ids=["duration", "hierarchical", "hierarchical-attention"]
)
def test_one_weight_set_reads_100_and_500_hz(
    tokenizer: DurationTokenizer | HierarchicalTokenizer,
) -> None:
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    examples = [_example([100.0], 2.5, 1, slow=True), _example([500.0], 2.5, 1, slow=True),
                _example([100.0, 500.0], 2.5, 2)]  # fmt: skip
    batch, where = _packer(tokenizer, rows=3).pack(examples, trim=True)
    with torch.no_grad():
        h = model.features(batch.inputs)
    assert torch.isfinite(h).all()
    # the same slow signal at both rates: the token vectors agree closely (the stem's kernels
    # are functions of time, normalized as an integral)
    a, b = (h[r][batch.inputs.document[r] == d] for r, d in where[:2])
    assert a.shape == b.shape
    inner = slice(2, -2)  # the edges see different numbers of samples outside the channel
    error = (a[inner] - b[inner]).norm() / a[inner].norm()
    assert error < 0.1, float(error)


@pytest.mark.parametrize(
    "tokenizer", [D1, D2, D2_ATTENTION], ids=["duration", "hierarchical", "hierarchical-attention"]
)
def test_padding_and_other_examples_never_reach_an_example(
    tokenizer: DurationTokenizer | HierarchicalTokenizer,
) -> None:
    torch.manual_seed(0)
    model = Encoder(_config(tokenizer)).eval()
    alone, other = _example([100.0, 500.0], 2.5, 3), _example([500.0, 100.0, 250.0], 1.7, 4)
    single, _ = _packer(tokenizer, rows=1).pack([alone], trim=True)
    together, where = _packer(tokenizer, rows=1).pack([other, alone], trim=False)
    with torch.no_grad():
        a = model.features(single.inputs)[0][single.inputs.document[0] == 0]
        r, d = where[1]
        b = model.features(together.inputs)[r][together.inputs.document[r] == d]
    torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


def test_s4d_keeps_masked_frames_at_zero() -> None:
    torch.manual_seed(0)
    block = S4DBlock(8, 8)
    u = torch.randn(2, 30, 8)
    mask = torch.ones(2, 30, dtype=torch.bool)
    mask[1, 20:] = False
    out = block(u * mask[..., None], mask)
    assert (out[1, 20:] == 0).all() and torch.isfinite(out).all()
    changed = u.clone()
    changed[1, 20:] = 99.0  # masked frames: their values never reach the others
    torch.testing.assert_close(block(changed, mask)[1, :20], out[1, :20])
    assert (block(changed, mask)[1, 20:] == 0).all()


def test_the_duration_loss_uses_observed_samples_of_hidden_tokens_only() -> None:
    packer = Packer(
        datasets=["d/x"], max_tokens=1024, samples_per_token=8, max_channels=4, rows=1,
        mask_ratio=0.5, seed=1, slots=None, tokenizer=D1,
    )  # fmt: skip
    batch, _ = packer.pack([_example([100.0, 500.0], 1.0, 5)], trim=True)
    x = batch.inputs
    model = Encoder(_config(D1))
    predicted = model(x)
    assert predicted.shape == x.values.shape and x.hidden.any() and (~x.hidden[0, :10]).any()
    loss, per_token = masked_loss(predicted, x)
    noise = torch.randn_like(predicted) * (~(x.observed & x.hidden[..., None, None]))
    torch.testing.assert_close(masked_loss(predicted + noise, x)[0], loss)
    assert per_token[~x.hidden].sum() == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.stem.values.weight.grad is not None
    with pytest.raises(NotImplementedError):
        Encoder(_config(D2))(_packer(D2).pack([_example([100.0], 1.0, 6)], trim=True)[0].inputs)


def _records(rate: float) -> ptbxl.Records:
    rng = np.random.default_rng(0)
    codes = [["NORM"], ["IMI"]] * 12
    folds = [1] * 16 + [9] * 4 + [10] * 4
    examples = []
    for i in range(len(codes)):
        n = round(10 * rate)
        signals = tuple(
            Signal((rng.normal(size=n) + (i % 2) * np.sin(np.arange(n) / rate * 2 * np.pi * 5))
                   .astype(np.float32), rate, 0.0) for _ in range(2)
        )  # fmt: skip
        channels = tuple(
            Channel("x", k, rate, 1.0, 0.0, None, f"L{k}", "ecg", "mV") for k in (0, 1)
        )
        examples.append(Example(signals, Origin(ptbxl.DATASET, f"r{i}", f"p{i}", 0.0, channels)))
    return ptbxl.Records([f"r{i}" for i in range(len(codes))], examples, np.array(folds), codes)


STATEMENTS = [
    {"": "NORM", "diagnostic": "1.0", "form": "", "rhythm": "", "diagnostic_class": "NORM",
     "diagnostic_subclass": "NORM"},
    {"": "IMI", "diagnostic": "1.0", "form": "", "rhythm": "", "diagnostic_class": "MI",
     "diagnostic_subclass": "IMI"},
]  # fmt: skip


@pytest.mark.parametrize(
    "tokenizer", [D1, D2, D2_ATTENTION], ids=["duration", "hierarchical", "hierarchical-attention"]
)
@pytest.mark.parametrize("rate", [100.0, 500.0])
def test_ptbxl_fine_tuning_runs_with_the_new_tokenizers(
    tokenizer: DurationTokenizer | HierarchicalTokenizer, rate: float
) -> None:
    records = _records(rate)
    y, _, _ = ptbxl.labels(records, STATEMENTS, "diagnostic")
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["diagnostic"], rows=4, read_threads=1, workers=0,
        linear_probing=False, fixed_slots=True, methods=["from_scratch"], reference=None,
        linear_probing_l2=[1.0], linear_probing_steps=1,
        fine_tuning_epochs=2, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0, fine_tuning_examples_per_step=4,
        fine_tuning_discriminative_factor=0.1, fine_tuning_lower_blocks_fraction=0.25,
        fine_tuning_crop_s=2.5, evaluation_stride_fraction=1.0, evaluation_aggregate="mean",
        fine_tuning_slot_group="lower", fine_tuning_betas=(0.9, 0.95),
        fine_tuning_warmup_epochs=0.5, fine_tuning_schedule="cosine", fine_tuning_grad_clip=1.0,
        fine_tuning_head="linear", from_scratch_samples_per_token=None,
        from_scratch_tokenizer=tokenizer, from_scratch_model=None, seed=0,
    )  # fmt: skip
    logged: list[dict[str, float]] = []
    scores, _ = ptbxl.end_to_end(
        _config(tokenizer), records, y, 1024, settings, torch.device("cpu"), None,
        logged.append, "f",
    )  # fmt: skip
    assert 0.0 <= scores["macro_auc"] <= 1.0 and scores["epochs_run"] == 2.0
    assert all(np.isfinite(r["f/epoch_train_loss"]) for r in logged if "f/epoch" in r)


def test_old_checkpoints_read_as_sample_tokens() -> None:
    from biosignal_model.train import ADDED_LATER  # noqa: PLC0415

    assert ADDED_LATER[("model", "tokenizer")] == {"kind": "samples"}
    s = ptbxl.PtbxlSettings.model_fields["from_scratch_tokenizer"]
    assert s.is_required()  # no default: every config says it

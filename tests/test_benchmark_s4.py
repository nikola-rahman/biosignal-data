"""The S4 reference of Mehari & Strodthoff (benchmarks/s4.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_model.benchmarks import s4  # noqa: E402

SETTINGS = s4.S4Settings(
    input_size_s=0.5, valid_stride_fraction=1.0, validation_on="logits",
    test_on="probabilities", batch_size=4, epochs=2, learning_rate=1e-3, weight_decay=1e-3,
    betas=(0.9, 0.999), eps=1e-8, d_model=8, d_state=8, n_layers=2, dropout=0.2,
    dt_min=1e-3, dt_max=1e-1, loss="bce", class_balanced_beta=None, asymmetric=None,
    ema_decay=None, augmentation=None, schedule="constant", warmup_epochs=0,
    patience_epochs=None, distillation=None, save_chosen_to=None,
)  # fmt: skip


def test_the_nplr_kernel_is_the_kernel_of_the_discrete_state_space() -> None:
    """The generating-function kernel (Cauchy + Woodbury + bilinear) equals K_l = C dA^l dB of
    the dense bilinear discretization, for the original C before the length correction."""
    torch.manual_seed(0)
    length = 16
    kernel = s4.NPLRKernel(
        h=3, n=8, length=length, channels=2, dt_min=1e-3, dt_max=1e-1, dt_scale=1.0
    )
    k = kernel().double()
    da, db = kernel.discrete()
    da, db = da.to(torch.cdouble), db.to(torch.cdouble)
    eye = torch.eye(da.shape[-1], dtype=da.dtype)
    c_tilde = s4._conj(torch.view_as_complex(kernel.C).to(torch.cdouble))  # (c, H, 2N)
    c_orig = torch.linalg.solve(
        eye - s4._power(length, da).transpose(-1, -2), c_tilde.unsqueeze(-1)
    ).squeeze(-1)
    state, dense = db, []
    for _ in range(length):
        dense.append(torch.einsum("chn,hn->ch", c_orig, state).real)
        state = torch.einsum("hmn,hn->hm", da, state)
    dense_k = torch.stack(dense, -1)  # (c, H, L)
    assert k.shape == (2, 3, length) and torch.isfinite(k).all()
    torch.testing.assert_close(k, dense_k, rtol=1e-3, atol=1e-4 * dense_k.abs().max().item())


def test_the_model_has_their_shape_and_drops_whole_features() -> None:
    torch.manual_seed(0)
    model = s4.S4Model(SETTINGS, inputs=12, classes=5, length=50)
    x = torch.randn(3, 12, 50)
    assert model(x).shape == (3, 5)
    model.eval()
    torch.testing.assert_close(model(x), model(x))  # no dropout in eval
    y = s4._channel_dropout(torch.ones(4, 6, 10), 0.5, training=True)
    assert ((y == 0).all(-1) | (y == 2).all(-1)).all()  # a feature is dropped for all times
    layer = s4.S4Layer(SETTINGS, length=50, dt_scale=1.0)
    assert layer(torch.randn(2, 8, 50)).shape == (2, 8, 50)
    buffers = {n for n, _ in model.named_buffers()}
    assert any(n.endswith("log_dt") for n in buffers)  # A, B, dt fixed (trainable=None)
    fixed = ("log_dt", ".w", ".B", ".P", ".Q")
    assert all(not n.endswith(fixed) for n, _ in model.named_parameters())


def test_training_chooses_on_validation_and_scores_test_once_per_best() -> None:
    rng = np.random.default_rng(0)
    folds = np.array([1] * 16 + [9] * 6 + [10] * 6)
    y = rng.integers(0, 2, (28, 3)).astype(np.float32)
    y[:, 0] = [1, 0] * 14
    x = rng.normal(size=(28, 12, 100)).astype(np.float32)
    x[:, 0] += 2 * y[:, :1]  # label 0 is visible in lead I
    logged: list[dict[str, float]] = []

    def auc(yy: np.ndarray, pp: np.ndarray) -> tuple[float, int]:
        from biosignal_model.benchmarks.ptbxl import macro_auc  # noqa: PLC0415

        return macro_auc(yy, pp)

    out, (p_val, p_test) = s4.train_and_score(
        x, y, folds, (1,), (9,), (10,), 100.0, SETTINGS, 0, torch.device("cpu"),
        logged.append, "r", auc,
    )  # fmt: skip
    assert p_val.shape == ((folds == 9).sum(), y.shape[1])
    assert p_test.shape == ((folds == 10).sum(), y.shape[1])
    assert auc(y[folds == 10], p_test)[0] == out["macro_auc"]
    epochs = [r for r in logged if "r/epoch" in r]
    assert [r["r/epoch"] for r in epochs] == [1.0, 2.0]
    chosen = [r for r in epochs if "r/chosen_epoch" in r]
    assert chosen and chosen[-1]["r/chosen_test_macro_auc"] == out["macro_auc"]
    assert out["epoch"] == chosen[-1]["r/chosen_epoch"] and 0.0 <= out["macro_auc"] <= 1.0


def test_every_ptbxl_config_is_valid_and_the_s4_config_has_their_recipe() -> None:
    import yaml  # noqa: PLC0415

    from biosignal_model.benchmarks import ptbxl  # noqa: PLC0415

    runs = Path(__file__).resolve().parents[1] / "runs"
    configs = sorted(runs.glob("bench-ptbxl-*.yaml"))
    assert configs
    for path in configs:
        ptbxl.PtbxlSettings.model_validate(yaml.safe_load(path.read_text())["settings"])
    s = yaml.safe_load((runs / "bench-ptbxl-reference-s4-100hz.yaml").read_text())["settings"]
    r = ptbxl.PtbxlSettings.model_validate(s).reference_s4
    assert r is not None and (r.d_model, r.d_state, r.n_layers, r.batch_size) == (512, 8, 4, 128)
    assert (r.input_size_s, r.epochs, r.learning_rate, r.weight_decay) == (2.5, 100, 1e-3, 1e-3)


CHANNELS = s4.S4ChannelsSettings(
    **SETTINGS.model_dump(), heads=2, base_rate_hz=100.0, identity_std=0.02
)


def test_the_channel_set_model_takes_the_leads_as_a_set() -> None:
    torch.manual_seed(0)
    model = s4.S4ChannelModel(CHANNELS, leads=12, classes=5, length=50, rate_hz=100.0).eval()
    x = torch.randn(3, 12, 50)
    assert model(x).shape == (3, 5) and torch.isfinite(model(x)).all()
    # without the identity vectors, any order of the leads gives the same output
    with torch.no_grad():
        model.identity.zero_()
    order = torch.randperm(12)
    torch.testing.assert_close(model(x[:, order]), model(x), rtol=1e-4, atol=1e-5)
    # with them, the leads are told apart
    with torch.no_grad():
        model.identity.normal_()
    assert not torch.allclose(model(x[:, order]), model(x), atol=1e-4)


def test_the_channel_set_time_steps_follow_the_rate() -> None:
    torch.manual_seed(0)
    at_100 = s4.S4ChannelModel(CHANNELS, leads=2, classes=3, length=50, rate_hz=100.0)
    torch.manual_seed(0)
    at_500 = s4.S4ChannelModel(CHANNELS, leads=2, classes=3, length=50, rate_hz=500.0)
    dt_100 = at_100.layers[0].kernel._parts()[-1]
    dt_500 = at_500.layers[0].kernel._parts()[-1]
    torch.testing.assert_close(dt_500 * 5, dt_100)  # the same steps in seconds


def test_the_channel_set_model_trains_and_is_chosen_on_validation() -> None:
    rng = np.random.default_rng(0)
    folds = np.array([1] * 16 + [9] * 6 + [10] * 6)
    y = rng.integers(0, 2, (28, 3)).astype(np.float32)
    x = rng.normal(size=(28, 12, 100)).astype(np.float32)
    x[:, 0] += 2 * y[:, :1]
    logged: list[dict[str, float]] = []

    def auc(yy: np.ndarray, pp: np.ndarray) -> tuple[float, int]:
        from biosignal_model.benchmarks.ptbxl import macro_auc  # noqa: PLC0415

        return macro_auc(yy, pp)

    out, _ = s4.train_and_score(
        x, y, folds, (1,), (9,), (10,), 100.0, CHANNELS, 0, torch.device("cpu"),
        logged.append, "c", auc,
    )  # fmt: skip
    assert 0.0 <= out["macro_auc"] <= 1.0 and out["epoch"] in (1.0, 2.0)
    assert logged[0]["c/parameters"] > 0


def test_the_channel_set_config_has_the_size_of_the_reference() -> None:
    import yaml  # noqa: PLC0415

    from biosignal_model.benchmarks import ptbxl  # noqa: PLC0415

    runs = Path(__file__).resolve().parents[1] / "runs"
    s = yaml.safe_load((runs / "bench-ptbxl-s4-channels-100hz.yaml").read_text())["settings"]
    c = ptbxl.PtbxlSettings.model_validate(s).reference_s4_channels
    r = yaml.safe_load((runs / "bench-ptbxl-reference-s4-100hz.yaml").read_text())["settings"]
    ref = ptbxl.PtbxlSettings.model_validate(r).reference_s4
    assert c is not None and ref is not None
    size = sum(p.numel() for p in s4.S4ChannelModel(c, 12, 71, 250, 100.0).parameters())
    ref_size = sum(p.numel() for p in s4.S4Model(ref, 12, 71, 250).parameters())
    assert abs(size / ref_size - 1) <= 0.10

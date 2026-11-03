"""The recipe parts for supervised training from scratch (benchmarks/supervised.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_model.benchmarks import s4, supervised  # noqa: E402
from tests.test_benchmark_s4 import SETTINGS  # noqa: E402


def test_class_balanced_weights_follow_the_effective_number() -> None:
    w = supervised.class_balanced_weights(np.array([1, 10, 1000]), beta=0.999)
    e = np.array([(1 - 0.999**n) / (1 - 0.999) for n in (1, 10, 1000)])  # 1, 9.955, 632.3
    np.testing.assert_allclose(w, (1 / e) * 3 / (1 / e).sum())
    assert math.isclose(w.sum(), 3.0) and w[0] > w[1] > w[2]
    loss = supervised.loss_function(
        SETTINGS.model_copy(update={"loss": "class_balanced_bce", "class_balanced_beta": 0.999}),
        np.array([[1, 0], [1, 1], [1, 0]]), torch.device("cpu"),
    )  # fmt: skip
    assert loss is not None
    logits, target = torch.tensor([[0.0, 2.0]]), torch.tensor([[1.0, 0.0]])
    wc = supervised.class_balanced_weights(np.array([3, 1]), 0.999)
    bce = [math.log(2), math.log(1 + math.exp(2))]
    assert math.isclose(loss(logits, target).item(), (wc[0] * bce[0] + wc[1] * bce[1]) / 2,
                        rel_tol=1e-5)  # fmt: skip


def test_asymmetric_loss_on_a_tiny_example() -> None:
    s = supervised.AsymmetricLossSettings(gamma_neg=4, gamma_pos=0, clip=0.05)
    logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    loss = supervised.asymmetric_loss(logits, target, s)
    p1 = 1 / (1 + math.exp(-1.0))
    neg = 1 - p1 + 0.05  # the clipped negative probability
    expected = (math.log(2) + -math.log(neg) * (1 - neg) ** 4) / 2  # positive: gamma_pos 0
    assert math.isclose(loss.item(), expected, rel_tol=1e-6)
    loss.backward()  # the focusing weight has no gradient: d/dx of -w log(neg) with w fixed
    assert math.isclose(logits.grad[0, 1].item(), (1 - neg) ** 4 * p1 * (1 - p1) / neg / 2,
                        rel_tol=1e-5)  # fmt: skip
    easy = supervised.asymmetric_loss(torch.tensor([[-5.0]]), torch.tensor([[0.0]]), s)
    assert easy.item() == 0.0  # p = 0.0067 < clip: an easy negative is discarded


def test_the_moving_average_update() -> None:
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = supervised.Ema(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight, torch.full((1, 2), 1.2))  # 0.9 * 1 + 0.1 * 3
    assert not ema.model.weight.requires_grad


def test_augmentation_shapes_and_ranges() -> None:
    a = supervised.AugmentationSettings(
        stretch=(0.95, 1.05), scale_std=0.1, wander_mv=0.1, wander_hz=(0.167, 0.5),
        noise_mv=0.01, lead_dropout=0.1,
    )  # fmt: skip
    g = torch.Generator().manual_seed(0)
    records = torch.ones(64, 12, 1000)
    x = supervised.Augmentation(a, 100.0).crops(records, 250, g)
    assert x.shape == (64, 12, 250) and torch.isfinite(x).all()
    alive = (x != 0).any(-1)
    assert alive.any(-1).all() and 0.03 < 1 - alive.float().mean().item() < 0.2  # dropout ~0.1
    # a constant record stays near its scaled level: |wander| <= 0.1, noise 0.01, scale ~N(1, .1)
    kept = x[alive]
    assert (kept - kept.mean(-1, keepdim=True)).abs().max() < 0.1 * 2 + 0.06
    none = supervised.AugmentationSettings(
        stretch=(1.0, 1.0), scale_std=0.0, wander_mv=0.0, wander_hz=(0.2, 0.2), noise_mv=0.0,
        lead_dropout=0.0,
    )  # fmt: skip
    ramp = torch.arange(1000.0).expand(2, 3, 1000).contiguous()
    y = supervised.Augmentation(none, 100.0).crops(ramp, 250, g)
    torch.testing.assert_close(y[:, :, 1:] - y[:, :, :-1], torch.ones(2, 3, 249))  # a plain crop
    fast = supervised.AugmentationSettings(**(none.model_dump() | {"stretch": (1.04, 1.04)}))
    z = supervised.Augmentation(fast, 100.0).crops(ramp, 250, g)
    torch.testing.assert_close(z[:, :, -1] - z[:, :, 0], torch.full((2, 3), 259.0))  # 260 taken


def test_warmup_cosine() -> None:
    f = supervised.warmup_cosine(10, 110)
    assert math.isclose(f(0), 0.1) and math.isclose(f(9), 1.0) and math.isclose(f(10), 1.0)
    assert math.isclose(f(60), 0.5, abs_tol=1e-9) and f(110) < 1e-9


def test_training_with_every_part_and_distillation(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    folds = np.array([1] * 16 + [9] * 6 + [10] * 6)
    y = rng.integers(0, 2, (28, 3)).astype(np.float32)
    y[:, 0] = [1, 0] * 14
    x = rng.normal(size=(28, 12, 100)).astype(np.float32)
    x[:, 0] += 2 * y[:, :1]

    def auc(yy: np.ndarray, pp: np.ndarray) -> tuple[float, int]:
        from biosignal_model.benchmarks.ptbxl import macro_auc  # noqa: PLC0415

        return macro_auc(yy, pp)

    weights = tmp_path / "teacher.pt"
    full = SETTINGS.model_copy(update={
        "loss": "asymmetric", "asymmetric": supervised.AsymmetricLossSettings(
            gamma_neg=4, gamma_pos=0, clip=0.05),
        "ema_decay": 0.9, "schedule": "warmup_cosine", "warmup_epochs": 1.0,
        "patience_epochs": 1, "epochs": 4, "save_chosen_to": str(weights),
        "augmentation": supervised.AugmentationSettings(
            stretch=(0.95, 1.05), scale_std=0.1, wander_mv=0.1, wander_hz=(0.167, 0.5),
            noise_mv=0.01, lead_dropout=0.1),
    })  # fmt: skip
    logged: list[dict[str, float]] = []
    out, _ = s4.train_and_score(x, y, folds, (1,), (9,), (10,), 100.0, full, 0,
                                torch.device("cpu"), logged.append, "r", auc)  # fmt: skip
    assert weights.exists() and 0.0 <= out["macro_auc"] <= 1.0
    epochs = [r["r/epoch"] for r in logged if "r/epoch" in r]
    assert epochs[-1] - out["epoch"] <= 1  # patience 1
    assert all("r/epoch_validation_macro_auc_raw" in r for r in logged if "r/epoch" in r)
    student = full.model_copy(update={
        "save_chosen_to": None, "distillation": supervised.DistillationSettings(
            teacher_weights=str(weights), teacher_fraction=0.5),
    })  # fmt: skip
    out2, _ = s4.train_and_score(x, y, folds, (1,), (9,), (10,), 100.0, student, 0,
                                 torch.device("cpu"), logged.append, "d", auc)  # fmt: skip
    assert 0.0 <= out2["macro_auc"] <= 1.0


def test_the_recipe_fields_are_consistent() -> None:
    with pytest.raises(ValueError, match="class_balanced_beta"):
        s4.S4Settings(**(SETTINGS.model_dump() | {"loss": "class_balanced_bce"}))
    with pytest.raises(ValueError, match="warmup_epochs"):
        s4.S4Settings(**(SETTINGS.model_dump() | {"warmup_epochs": 1.0}))

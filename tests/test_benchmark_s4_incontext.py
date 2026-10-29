"""The in-context S4 (benchmarks/s4_incontext.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_model.benchmarks import s4_incontext as ic  # noqa: E402
from biosignal_model.benchmarks.s4 import S4Settings  # noqa: E402

SETTINGS = ic.S4InContextSettings(
    input_size_s=0.5, valid_stride_fraction=1.0, validation_on="logits",
    test_on="probabilities", batch_size=4, epochs=2, learning_rate=1e-3, weight_decay=1e-3,
    betas=(0.9, 0.999), eps=1e-8, d_model=16, d_state=8, n_layers=2, dropout=0.2,
    dt_min=1e-3, dt_max=1e-1, loss="bce", class_balanced_beta=None, asymmetric=None,
    ema_decay=None, augmentation=None, schedule="constant", warmup_epochs=0,
    patience_epochs=None, distillation=None, save_chosen_to=None, context_k=3, icl_layers=2,
    icl_heads=2, icl_feed_forward=32, aux_weight=1.0, context_labels="labels",
)  # fmt: skip


def _memory(n: int, d: int, classes: int, patients: list[int]) -> ic.Memory:
    e = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    y = (torch.rand(n, classes) > 0.5).float()
    return ic.Memory(e, y, torch.tensor(patients))


def test_retrieval_never_takes_a_record_of_the_querys_patient() -> None:
    torch.manual_seed(0)
    memory = _memory(12, 16, 3, [0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8])
    query = memory.embeddings[:4].clone()  # each the same as a memory record
    patients = memory.patients[:4]
    indices, sims = ic.retrieve(query, patients, memory, 5)
    assert indices.shape == sims.shape == (4, 5)
    assert (memory.patients[indices] != patients[:, None]).all()
    assert torch.isfinite(sims).all() and (sims[:, :-1] >= sims[:, 1:]).all()
    # without the exclusion the record itself (cosine 1) would come first
    other = torch.full_like(patients, 99)
    first, _ = ic.retrieve(query, other, memory, 1)
    assert first[:, 0].tolist() == [0, 1, 2, 3]
    # the most similar records of other patients, in order
    sim = query @ memory.embeddings.T
    for b in range(4):
        sim[b, memory.patients == patients[b]] = -2
    torch.testing.assert_close(sims, sim.topk(5, dim=1).values)


def test_the_memory_has_no_gradient_and_follows_the_weights() -> None:
    torch.manual_seed(0)
    model = ic.S4InContextModel(SETTINGS, inputs=12, classes=3, length=50)
    model.train()
    data = torch.randn(10, 12, 100)
    labels = (torch.rand(10, 3) > 0.5).float()
    patients = torch.arange(10)
    ids = np.arange(8)
    starts = [0, 50]
    memory = ic.build_memory(model.encoder, data, ids, labels, patients, starts, 50, 3)
    assert model.encoder.training  # back in its mode
    assert memory.embeddings.shape == (8, 16) and not memory.embeddings.requires_grad
    torch.testing.assert_close(memory.embeddings.norm(dim=-1), torch.ones(8))
    assert torch.equal(memory.labels, labels[:8]) and torch.equal(memory.patients, patients[:8])
    model.eval()
    with torch.no_grad():  # the mean of the eval crops' embeddings, normalized
        e = (model.encoder(data[:8, :, :50]) + model.encoder(data[:8, :, 50:])) / 2
    torch.testing.assert_close(memory.embeddings, torch.nn.functional.normalize(e, dim=-1))
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(0.1 * torch.randn_like(p))
    again = ic.build_memory(model.encoder, data, ids, labels, patients, starts, 50, 3)
    assert not torch.allclose(again.embeddings, memory.embeddings)


def test_shapes_and_zero_label_input_of_the_control() -> None:
    torch.manual_seed(0)
    memory = _memory(20, 16, 3, list(range(20)))
    flipped = ic.Memory(memory.embeddings, 1 - memory.labels, memory.patients)
    x = torch.randn(5, 12, 50)
    patients = torch.full((5,), 100)
    outputs = {}
    for mode in ("labels", "zeros"):
        torch.manual_seed(1)
        model = ic.S4InContextModel(
            SETTINGS.model_copy(update={"context_labels": mode}), 12, 3, 50
        ).eval()
        icl, aux = model(x, patients, memory)
        assert icl.shape == aux.shape == (5, 3)
        assert torch.equal(model.label_input(memory.labels), memory.labels if mode == "labels"
                           else torch.zeros_like(memory.labels))  # fmt: skip
        outputs[mode] = (icl, model(x, patients, flipped)[0])
    assert not torch.allclose(*outputs["labels"])  # the labels are an input
    torch.testing.assert_close(*outputs["zeros"])  # the labels are not an input


def test_the_in_context_path_trains_the_encoder() -> None:
    torch.manual_seed(0)
    model = ic.S4InContextModel(SETTINGS, 12, 3, 50).train()
    memory = _memory(20, 16, 3, list(range(20)))
    icl, _ = model(torch.randn(4, 12, 50), torch.full((4,), 100), memory)
    icl.sum().backward()
    for name, p in model.encoder.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert model.aux_head.weight.grad is None  # only the in-context loss was used
    assert memory.embeddings.grad is None and not memory.embeddings.requires_grad


def test_a_tiny_run_refreshes_the_memory_and_logs_like_reference_s4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from biosignal_model.benchmarks.ptbxl import macro_auc  # noqa: PLC0415

    rng = np.random.default_rng(0)
    folds = np.array([1] * 16 + [9] * 6 + [10] * 6)
    patients = np.array([k // 2 for k in range(16)] + list(range(100, 112)))
    y = rng.integers(0, 2, (28, 3)).astype(np.float32)
    y[:, 0] = [1, 0] * 14
    x = rng.normal(size=(28, 12, 100)).astype(np.float32)
    x[:, 0] += 2 * y[:, :1]
    refreshes: list[int] = []
    build = ic.build_memory

    def counted(*args: object, **kwargs: object) -> ic.Memory:
        memory = build(*args, **kwargs)  # type: ignore[arg-type]
        refreshes.append(len(memory.patients))
        return memory

    monkeypatch.setattr(ic, "build_memory", counted)
    logged: list[dict[str, float]] = []
    out, (p_val, p_test) = ic.train_and_score(
        x, y, folds, patients, (1,), (9,), (10,), 100.0, SETTINGS, 0, torch.device("cpu"),
        logged.append, "all/reference_s4_incontext", macro_auc,
    )  # fmt: skip
    assert refreshes == [16, 16, 16]  # before epoch 1 and after each epoch, folds 1-8 only
    assert p_val.shape == (6, 3) and p_test.shape == (6, 3)
    assert macro_auc(y[folds == 10], p_test)[0] == out["macro_auc"]
    prefix = "all/reference_s4_incontext"
    epochs = [r for r in logged if f"{prefix}/epoch" in r]
    assert [r[f"{prefix}/epoch"] for r in epochs] == [1.0, 2.0]
    for r in epochs:
        assert {f"{prefix}/epoch_validation_macro_auc", f"{prefix}/epoch_validation_macro_auc_aux",
                f"{prefix}/memory_refresh_s", f"{prefix}/epoch_train_loss"} <= set(r)  # fmt: skip
    chosen = [r for r in epochs if f"{prefix}/chosen_epoch" in r]
    assert chosen and chosen[-1][f"{prefix}/chosen_test_macro_auc"] == out["macro_auc"]
    assert out["epoch"] == chosen[-1][f"{prefix}/chosen_epoch"]


def test_too_few_other_patients_and_other_recipe_parts_are_refused() -> None:
    with pytest.raises(ValueError, match="published recipe"):
        ic.S4InContextSettings(**{**SETTINGS.model_dump(), "ema_decay": 0.999})
    folds = np.array([1] * 4 + [9, 10])
    with pytest.raises(ValueError, match="context_k"):
        ic.train_and_score(
            np.zeros((6, 12, 100), np.float32), np.zeros((6, 3), np.float32), folds,
            np.array([0, 0, 1, 2, 3, 4]), (1,), (9,), (10,), 100.0,
            SETTINGS.model_copy(update={"context_k": 3}), 0, torch.device("cpu"),
            lambda _: None, "r", lambda a, b: (0.5, 1),
        )  # fmt: skip


def test_the_in_context_configs_differ_only_in_the_label_input() -> None:
    import yaml  # noqa: PLC0415

    from biosignal_model.benchmarks import ptbxl  # noqa: PLC0415

    runs = Path(__file__).resolve().parents[1] / "runs"

    def recipe(name: str) -> ic.S4InContextSettings:
        s = yaml.safe_load((runs / name).read_text())["settings"]
        settings = ptbxl.PtbxlSettings.model_validate(s)
        assert settings.methods == ["reference_s4_incontext"] and settings.seed == 0
        r = settings.reference_s4_incontext
        assert r is not None
        return r

    k32 = recipe("bench-ptbxl-incontext-k32-100hz.yaml")
    control = recipe("bench-ptbxl-incontext-k32-nolabels-100hz.yaml")
    assert (k32.context_k, k32.icl_layers, k32.icl_heads, k32.icl_feed_forward) == (32, 4, 8, 2048)
    assert (k32.aux_weight, k32.context_labels, control.context_labels) == (1.0, "labels", "zeros")
    assert k32.patience_epochs == 25
    ignore = {"context_labels", "save_chosen_to"}
    assert k32.model_dump(exclude=ignore) == control.model_dump(exclude=ignore)
    s = yaml.safe_load((runs / "bench-ptbxl-sota-baseline-100hz.yaml").read_text())["settings"]
    baseline = S4Settings.model_validate(s["reference_s4"])
    assert k32.model_dump(include=set(S4Settings.model_fields) - {"save_chosen_to"}) == (
        baseline.model_dump(exclude={"save_chosen_to"})
    )

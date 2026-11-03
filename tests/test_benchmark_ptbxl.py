from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.batch import Inputs, SamplesTokenizer  # noqa: E402
from biosignal_model.benchmarks import BenchmarkRun, ptbxl  # noqa: E402
from biosignal_model.model import Encoder, ModelConfig  # noqa: E402

STATEMENTS = [  # a few rows of scp_statements.csv
    {"": "NORM", "diagnostic": "1.0", "form": "", "rhythm": "", "diagnostic_class": "NORM",
     "diagnostic_subclass": "NORM"},
    {"": "IMI", "diagnostic": "1.0", "form": "", "rhythm": "", "diagnostic_class": "MI",
     "diagnostic_subclass": "IMI"},
    {"": "LVOLT", "diagnostic": "", "form": "1.0", "rhythm": "", "diagnostic_class": "",
     "diagnostic_subclass": ""},
    {"": "SR", "diagnostic": "", "form": "", "rhythm": "1.0", "diagnostic_class": "",
     "diagnostic_subclass": ""},
    {"": "AFIB", "diagnostic": "", "form": "", "rhythm": "1.0", "diagnostic_class": "",
     "diagnostic_subclass": ""},
]  # fmt: skip


def _records(codes: list[list[str]], folds: list[int]) -> ptbxl.Records:
    rng = np.random.default_rng(0)
    examples = []
    for i in range(len(codes)):
        signals = tuple(
            Signal(rng.normal(size=1000).astype(np.float32), 100.0, 0.0) for _ in range(2)
        )
        channels = tuple(
            Channel("100Hz", k, 100.0, 1.0, 0.0, None, f"L{k}", "ecg", "mV") for k in range(2)
        )
        examples.append(Example(signals, Origin(ptbxl.DATASET, f"r{i}", f"p{i}", 0.0, channels)))
    return ptbxl.Records([f"r{i}" for i in range(len(codes))], examples, np.array(folds), codes)


def test_labels_follow_the_authors_aggregation() -> None:
    # record 0: {'NORM': 100, 'LVOLT': 0, 'SR': 0}: a key counts whatever its likelihood
    records = _records([["NORM", "LVOLT", "SR"], ["IMI", "AFIB"], ["SR"]], [1, 9, 10])
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "all")
    assert classes == ["AFIB", "IMI", "LVOLT", "NORM", "SR"] and keep.all()
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "superdiagnostic")
    assert classes == ["MI", "NORM"] and keep.tolist() == [True, True, False]
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "rhythm")
    assert classes == ["AFIB", "SR"] and y.tolist() == [[0, 1], [1, 0], [0, 1]]
    y, classes, keep = ptbxl.labels(records, STATEMENTS, "form")
    assert classes == ["LVOLT"] and keep.tolist() == [True, False, False]


def test_macro_auc_gives_ties_their_average_rank_and_skips_one_class_labels() -> None:
    y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p = np.array([[0.5, 0.1], [0.5, 0.2], [0.9, 0.3], [0.1, 0.4]])
    auc, n = ptbxl.macro_auc(y, p)
    # label 0: pairs (pos, neg) = (0.5,0.5) tie -> 0.5, (0.5,0.1) 1, (0.9,0.5) 1, (0.9,0.1) 1
    assert n == 1 and auc == pytest.approx(3.5 / 4)


def test_the_probe_chooses_on_fold_9_and_scores_fold_10() -> None:
    rng = np.random.default_rng(2)
    folds = np.array([1] * 300 + [9] * 100 + [10] * 100)
    y = rng.integers(0, 2, (500, 2)).astype(np.float32)
    features = rng.normal(size=(500, 6))
    features[:, 0] += 3 * y[:, 0]
    features[:, 1] += 3 * y[:, 1]
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["all"], rows=2, read_threads=1, workers=0,
        linear_probing=True, fixed_slots=False,
        methods=["fine_tuning", "from_scratch"], reference=None, reference_s4=None,
        reference_s4_channels=None, reference_s4_incontext=None,
        linear_probing_l2=[1e-3, 1.0], linear_probing_steps=50,
        fine_tuning_epochs=1, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0,
        fine_tuning_examples_per_step=8, fine_tuning_discriminative_factor=0.1,
        fine_tuning_lower_blocks_fraction=0.25, fine_tuning_crop_s=None,
        evaluation_stride_fraction=1.0, evaluation_aggregate="mean", fine_tuning_slot_group="lower",
        fine_tuning_betas=(0.9, 0.999), fine_tuning_warmup_epochs=0.0,
        fine_tuning_schedule="constant", fine_tuning_grad_clip=None, fine_tuning_head="linear",
        from_scratch_samples_per_token=None, from_scratch_tokenizer=None, from_scratch_model=None,
        fine_tuning_rungs=None, fine_tuning_label_fraction=1.0,
        seed=0,
    )  # fmt: skip
    logged: list[dict[str, float]] = []
    scores = ptbxl.linear_probing(
        features, y, folds, settings, logged.append, "p", torch.device("cpu")
    )
    assert scores["macro_auc"] > 0.95 and scores["labels"] == 2 and scores["l2"] in (1e-3, 1.0)
    assert [r["p/l2"] for r in logged] == [1e-3, 1.0]  # every L2 tried, with its fold-9 AUC
    assert max(r["p/validation_macro_auc"] for r in logged) == scores["validation_macro_auc"]


def test_fine_tuning_runs_end_to_end_on_a_small_model(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
        tokenizer=SamplesTokenizer(kind="samples"), dropout=0.0, drop_path=0.0,
    )  # fmt: skip
    codes = [["NORM"], ["IMI"]] * 12
    records = _records(codes, [1] * 16 + [9] * 4 + [10] * 4)
    y, _, _ = ptbxl.labels(records, STATEMENTS, "diagnostic")
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["diagnostic"], rows=2, read_threads=1, workers=0,
        linear_probing=True, fixed_slots=False,
        methods=["fine_tuning", "from_scratch"], reference=None, reference_s4=None,
        reference_s4_channels=None, reference_s4_incontext=None,
        linear_probing_l2=[1.0], linear_probing_steps=10,
        fine_tuning_epochs=2, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0,
        fine_tuning_examples_per_step=4, fine_tuning_discriminative_factor=0.1,
        fine_tuning_lower_blocks_fraction=0.25, fine_tuning_crop_s=2.5,
        evaluation_stride_fraction=1.0, evaluation_aggregate="mean", fine_tuning_slot_group="lower",
        fine_tuning_betas=(0.9, 0.999), fine_tuning_warmup_epochs=0.0,
        fine_tuning_schedule="constant", fine_tuning_grad_clip=None, fine_tuning_head="linear",
        from_scratch_samples_per_token=None, from_scratch_tokenizer=None, from_scratch_model=None,
        fine_tuning_rungs=None, fine_tuning_label_fraction=1.0,
        seed=0,
    )  # fmt: skip
    device = torch.device("cpu")
    logged: list[dict[str, float]] = []
    for start in (None, Encoder(config).state_dict()):
        logged.clear()
        scores, _ = ptbxl.end_to_end(
            config, records, y, 256, settings, device, start, logged.append, "f"
        )
        assert 0.0 <= scores["macro_auc"] <= 1.0 and scores["epoch"] in (1.0, 2.0)
        epochs = [r for r in logged if "f/epoch" in r]
        assert [r["f/epoch"] for r in epochs] == [1.0, 2.0]
        assert all(np.isfinite(r["f/epoch_train_loss"]) for r in epochs)
        chosen = [r for r in epochs if "f/chosen_epoch" in r]  # fold 10 only when chosen
        assert chosen[-1]["f/chosen_test_macro_auc"] == scores["macro_auc"]
    # With patience 2, the run stops 2 epochs after its best fold-9 epoch (epoch 1 here).
    aucs = [0.6, 0.5, 0.5, 0.5]  # fold 9 of epoch 1, fold 10 of epoch 1, fold 9 of 2 and 3
    monkeypatch.setattr(ptbxl, "macro_auc", lambda y, p: (aucs.pop(0), 1))
    patient = settings.model_copy(update={"fine_tuning_epochs": 6, "fine_tuning_patience": 2})
    logged.clear()
    scores, _ = ptbxl.end_to_end(config, records, y, 256, patient, device, None, logged.append, "f")
    assert scores["epochs_run"] == 3.0 and scores["epoch"] == 1.0 and scores["macro_auc"] == 0.5
    assert [r["f/epoch"] for r in logged if "f/epoch" in r] == [1.0, 2.0, 3.0] and aucs == []
    monkeypatch.undo()
    # Every option of the 2026-10-29 tests together: they run and give finite scores.
    options = settings.model_copy(update={
        "fixed_slots": True, "fine_tuning_slot_group": "head", "evaluation_stride_fraction": 0.5,
        "evaluation_aggregate": "max", "fine_tuning_betas": (0.9, 0.95),
        "fine_tuning_warmup_epochs": 0.5, "fine_tuning_schedule": "cosine",
        "fine_tuning_grad_clip": 1.0, "fine_tuning_head": "reference",
    })  # fmt: skip
    logged.clear()
    scores, _ = ptbxl.end_to_end(config, records, y, 256, options, device, None, logged.append, "o")
    assert 0.0 <= scores["macro_auc"] <= 1.0 and scores["epochs_run"] == 2.0
    assert all(np.isfinite(r["o/epoch_train_loss"]) for r in logged if "o/epoch" in r)
    logged.clear()
    features = ptbxl.embed(
        Encoder(config), records.examples, 256, 2, device, logged.append, "e", None
    )
    assert features.shape == (24, 32) and np.isfinite(features).all()
    assert logged[-1]["embed/e/batches"] == logged[-1]["embed/e/of"]


def test_layer_dependent_learning_rates_cover_every_weight_a_task_uses() -> None:
    config = ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=8, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
        tokenizer=SamplesTokenizer(kind="samples"), dropout=0.0, drop_path=0.0,
    )  # fmt: skip
    model, head = Encoder(config), torch.nn.Linear(32, 3)
    groups = ptbxl.parameter_groups(model, head, 1e-3, 0.1, 0.25, "lower")
    assert [g["lr"] for g in groups] == pytest.approx([1e-3, 1e-4, 1e-5])
    ids = [{id(p) for p in g["params"]} for g in groups]
    used = {id(p) for p in head.parameters()} | {
        id(p) for n, p in model.named_parameters() if not n.startswith("head.")
    }
    assert set.union(*ids) == used and sum(map(len, ids)) == len(used)  # each once
    g = model.layer_groups()
    lower = {id(p) for layer in g.layers[:2] for p in layer} | {id(p) for p in g.input}
    assert lower <= ids[2] and not lower & ids[1]  # 2 of 8 layers, with the input part
    assert id(model.hidden) in lower and len(g.layers) == 8
    assert {id(p) for p in model.slot.parameters()} <= ids[2]
    moved = ptbxl.parameter_groups(model, head, 1e-3, 0.1, 0.25, "head")
    slot = {id(p) for p in g.channel}
    assert slot == {id(p) for p in model.slot.parameters()}
    assert slot <= {id(p) for p in moved[0]["params"]}
    assert not slot & {id(p) for p in moved[2]["params"]}


def test_half_overlapping_crops_as_strodthoff() -> None:
    example = _records([["NORM"]], [1]).examples[0]  # 10 s at 100 Hz
    crops = ptbxl._crops(example, 2.5, 0.5)  # stride_length_valid = input_size // 2
    assert len(crops) == 7 and all(len(s.values) == 250 for c in crops for s in c.signals)
    np.testing.assert_array_equal(crops[1].signals[0].values, example.signals[0].values[125:375])


def test_the_schedule_warms_up_then_decays_to_zero() -> None:
    s = _settings_for_schedule("cosine")
    f = ptbxl._schedule(s, steps_per_epoch=10)  # 2 warmup epochs, 10 epochs
    assert f(0) == pytest.approx(0.05) and f(19) == pytest.approx(1.0)
    assert f(20) == pytest.approx(1.0) and f(99) < 0.01 and f(60) == pytest.approx(0.5)
    assert ptbxl._schedule(_settings_for_schedule("constant"), 10)(99) == 1.0


def _settings_for_schedule(schedule: Literal["constant", "cosine"]) -> ptbxl.PtbxlSettings:
    return ptbxl.PtbxlSettings(
        rate=100, tasks=["all"], rows=2, read_threads=1, workers=0, linear_probing=False,
        fixed_slots=False, methods=["fine_tuning"], reference=None, reference_s4=None,
        reference_s4_channels=None, reference_s4_incontext=None,
        linear_probing_l2=[1.0], linear_probing_steps=1,
        fine_tuning_epochs=10, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0, fine_tuning_examples_per_step=4,
        fine_tuning_discriminative_factor=0.1, fine_tuning_lower_blocks_fraction=0.25,
        fine_tuning_crop_s=2.5, evaluation_stride_fraction=0.5, evaluation_aggregate="max",
        fine_tuning_slot_group="head", fine_tuning_betas=(0.9, 0.95),
        fine_tuning_warmup_epochs=2.0, fine_tuning_schedule=schedule,
        fine_tuning_grad_clip=1.0, fine_tuning_head="reference",
        from_scratch_samples_per_token=None, from_scratch_tokenizer=None, from_scratch_model=None,
        fine_tuning_rungs=None, fine_tuning_label_fraction=1.0,
        seed=0,
    )  # fmt: skip


def test_crops_cover_a_record_without_overlap() -> None:
    example = _records([["NORM"]], [1]).examples[0]  # 10 s at 100 Hz
    crops = ptbxl._crops(example, 2.5, 1.0)
    assert len(crops) == 4 and all(len(s.values) == 250 for c in crops for s in c.signals)
    np.testing.assert_array_equal(
        np.concatenate([c.signals[0].values for c in crops]), example.signals[0].values
    )
    import random  # noqa: PLC0415

    crop = ptbxl._random_crop(example, 2.5, random.Random(0))  # noqa: S311 - not secret
    values = example.signals[0].values
    start = int(np.flatnonzero(values == crop.signals[0].values[0])[0])
    np.testing.assert_array_equal(crop.signals[0].values, values[start : start + 250])
    np.testing.assert_array_equal(
        crop.signals[1].values, example.signals[1].values[start : start + 250]
    )


def _small() -> ModelConfig:
    return ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=2, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
        tokenizer=SamplesTokenizer(kind="samples"), dropout=0.0, drop_path=0.0,
    )  # fmt: skip


def test_fixed_slots_give_each_lead_name_one_slot() -> None:
    examples = _records([["NORM"]] * 3, [1] * 3).examples
    names = sorted({c.name for e in examples for c in e.origin.channels})
    assert ptbxl.lead_slots(examples, fixed=False) is None
    assert ptbxl.lead_slots(examples, fixed=True) == {n: k for k, n in enumerate(names)}


def test_a_trimmed_batch_gives_each_example_the_same_features() -> None:
    config = _small()
    model = Encoder(config).eval()
    examples = _records([["NORM"]] * 5, [1] * 5).examples
    packer = ptbxl._packer(config, 512, 4, None)  # 40 tokens per example
    device = torch.device("cpu")
    packer.reseed(1)
    full, where_full = packer.pack(examples, trim=False)
    packer.reseed(1)
    trimmed, where = packer.pack(examples, trim=True)
    assert where == where_full
    assert trimmed.inputs.values.shape[:2] == (1, 256)  # 200 tokens, to a multiple of 128
    assert full.inputs.values.shape[:2] == (4, 512)
    with torch.no_grad():
        a = ptbxl._mean_features(model, full, where, device)
        b = ptbxl._mean_features(model, trimmed, where, device)
        x = full.inputs
        assert isinstance(x, Inputs)
        loop = torch.stack([model.features(x)[r][x.document[r] == d].mean(0) for r, d in where])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a, loop, rtol=1e-5, atol=1e-5)


def _crop(example: Example, rng: random.Random) -> Example:
    """At module level, so that workers started by spawn or forkserver can import it."""
    return ptbxl._random_crop(example, 2.5, rng)


def test_batches_made_in_workers_are_the_same_as_in_the_main_process() -> None:
    config = _small()
    examples = _records([["NORM"]] * 12, [1] * 12).examples
    groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    made = {}
    for workers in (0, 2):
        packer = ptbxl._packer(config, 256, 2, None)
        made[workers] = list(ptbxl._packed(examples, groups, _crop, packer, "s/1", workers))
    assert len(made[0]) == len(made[2]) == 3
    for (a, wa), (b, wb) in zip(made[0], made[2], strict=True):
        assert wa == wb
        for name in ("values", "observed", "time_s", "slot", "document"):
            torch.testing.assert_close(getattr(a.inputs, name), getattr(b.inputs, name))


def test_the_reference_xresnet1d101_has_the_papers_shape_schedule_and_crops() -> None:
    from biosignal_model.benchmarks import xresnet1d  # noqa: PLC0415

    model = xresnet1d.xresnet1d101(12, 71, 5, 0.5, [128]).eval()
    assert model(torch.zeros(2, 12, 250)).shape == (2, 71)
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
    # the stem, 3 per block, and one projection (the first block: 64 -> 256 channels)
    assert len(convs) == 3 + 3 * 33 + 1
    last = model[-1]
    assert isinstance(last, torch.nn.Sequential)
    head = list(last)
    assert [type(m).__name__ for m in head] == [
        "_ConcatPool", "BatchNorm1d", "Dropout", "Linear", "ReLU", "BatchNorm1d", "Dropout",
        "Linear",
    ] and head[2].p == 0.25 and head[6].p == 0.5  # fmt: skip
    s = xresnet1d.ReferenceSettings(
        input_size_s=2.5, valid_stride_fraction=0.5, aggregate="max", batch_size=4, epochs=1,
        max_lr=1e-2, weight_decay=1e-2, pct_start=0.3, div_factor=25.0, final_div=25e4,
        moms=(0.95, 0.85), beta2=0.99, eps=1e-8, kernel_size=5, ps_head=0.5, lin_ftrs_head=[128],
    )  # fmt: skip
    assert xresnet1d.one_cycle(0, 100, s) == pytest.approx((1e-2 / 25, 0.95))
    assert xresnet1d.one_cycle(30, 100, s) == pytest.approx((1e-2, 0.85))
    assert xresnet1d.one_cycle(100, 100, s) == pytest.approx((1e-2 / 25e4, 0.95))
    assert xresnet1d.crop_starts(1000, 250, 125) == [0, 125, 250, 375, 500, 625, 750]
    rng = np.random.default_rng(0)
    y = np.tile(np.eye(2), (6, 1))
    x = rng.normal(size=(12, 12, 1000)).astype(np.float32) + y[:, :1, None] * 3.0
    logged: list[dict[str, float]] = []
    scores, (p_val, p_test) = xresnet1d.train_and_score(
        x,
        y,
        np.array([1] * 8 + [9] * 2 + [10] * 2),
        (1,),
        (9,),
        (10,),
        100.0,
        s,
        0,
        torch.device("cpu"),
        logged.append,
        "r",
        ptbxl.macro_auc,
    )
    assert 0.0 <= scores["macro_auc"] <= 1.0 and scores["epoch"] == 1.0
    assert p_val.shape == p_test.shape == (2, 2)
    assert ptbxl.macro_auc(y[10:], p_test)[0] == scores["macro_auc"]
    assert [r["r/epoch"] for r in logged] == [1.0]


def test_label_groups_are_scored_on_their_own_and_fold_9_still_chooses() -> None:
    config = ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
        tokenizer=SamplesTokenizer(kind="samples"), dropout=0.1, drop_path=0.1,
    )  # fmt: skip
    codes = [["NORM", "SR"], ["IMI", "LVOLT"], ["NORM", "LVOLT"], ["IMI", "SR"]] * 6
    records = _records(codes, [1] * 16 + [9] * 4 + [10] * 4)
    y, classes, _ = ptbxl.labels(records, STATEMENTS, "all")
    groups = ptbxl.label_groups(classes, STATEMENTS)
    assert {g: [c for c, k in zip(classes, m, strict=True) if k] for g, m in groups.items()} == {
        "diagnostic": ["IMI", "NORM"],
        "form": ["LVOLT"],
        "rhythm": ["SR"],
    }
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["all"], rows=2, read_threads=1, workers=0,
        linear_probing=False, fixed_slots=False, methods=["from_scratch"], reference=None,
        reference_s4=None, reference_s4_channels=None, reference_s4_incontext=None,
        linear_probing_l2=[1.0], linear_probing_steps=10,
        fine_tuning_epochs=2, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0, fine_tuning_examples_per_step=4,
        fine_tuning_discriminative_factor=0.1, fine_tuning_lower_blocks_fraction=0.25,
        fine_tuning_crop_s=2.5, evaluation_stride_fraction=1.0, evaluation_aggregate="mean",
        fine_tuning_slot_group="lower", fine_tuning_betas=(0.9, 0.999),
        fine_tuning_warmup_epochs=0.0, fine_tuning_schedule="constant",
        fine_tuning_grad_clip=None, fine_tuning_head="linear",
        from_scratch_samples_per_token=None, from_scratch_tokenizer=None, from_scratch_model=None,
        fine_tuning_rungs=None, fine_tuning_label_fraction=1.0,
        seed=0,
    )  # fmt: skip
    logged: list[dict[str, float]] = []
    scores, (p_val, p_test) = ptbxl.end_to_end(
        config,
        records,
        y,
        256,
        settings,
        torch.device("cpu"),
        None,
        logged.append,
        "f",
        scored_groups=groups,
    )
    # the chosen epoch's probabilities: fold 9 and fold 10, records in order
    assert p_val.shape == p_test.shape == (4, len(classes))
    assert ptbxl.macro_auc(y[16:20], p_val)[0] == scores["validation_macro_auc"]
    assert ptbxl.macro_auc(y[20:], p_test)[0] == scores["macro_auc"]
    epochs = [r for r in logged if "f/epoch" in r]
    for g in groups:
        assert all(0.0 <= r[f"f/epoch_validation_macro_auc/{g}"] <= 1.0 for r in epochs)
        chosen = [r for r in epochs if f"f/chosen_test_macro_auc/{g}" in r]
        assert chosen[-1][f"f/chosen_test_macro_auc/{g}"] == scores[f"macro_auc_{g}"]
    best = max(epochs, key=lambda r: r["f/epoch_validation_macro_auc"])
    assert scores["epoch"] == best["f/epoch"]  # all labels on fold 9 choose the epoch


def test_cooldown_branches_score_each_rung_and_leave_the_main_run_as_it_was() -> None:
    config = ModelConfig(
        samples_per_token=50, max_channels=4, width=32, layers=1, heads=4, feed_forward=64,
        min_wavelength_s=0.01, max_wavelength_s=1000.0, attention="dense",
        tokenizer=SamplesTokenizer(kind="samples"), dropout=0.1, drop_path=0.1,
    )  # fmt: skip
    codes = [["NORM", "SR"], ["IMI", "LVOLT"], ["NORM", "LVOLT"], ["IMI", "SR"]] * 6
    records = _records(codes, [1] * 16 + [9] * 4 + [10] * 4)
    y, classes, _ = ptbxl.labels(records, STATEMENTS, "all")
    groups = ptbxl.label_groups(classes, STATEMENTS)
    settings = _settings_for_schedule("constant").model_copy(update={
        "fine_tuning_epochs": 4, "fine_tuning_warmup_epochs": 0.5,
        "fine_tuning_schedule": "constant", "fine_tuning_patience": None,
        "fine_tuning_examples_per_step": 4, "fine_tuning_crop_s": 2.5, "workers": 0,
    })  # fmt: skip
    rungs = ptbxl.CooldownRungs(epochs=[1, 2, 4], fraction=0.5, min_epochs=1)
    assert [ptbxl.cooldown_epochs(rungs, e) for e in (1, 2, 4)] == [1, 1, 2]
    assert [ptbxl.cooldown_epochs(ptbxl.CooldownRungs(epochs=[20], fraction=0.2, min_epochs=2), e)
            for e in (5, 10, 20)] == [2, 2, 4]  # fmt: skip
    logged: dict[str, list[dict[str, float]]] = {"three": [], "last": []}
    results = {}
    for name, epochs in (("three", [1, 2, 4]), ("last", [4])):
        with_rungs = settings.model_copy(update={"fine_tuning_rungs": ptbxl.CooldownRungs(
            epochs=epochs, fraction=0.5, min_epochs=1)})  # fmt: skip
        results[name] = ptbxl.end_to_end(
            config, records, y, 256, with_rungs, torch.device("cpu"), None,
            logged[name].append, "f", scored_groups=groups,
        )  # fmt: skip

    def main(name: str) -> list[tuple[float, ...]]:
        keys = ("f/epoch", "f/epoch_train_loss", "f/epoch_train_eval_loss",
                "f/epoch_validation_loss", "f/epoch_validation_macro_auc")  # fmt: skip
        return [tuple(r[k] for k in keys) for r in logged[name] if "f/epoch" in r]

    # the branches at rungs 1 and 2 leave the main run exactly as it was (weights, optimizer,
    # scheduler, data order, random state: dropout and stochastic depth are on)
    assert main("three") == main("last") and len(main("three")) == 4
    trained = [r["f/train_loss"] for r in logged["three"] if "f/train_loss" in r]
    assert trained == [r["f/train_loss"] for r in logged["last"] if "f/train_loss" in r]
    rung_logs = [r for r in logged["three"] if "f/rung_epoch" in r]
    assert [(r["f/rung_epoch"], r["f/rung_cooldown_epochs"]) for r in rung_logs] == [
        (1.0, 1.0),
        (2.0, 1.0),
        (4.0, 2.0),
    ]
    # the last rung is the same branch in both runs
    last = [r for r in logged["last"] if "f/rung_epoch" in r]
    assert len(last) == 1 and last[0] == rung_logs[-1]
    scores, (p_val, p_test) = results["three"]
    for r in rung_logs:
        e = int(r["f/rung_epoch"])
        assert scores[f"rung_{e}/train_minus_validation_loss"] == pytest.approx(
            r["f/rung_train_eval_loss"] - r["f/rung_validation_loss"]
        )
        assert scores[f"rung_{e}/test_macro_auc"] == r["f/rung_test_macro_auc"]
        assert all(f"rung_{e}/validation_macro_auc_{g}" in scores for g in groups)
    best = max(rung_logs, key=lambda r: r["f/rung_validation_macro_auc"])  # fold 9 chooses
    assert scores["epoch"] == best["f/rung_epoch"]
    assert scores["macro_auc"] == best["f/rung_test_macro_auc"]
    assert ptbxl.macro_auc(y[20:], p_test)[0] == scores["macro_auc"]
    assert ptbxl.macro_auc(y[16:20], p_val)[0] == scores["validation_macro_auc"]
    assert not any("f/chosen_test_macro_auc" in r for r in logged["three"])  # no fold 10 per epoch
    # rungs need the constant schedule and end at the last epoch
    for update in ({"fine_tuning_schedule": "cosine"}, {"fine_tuning_epochs": 5},
                   {"fine_tuning_patience": 2}):  # fmt: skip
        with pytest.raises(ValueError, match="rung"):
            ptbxl.PtbxlSettings.model_validate(
                settings.model_dump() | {"fine_tuning_rungs": rungs.model_dump()} | update
            )


def test_per_label_scores_give_each_label_its_auc_and_positives_per_fold() -> None:
    y_val = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    p_val = np.array([[0.9, 0.1], [0.2, 0.3], [0.4, 0.2], [0.6, 0.4]])
    y_test = np.array([[1, 1], [0, 0], [0, 1]])
    p_test = np.array([[0.3, 0.8], [0.3, 0.1], [0.1, 0.5]])
    out = ptbxl.per_label(["A", "B"], y_val, p_val, y_test, p_test)
    assert out == {
        "label/A/validation_positives": 2.0, "label/A/validation_auc": 0.75,
        "label/A/test_positives": 1.0, "label/A/test_auc": 0.75,  # a tie: half a pair
        "label/B/validation_positives": 0.0,  # one class: no AUC
        "label/B/test_positives": 2.0, "label/B/test_auc": 1.0,
    }  # fmt: skip
    # the macro-AUC is the mean of these AUCs
    assert ptbxl.macro_auc(y_test, p_test)[0] == pytest.approx((0.75 + 1.0) / 2)


def test_drop_path_drops_whole_items_in_training_only() -> None:
    from biosignal_model.model import _drop_path  # noqa: PLC0415

    torch.manual_seed(0)
    y = torch.ones(2000, 3, 5)
    out = _drop_path(y, 0.25, training=True)
    per_item = out.flatten(1)
    assert all(v == 0.0 or v == pytest.approx(1 / 0.75) for v in per_item.unique().tolist())
    assert torch.equal(per_item.min(1).values, per_item.max(1).values)  # whole items
    assert 0.2 < (per_item[:, 0] == 0).float().mean().item() < 0.3
    assert torch.equal(_drop_path(y, 0.25, training=False), y)


def test_every_ptbxl_run_file_is_valid() -> None:
    files = sorted((Path(__file__).parents[1] / "runs").glob("bench-ptbxl-*.yaml"))
    assert files
    for f in files:
        settings = ptbxl.PtbxlSettings.model_validate(BenchmarkRun.load(f).settings)
        if f.name.startswith("bench-ptbxl-rungs-"):
            assert settings.fine_tuning_rungs is not None, f.name


def test_a_label_fraction_keeps_whole_patients_and_the_same_ones_for_a_seed() -> None:
    codes = [["NORM"]] * 40
    records = _records(codes, [1] * 40)
    records.examples[:] = [  # two records a patient
        Example(e.signals, Origin(e.origin.dataset_id, e.origin.recording_id, f"p{i // 2}",
                                  0.0, e.origin.channels))
        for i, e in enumerate(records.examples)
    ]  # fmt: skip
    train = list(range(40))
    assert ptbxl._labelled(records, train, 1.0, 0) == train
    tenth = ptbxl._labelled(records, train, 0.1, 0)
    patients = {records.examples[i].origin.subject_id for i in tenth}
    assert len(patients) == 2 and len(tenth) == 4  # 10 % of 20 patients, both their records
    assert tenth == ptbxl._labelled(records, train, 0.1, 0)
    assert tenth != ptbxl._labelled(records, train, 0.1, 1)

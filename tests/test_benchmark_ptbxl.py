from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.benchmarks import ptbxl  # noqa: E402
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
        methods=["fine_tuning", "from_scratch"],
        linear_probing_l2=[1e-3, 1.0], linear_probing_steps=50,
        fine_tuning_epochs=1, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0,
        fine_tuning_examples_per_step=8, fine_tuning_discriminative_factor=0.1,
        fine_tuning_lower_blocks_fraction=0.25, fine_tuning_crop_s=None, seed=0,
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
    )  # fmt: skip
    codes = [["NORM"], ["IMI"]] * 12
    records = _records(codes, [1] * 16 + [9] * 4 + [10] * 4)
    y, _, _ = ptbxl.labels(records, STATEMENTS, "diagnostic")
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["diagnostic"], rows=2, read_threads=1, workers=0,
        linear_probing=True, fixed_slots=False,
        methods=["fine_tuning", "from_scratch"],
        linear_probing_l2=[1.0], linear_probing_steps=10,
        fine_tuning_epochs=2, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0,
        fine_tuning_examples_per_step=4, fine_tuning_discriminative_factor=0.1,
        fine_tuning_lower_blocks_fraction=0.25, fine_tuning_crop_s=2.5, seed=0,
    )  # fmt: skip
    device = torch.device("cpu")
    logged: list[dict[str, float]] = []
    for start in (None, Encoder(config).state_dict()):
        logged.clear()
        scores = ptbxl.end_to_end(
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
    scores = ptbxl.end_to_end(config, records, y, 256, patient, device, None, logged.append, "f")
    assert scores["epochs_run"] == 3.0 and scores["epoch"] == 1.0 and scores["macro_auc"] == 0.5
    assert [r["f/epoch"] for r in logged if "f/epoch" in r] == [1.0, 2.0, 3.0] and aucs == []
    monkeypatch.undo()
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
    )  # fmt: skip
    model, head = Encoder(config), torch.nn.Linear(32, 3)
    groups = ptbxl.parameter_groups(model, head, 1e-3, 0.1, 0.25)
    assert [g["lr"] for g in groups] == pytest.approx([1e-3, 1e-4, 1e-5])
    ids = [{id(p) for p in g["params"]} for g in groups]
    used = {id(p) for p in head.parameters()} | {
        id(p) for n, p in model.named_parameters() if not n.startswith("head.")
    }
    assert set.union(*ids) == used and sum(map(len, ids)) == len(used)  # each once
    lower = {id(p) for b in model.blocks[:2] for p in b.parameters()} | {id(model.hidden)}
    assert lower <= ids[2] and not lower & ids[1]  # 2 of 8 blocks, with the embedding


def test_crops_cover_a_record_without_overlap() -> None:
    example = _records([["NORM"]], [1]).examples[0]  # 10 s at 100 Hz
    crops = ptbxl._crops(example, 2.5)
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
    )  # fmt: skip


def test_a_trimmed_batch_gives_each_example_the_same_features() -> None:
    config = _small()
    model = Encoder(config).eval()
    examples = _records([["NORM"]] * 5, [1] * 5).examples
    packer = ptbxl._packer(config, 512, 4)  # 40 tokens per example
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
        loop = torch.stack([model.features(x)[r][x.document[r] == d].mean(0) for r, d in where])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a, loop, rtol=1e-5, atol=1e-5)


def test_batches_made_in_workers_are_the_same_as_in_the_main_process() -> None:
    import random  # noqa: PLC0415

    config = _small()
    examples = _records([["NORM"]] * 12, [1] * 12).examples
    groups = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

    def crop(example: Example, rng: random.Random) -> Example:
        return ptbxl._random_crop(example, 2.5, rng)

    made = {}
    for workers in (0, 2):
        packer = ptbxl._packer(config, 256, 2)
        made[workers] = list(ptbxl._packed(examples, groups, crop, packer, "s/1", workers))
    assert len(made[0]) == len(made[2]) == 3
    for (a, wa), (b, wb) in zip(made[0], made[2], strict=True):
        assert wa == wb
        for name in ("values", "observed", "time_s", "slot", "document"):
            torch.testing.assert_close(getattr(a.inputs, name), getattr(b.inputs, name))

"""The v4 model and its latent objective (biosignal_model.v4), on the CPU with dense attention:
nothing hidden reaches the student or the decoder, packing does not change an example's
output, the teacher is an EMA without gradients, and the objective can be learned."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

F = torch.nn.functional

from biosignal_data.storage.stores import LocalStore, StoreFactory  # noqa: E402
from biosignal_data.training.loader import Channel, Example, Origin, Signal  # noqa: E402
from biosignal_model.channel_batch import (  # noqa: E402
    ChannelBatch,
    ChannelPacker,
    MaskSpec,
    cycle_period,
)
from biosignal_model.train import TrainConfig, train  # noqa: E402
from biosignal_model.v4 import Data2Vec, V4Spec, layout, per_channel_norm  # noqa: E402
from tests.test_model import _short_config  # noqa: E402

TOKEN_S = 0.04
MASK = MaskSpec(
    ratio=0.6, cycles=0.0, time_spans=1.0, channel_groups=0.0, single_tokens=0.0,
    time_span_s=(0.2, 1.0), group_span_s=(0.2, 1.0),
    cycle_period_s=(0.3, 2.0), cycle_tooth=(0.1, 0.25), cycle_strength=0.3,
)  # fmt: skip
SPEC = V4Spec.model_validate({
    "format": "v4",
    "local": {"kind": "s4d", "width": 16, "layers": 2, "states": 4, "min_time_s": 0.001,
              "max_time_s": 1.0, "kernel_time_constants": 5.0,
              "discretization": "zoh_centred", "masked_modes": "static_gain",
              "mixing_layers": [0], "mixing_heads": 2, "dropout": 0.0, "checkpoint": False},
    "global": {"width": 32, "layers": 3, "heads": 4, "feed_forward": 64, "pool_heads": 4,
               "max_channels": 8, "min_wavelength_s": 0.01, "max_wavelength_s": 100.0,
               "attention": "dense", "dropout": 0.0, "drop_path": 0.0},
    "objective": {"kind": "data2vec", "top_k": 2, "decoder_width": 16, "decoder_layers": 1,
                  "decoder_heads": 2, "decoder_feed_forward": 32, "ema_start": 0.99,
                  "ema_end": 0.999, "ema_steps": 100},
})  # fmt: skip
RATES = [100.0, 250.0, 100.0]


def _example(seed: int, seconds: float = 3.0) -> Example:
    rng = np.random.default_rng(seed)
    signals, channels = [], []
    for i, rate in enumerate(RATES):
        t = np.arange(round(seconds * rate)) / rate
        v = (np.sin(2 * np.pi * (1 + i) * t) + 0.1 * rng.normal(size=t.size)).astype(np.float32)
        signals.append(Signal(v, rate, 0.0))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", f"r{seed}", "s", 0.0, tuple(channels)))


def _batch(examples: list[Example], rows: int = 1, seed: int = 0, max_tokens: int = 4000,
           mask: MaskSpec | None = MASK) -> ChannelBatch:  # fmt: skip
    packer = ChannelPacker(
        datasets=["d/x"], token_s=TOKEN_S, max_tokens=max_tokens, max_samples=100_000,
        max_channels=8, rows=rows, mask=mask, seed=seed,
    )  # fmt: skip
    return packer.pack(examples)[0]


def _model() -> Data2Vec:
    torch.manual_seed(0)
    return Data2Vec(SPEC)


def test_predictions_do_not_change_when_only_what_the_teacher_sees_changes() -> None:
    """The decoder's predictions read the masked input only: other values at the hidden
    samples (seen by the teacher alone) change the targets, not one bit of the predictions."""
    model = _model().eval()
    batch = _batch([_example(1)])
    hidden = batch.inputs.hidden
    assert hidden.any()
    changed = dataclasses.replace(
        batch, targets=dataclasses.replace(
            batch.targets,
            normalized=torch.where(hidden, torch.randn_like(batch.targets.normalized) * 5,
                                   batch.targets.normalized),
        ),
    )  # fmt: skip
    with torch.no_grad():
        assert torch.equal(model.predictions(batch.inputs), model.predictions(changed.inputs))
        assert not torch.allclose(model.targets(batch), model.targets(changed))


def test_the_student_sequence_holds_the_visible_tokens_only() -> None:
    model = _model().eval()
    batch = _batch([_example(2)])
    t = batch.inputs.tokens
    out, layers, where = model.student(batch.inputs, ~t.hidden)
    assert torch.equal(where.index, (~t.hidden).nonzero(as_tuple=True)[0])
    assert out.shape[0] == int((~t.hidden).sum()) and all(h.shape == out.shape for h in layers)
    assert (where.document >= 0).sum() == out.shape[0]  # the rest is padding


def test_an_example_gives_the_same_output_alone_or_packed_with_others() -> None:
    """Attention stays within one example, and nothing else couples examples in a row."""
    model = _model().eval()
    a, b = _example(3), _example(4)
    alone = _batch([a], mask=None)
    together = _batch([b, a], mask=None)  # a is second in the row
    # the packer draws the slots in order: give a the slots it has alone
    second_channels = together.inputs.channels.document == 1
    together.inputs.channels.slot[second_channels] = alone.inputs.channels.slot
    with torch.no_grad():
        out_alone = model.student(alone.inputs, torch.ones_like(alone.inputs.tokens.hidden))[0]
        t = together.inputs
        out_both = model.student(t, torch.ones_like(t.tokens.hidden))[0]
    second = t.channels.document[t.tokens.channel] == 1
    torch.testing.assert_close(out_both[second], out_alone, atol=1e-5, rtol=1e-5)


def test_the_teacher_gets_no_gradient_and_follows_the_student_by_its_decay() -> None:
    model = _model().train()
    loss, stats = model.loss(_batch([_example(5), _example(6)], rows=2))
    loss.backward()
    assert all(p.grad is None for p in model.teacher.parameters())
    grads = [p.grad for p in model.student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.mask.grad is not None and float(model.mask.grad.abs().sum()) > 0
    assert float(stats["target_std"]) > 0 and float(stats["hidden_tokens"]) > 0
    assert model.ema_decay(0) == 0.99 and model.ema_decay(50) == pytest.approx(0.9945)
    assert model.ema_decay(10_000) == 0.999
    with torch.no_grad():
        for p in model.student.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.teacher.parameters()]
    model.update_teacher(0)
    for t, s, b in zip(model.teacher.parameters(), model.student.parameters(), before,
                       strict=True):  # fmt: skip
        torch.testing.assert_close(t, 0.99 * b + 0.01 * s)


def test_no_part_of_a_target_follows_from_its_channel_alone() -> None:
    """The targets are instance-normalized per channel of each example: over a channel's
    tokens every feature has mean 0, so the channel's embedding (which the decoder gets)
    predicts nothing of them; per_channel_norm is torch's instance_norm (variance 1, up to its
    epsilon for features with little spread)."""
    model = _model().eval()
    batch = _batch([_example(12), _example(13)], rows=2)
    target = model.all_targets(batch)
    channel = batch.inputs.tokens.channel
    for k in range(batch.inputs.channels.row.numel()):
        mine = target[channel == k]
        torch.testing.assert_close(mine.mean(0), torch.zeros(mine.shape[1]), atol=1e-4, rtol=0)
    h = torch.randn(30, 8)
    one = torch.zeros(30, dtype=torch.int64)
    expected = F.instance_norm(h.T[None]).squeeze(0).T
    torch.testing.assert_close(per_channel_norm(h, one, 1), expected, atol=1e-5, rtol=1e-5)


def test_the_objective_runs_under_bf16_autocast() -> None:
    """As on the GPU: linear maps give bf16 under autocast, the learned vectors stay float32."""
    model = _model().train()
    batch = _batch([_example(11)])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, stats = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and float(stats["target_std"]) > 0


def test_the_layout_puts_each_row_from_place_zero_and_pads_to_the_block_size() -> None:
    batch = _batch([_example(7), _example(8)], rows=2, max_tokens=300)  # 225 tokens each
    x = batch.inputs
    assert x.rows == 2
    select = ~x.tokens.hidden
    where = layout(x.tokens, x.channels, x.rows, select)
    assert where.length % 128 == 0
    for r in range(2):
        n = int((select & (x.tokens.row == r)).sum())
        assert (where.document[r, :n] >= 0).all() and (where.document[r, n:] == -1).all()
    values = torch.arange(where.index.numel(), dtype=torch.float32)[:, None]
    assert torch.equal(where.gather(where.scatter(values)), values)


def test_the_objective_can_be_learned_on_one_batch() -> None:
    """A sanity check of the wiring: on one fixed batch with the teacher fixed (decay 1), the
    loss falls clearly within a few dozen steps (to ~66 %; the per-channel normalized targets
    hold nothing the channel embedding alone gives, so they fall less than before)."""
    spec = SPEC.model_copy(update={"objective": SPEC.objective.model_copy(
        update={"ema_start": 1.0, "ema_end": 1.0})})  # fmt: skip
    torch.manual_seed(0)
    model = Data2Vec(spec).train()
    batch = _batch([_example(9), _example(10)], rows=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    for step in range(60):
        loss, _ = model.loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_teacher(step)
        losses.append(loss.item())
    assert losses[-1] < 0.8 * losses[0]


def test_a_v4_run_trains_validates_and_checkpoints_with_its_teacher(
    store: LocalStore, tmp_path: Path
) -> None:
    """train() end to end on the CPU with the v4 model and the latent objective: 10 steps on
    the test index, validation and checkpoints; the checkpoint holds the EMA teacher."""
    config = _short_config(store, tmp_path).model_dump(mode="json")
    config["data"]["spans"] = "per_group"
    config["tokens"] = {"format": "channels", "token_s": TOKEN_S, "max_tokens": 600,
                        "max_samples": 4000, "max_channels": 8, "rows": 2,
                        "mask": MASK.model_dump(mode="json")}  # fmt: skip
    config["model"] = SPEC.model_dump(mode="json", by_alias=True)
    last = train(TrainConfig.model_validate(config), StoreFactory(str(store.root), None), "abc")
    assert np.isfinite(last["train/loss"]) and last["train/target_std"] > 0
    assert "validation/loss" in last and np.isfinite(last["validation/loss"])
    saved = torch.load(store._path("models/t/step-00000010.pt"), weights_only=True)
    assert any(k.startswith("teacher.") for k in saved["model"])
    # a benchmark reads the run's config back from the checkpoint (TrainConfig.from_checkpoint)
    assert TrainConfig.from_checkpoint(saved["config"]) == TrainConfig.model_validate(config)
    assert "global" in saved["config"]["model"]  # written under the config's name
    first_runs = saved["config"] | {
        "model": {
            ("global_" if k == "global" else k): v for k, v in saved["config"]["model"].items()
        }
    }
    assert TrainConfig.from_checkpoint(first_runs) == TrainConfig.model_validate(config)


def test_ptbxl_fine_tunes_a_v4_model_from_its_checkpoint_weights() -> None:
    """The PTB-XL fine-tuning path with a v4 model: the student's weights from a Data2Vec
    state, channel batches, token means per record, layer-wise rates; it runs and scores."""
    from biosignal_model.benchmarks import ptbxl  # noqa: PLC0415
    from biosignal_model.channel_batch import ChannelTokens  # noqa: PLC0415
    from biosignal_model.v4 import V4Config, V4Task  # noqa: PLC0415
    from tests.test_benchmark_ptbxl import STATEMENTS, _records  # noqa: PLC0415

    tokens = ChannelTokens(format="channels", token_s=TOKEN_S, max_tokens=600,
                           max_samples=4000, max_channels=8, rows=4, mask=None)  # fmt: skip
    config = V4Config(SPEC, tokens)
    codes = [["NORM", "SR"], ["IMI", "LVOLT"], ["NORM", "LVOLT"], ["IMI", "SR"]] * 6
    records = _records(codes, [1] * 16 + [9] * 4 + [10] * 4)
    y, classes, _ = ptbxl.labels(records, STATEMENTS, "all")
    state = ptbxl._pretrained_state(config, {"model": _model().state_dict()})
    student = V4Task(config)
    student.load_state_dict(state)  # every student weight, nothing else
    groups = student.layer_groups()
    grouped = groups.input + groups.channel + [p for g in groups.layers for p in g] + groups.output
    assert {id(p) for p in grouped} == {id(p) for p in student.parameters()}
    settings = ptbxl.PtbxlSettings(
        rate=100, tasks=["all"], rows=4, read_threads=1, workers=0, linear_probing=False,
        fixed_slots=False, methods=["fine_tuning"], reference=None, reference_s4=None,
        reference_s4_channels=None, reference_s4_incontext=None,
        linear_probing_l2=[1.0], linear_probing_steps=10,
        fine_tuning_epochs=2, fine_tuning_patience=None, fine_tuning_learning_rate=1e-3,
        fine_tuning_weight_decay=0.0, fine_tuning_examples_per_step=4,
        fine_tuning_discriminative_factor=0.1, fine_tuning_lower_blocks_fraction=0.25,
        fine_tuning_crop_s=2.5, evaluation_stride_fraction=1.0, evaluation_aggregate="mean",
        fine_tuning_slot_group="lower", fine_tuning_betas=(0.9, 0.999),
        fine_tuning_warmup_epochs=0.0, fine_tuning_schedule="constant",
        fine_tuning_grad_clip=None, fine_tuning_head="linear",
        from_scratch_samples_per_token=None, from_scratch_tokenizer=None, from_scratch_model=None,
        fine_tuning_rungs=None, fine_tuning_label_fraction=1.0, seed=0,
    )  # fmt: skip
    scores, (p_val, p_test) = ptbxl.end_to_end(
        config, records, y, 600, settings, torch.device("cpu"), state, lambda m: None, "f"
    )
    assert p_val.shape == p_test.shape == (4, len(classes))
    assert 0.0 <= scores["macro_auc"] <= 1.0 and scores["epoch"] in (1.0, 2.0)


# --- leakage: nothing but the visible signal tells the predictions anything ----------------


def _content_free_features(batch: ChannelBatch) -> torch.Tensor:
    """Per token, everything a predictor could know without reading any sample: its channel
    (one-hot), slot, type, level and scale, rate, its time (raw and as sines at 0.04-10 s),
    its place in its channel and its distance to the ends of its example."""
    x = batch.inputs
    t, c = x.tokens, x.channels
    ch = t.channel
    n_ch = c.row.numel()
    parts = [F.one_hot(ch, n_ch).double(), F.one_hot(c.slot[ch], 8).double(),
             c.level[ch, None].double(), c.log_scale[ch, None].double(),
             c.rate_hz[ch, None].double() / 500]  # fmt: skip
    time = t.centre.double()
    for wavelength in (0.04, 0.1, 0.3, 1.0, 3.0, 10.0):
        parts += [torch.sin(2 * torch.pi * time / wavelength)[:, None],
                  torch.cos(2 * torch.pi * time / wavelength)[:, None]]  # fmt: skip
    first = torch.full((n_ch,), torch.inf, dtype=torch.float64).scatter_reduce(
        0, ch, time, reduce="amin"
    )
    last = torch.full((n_ch,), -torch.inf, dtype=torch.float64).scatter_reduce(
        0, ch, time, reduce="amax"
    )
    parts += [time[:, None], (time - first[ch])[:, None], (last[ch] - time)[:, None]]
    return torch.cat([*parts, torch.ones(len(ch), 1, dtype=torch.float64)], dim=1)


def test_a_predictor_that_reads_no_signal_explains_almost_nothing_of_the_targets() -> None:
    """Fit a ridge regression from everything but the samples (channel, slot, level, scale,
    rate, time, place, distance to the ends) to the targets on half of the tokens; on the
    other half it must explain almost none of their variance. With the first version of the
    targets (a layer norm per token) such a predictor explained 88.6 %; now 4.4 %."""
    model = _model().eval()
    batch = _batch([_example(s, seconds=4.0) for s in range(20, 26)], rows=6)
    target = model.all_targets(batch).double()
    features = _content_free_features(batch)
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(0))
    fit, held = order[: len(order) // 2], order[len(order) // 2 :]
    a = features[fit]
    w = torch.linalg.solve(a.T @ a + 1e-3 * torch.eye(a.shape[1], dtype=torch.float64),
                           a.T @ target[fit])  # fmt: skip
    residual = target[held] - features[held] @ w
    explained = 1 - residual.pow(2).sum() / (target[held] - target[held].mean(0)).pow(2).sum()
    assert float(explained) < 0.05


COMB_MASK = MASK.model_copy(update={"cycles": 0.5, "time_spans": 0.5})


def _layout_features(batch: ChannelBatch, reach: int = 30) -> torch.Tensor:
    """Per token, the hidden flags of its channel's tokens from `reach` cells before to
    `reach` after it (0.5 past the ends): all a predictor could read of a mask's layout,
    which with combs shows the period."""
    t = batch.inputs.tokens
    out = torch.full((len(t.channel), 2 * reach + 1), 0.5, dtype=torch.float64)
    for ch in t.channel.unique().tolist():
        idx = (t.channel == ch).nonzero(as_tuple=True)[0]
        idx = idx[torch.argsort(t.centre[idx])]
        h = t.hidden[idx].double()
        for j, off in enumerate(range(-reach, reach + 1)):
            src = torch.arange(len(idx)) + off
            ok = (src >= 0) & (src < len(idx))
            out[idx[ok], j] = h[src[ok]]
    return out


def _beat_example(seed: int, seconds: float = 8.0) -> Example:
    """QRS-like spikes at a random period (0.6-1.0 s) and phase, the same beats on every
    channel (amplitudes differ), with noise: a signal with a clear period, whose content the
    time alone does not tell."""
    rng = np.random.default_rng(seed)
    period, shift = rng.uniform(0.6, 1.0), rng.uniform(0.0, 1.0)
    signals, channels = [], []
    for i, rate in enumerate(RATES):
        t = np.arange(round(seconds * rate)) / rate
        v = 0.05 * rng.normal(size=t.size)
        for at in np.arange(shift, seconds, period):
            v += (1 + i) * np.exp(-0.5 * ((t - at) / 0.015) ** 2)
        signals.append(Signal(v.astype(np.float32), rate, 0.0))
        channels.append(Channel("a", i, rate, 1.0, 0.0, None, f"c{i}", "ecg", "mV"))
    return Example(tuple(signals), Origin("d/x", f"b{seed}", "s", 0.0, tuple(channels)))


def _explained(features: torch.Tensor, target: torch.Tensor) -> float:
    """Held-out share of the targets' variance a ridge regression from `features` explains
    (fitted on one half of the tokens, scored on the other)."""
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(0))
    fit, held = order[: len(order) // 2], order[len(order) // 2 :]
    a = features[fit]
    w = torch.linalg.solve(a.T @ a + 1e-3 * torch.eye(a.shape[1], dtype=torch.float64),
                           a.T @ target[fit])  # fmt: skip
    residual = target[held] - features[held] @ w
    return float(1 - residual.pow(2).sum() / (target[held] - target[held].mean(0)).pow(2).sum())


def test_with_combs_the_mask_layout_tells_nothing_but_the_period() -> None:
    """Combs hide the same phase of every cycle, so their layout shows the period (as the
    visible cycles do). It must show nothing more of the content: from the content-free
    features and the layout around each token, a ridge regression explains no more of the
    targets with the example's own mask than with an independent mask of the same design
    (another seed: the same period, an unrelated phase). Measured 2026-11-04: own 0.084, others
    0.098-0.114; the layout of time spans adds nothing. (Content-free alone: the test above.)"""
    examples = [_beat_example(s) for s in range(40, 46)]
    m = COMB_MASK
    assert all(
        cycle_period(e.signals, m.cycle_period_s, m.cycle_strength, TOKEN_S) for e in examples
    )
    model = _model().eval()
    batch = _batch(examples, rows=6, mask=m, seed=0)
    target = model.all_targets(batch).double()
    free = _content_free_features(batch)
    own = _explained(torch.cat([free, _layout_features(batch)], dim=1), target)
    others = [
        _explained(torch.cat([free, _layout_features(_batch(examples, rows=6, mask=m, seed=s))],
                             dim=1), target)
        for s in (1, 2, 3)
    ]  # fmt: skip
    assert own <= float(np.mean(others)) + 0.02


def test_another_example_in_the_row_changes_nothing_of_this_ones_predictions() -> None:
    """Two examples in one row: any change to the second one's samples (hidden or visible)
    leaves the first one's predictions bit for bit (student and decoder attend within one
    example only)."""
    model = _model().eval()
    a, b = _example(30), _example(31)
    other = Example(
        tuple(Signal(np.random.default_rng(5).normal(size=s.values.size).astype(np.float32) * 9,
                     s.sampling_rate, s.start_s) for s in b.signals),
        b.origin,
    )  # fmt: skip
    first = _batch([a, b], seed=3)
    second = _batch([a, other], seed=3)
    assert torch.equal(first.inputs.tokens.hidden, second.inputs.tokens.hidden)
    t = first.inputs.tokens
    mine = (first.inputs.channels.document[t.channel] == 0)[t.hidden]
    with torch.no_grad():
        p1, p2 = model.predictions(first.inputs), model.predictions(second.inputs)
    assert mine.any() and not torch.equal(p1[~mine], p2[~mine])
    torch.testing.assert_close(p1[mine], p2[mine], atol=1e-6, rtol=1e-6)


def test_rows_joined_for_a_step_keep_the_masks_on_the_samples() -> None:
    """The train loop joins single-row batches into a step (channel_batch.concat): after it,
    hidden samples are still 0 and not observed for the student, present for the teacher,
    and changing them changes no prediction."""
    from biosignal_model.channel_batch import concat  # noqa: PLC0415

    rows = [_batch([_example(40 + k)], seed=k) for k in range(3)]
    joined = concat(rows)
    x = joined.inputs
    assert x.hidden.any()
    assert (x.values[x.hidden] == 0).all() and not x.observed[x.hidden].any()
    teacher = joined.teacher_inputs()
    seen = x.hidden & joined.targets.observed
    assert torch.equal(teacher.values[seen], joined.targets.normalized[seen])
    assert not teacher.hidden.any()
    changed = dataclasses.replace(joined, targets=dataclasses.replace(
        joined.targets, normalized=torch.where(x.hidden, torch.full_like(x.values, 7.0),
                                               joined.targets.normalized)))  # fmt: skip
    model = _model().eval()
    with torch.no_grad():
        assert torch.equal(model.predictions(joined.inputs), model.predictions(changed.inputs))

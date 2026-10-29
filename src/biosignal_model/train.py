"""Masked-token pretraining: the run, its settings, its logs.

Every setting is in one YAML file (see `TrainConfig`); nothing has a default. The run
logs metrics to Weights & Biases (no artifacts: no checkpoints, data or code go there) and
writes checkpoints to the store under `run.checkpoint_prefix`.

On several GPUs, run it with torchrun (one process per GPU): `tokens.rows` stays the rows
of one step over all GPUs, so each GPU packs rows / GPUs of them, and the loss is the one
of the whole step. Only the first process logs, validates and writes checkpoints.
"""

from __future__ import annotations

import gc
import hashlib
import io
import itertools
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import yaml
from pydantic import BaseModel, ConfigDict
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from biosignal_data.index.build import ROOT as INDEX_ROOT
from biosignal_data.storage.stores import Store, StoreFactory
from biosignal_data.training import mixture
from biosignal_data.training.loader import ExampleSource, Selection
from biosignal_model.batch import (
    SIGNAL_TYPES,
    Analysis,
    Batch,
    Inputs,
    Packer,
    Tokenizer,
    concat,
)
from biosignal_model.model import Encoder, ModelConfig, masked_loss

log = logging.getLogger(__name__)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MixtureSection(_Section):
    """How much of each dataset an epoch reads (biosignal_data.training.mixture): UniMax
    over `clusters` of datasets, then over the datasets of each cluster, with `budget_tokens`
    for one epoch and at most `max_epochs` passes over any dataset."""

    rule: Literal["unimax"]
    budget_tokens: float
    max_epochs: float
    clusters: dict[str, list[str]]


class DataSection(_Section):
    store: str  # s3://bucket or a local path
    index: str  # the index version, e.g. 2026-10-20
    train_split: str
    validation_split: str
    signal_types: list[str] | None  # None: every channel
    datasets: list[str] | None  # None: every dataset of the index
    cache_directory: Path
    cache_bytes: int
    block_s: float  # the unit of work: a block of a recording; also the longest example
    shuffle_buffer: int
    reads_in_flight: int
    workers: int  # data loader processes, of each GPU's process
    # Each epoch is read in this many pieces, each by new data loader processes: a worker's
    # memory grows as it reads (v2-unimax: ~5 GB a worker after an hour, 22 workers, and the
    # pod's 147 GB ran out); fresh workers start small again. 1: one set for the epoch.
    pieces_per_epoch: int
    # None: every block once per epoch (the mix follows the data); the training split only
    mixture: MixtureSection | None


class TokenSection(_Section):
    max_tokens: int
    samples_per_token: int
    max_channels: int
    rows: int  # rows of max_tokens tokens in one step (over all GPUs)
    mask_ratio: float


class ModelSection(_Section):
    width: int
    layers: int
    heads: int
    feed_forward: int
    min_wavelength_s: float
    max_wavelength_s: float
    attention: Literal["flex", "dense"]
    tokenizer: Tokenizer  # how channels become tokens (docs/v3-tokenizer.md)


class OptimSection(_Section):
    learning_rate: float
    min_learning_rate: float  # the end of the decay
    warmup_steps: int
    steps: int
    # cosine: from the end of warmup to `steps`. wsd (warmup, stable, decay): constant after
    # warmup, then a decay over the last `decay_steps`; a run continued from a checkpoint
    # of the stable phase, with more `steps`, has no jump in its learning rate.
    schedule: Literal["cosine", "wsd"]
    decay_steps: int | None  # wsd only
    decay_shape: Literal["linear", "1-sqrt"] | None  # wsd only; 1-sqrt: Hagele et al. 2024
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float


class RunSection(_Section):
    name: str
    seed: int
    device: str  # cuda or cpu
    precision: Literal["bf16", "fp32"]
    log_every: int
    validate_every: int
    validation_batches: int
    checkpoint_every: int
    checkpoint_prefix: str  # in the store, e.g. models/<name>
    wandb_project: str
    wandb_entity: str
    wandb_mode: Literal["online", "offline", "disabled"]
    # A step whose loss or gradient norm is not finite is not applied; its batch is saved
    # to bad_batch_directory (on the training machine). After max_bad_steps such steps in
    # a row the run stops with an error.
    max_bad_steps: int
    bad_batch_directory: Path
    # newest: continue from the newest checkpoint under checkpoint_prefix, if there is one
    # (model, optimizer, step, the same W&B run; the data in a new order); never: start anew.
    resume: Literal["newest", "never"]


class TrainConfig(_Section):
    data: DataSection
    tokens: TokenSection
    model: ModelSection
    optim: OptimSection
    run: RunSection

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    @classmethod
    def from_checkpoint(cls, config: dict[str, Any]) -> TrainConfig:
        """The config stored in a checkpoint. One written before a field existed gets the
        value that does what its run did (ADDED_LATER); a run's own config never does."""
        config = {section: dict(values) for section, values in config.items()}
        for (section, name), value in ADDED_LATER.items():
            config[section].setdefault(name, value)
        return cls.model_validate(config)

    def model_settings(self) -> ModelConfig:
        return ModelConfig(
            samples_per_token=self.tokens.samples_per_token,
            max_channels=self.tokens.max_channels,
            tokenizer=self.model.tokenizer,
            **self.model.model_dump(exclude={"tokenizer"}),
        )


# Fields added after checkpoints were written, with the value that means "as before":
# v1-pretrain (2026-10-27) read every block once per epoch with one set of workers.
ADDED_LATER: dict[tuple[str, str], Any] = {
    ("data", "mixture"): None,  # 44e5541
    ("data", "pieces_per_epoch"): 1,  # 7bc3284
    ("model", "tokenizer"): {"kind": "samples"},  # v3-tokenizer: v0-v2 cut 64-sample tokens
}


class BatchStream(IterableDataset[Batch]):
    """Batches of one split; each data loader worker of each rank packs its own part.
    `rank`, `ranks`: this process's part of the data (0, 1: all of it); `rows`: the rows
    of its batches."""

    def __init__(
        self,
        source: ExampleSource,
        config: TrainConfig,
        datasets: list[str],
        rank: int,
        ranks: int,
        rows: int,
    ):
        super().__init__()
        self.source = source
        self.config = config
        self.datasets = datasets
        self.rank, self.ranks, self.rows = rank, ranks, rows
        self.epoch = 0
        self.piece, self.pieces = 0, 1

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        part, parts = self.rank * workers + worker, self.ranks * workers
        t = self.config.tokens
        packer = Packer(
            datasets=self.datasets, max_tokens=t.max_tokens,
            samples_per_token=t.samples_per_token, max_channels=t.max_channels, rows=self.rows,
            mask_ratio=t.mask_ratio,
            seed=((self.config.run.seed * 1_000_003 + self.epoch) * 1_009 + part) * 101
            + self.piece, slots=None, tokenizer=self.config.model.tokenizer,
        )  # fmt: skip
        yield from packer.batches(
            self.source.examples(self.epoch, part, parts, self.piece, self.pieces)
        )


def _stream(
    config: TrainConfig, factory: StoreFactory, split: str, rank: int, ranks: int, rows: int
) -> BatchStream:
    """The batches of `split` for process `rank` of `ranks`: its part of the data, in
    batches of `rows` rows."""
    d, t = config.data, config.tokens
    store = factory()
    selection = Selection.load(
        store, d.index, split=split,
        signal_types=set(d.signal_types) if d.signal_types is not None else None,
        datasets=set(d.datasets) if d.datasets is not None else None,
    )  # fmt: skip
    if not selection.recordings:
        raise ValueError(f"index {d.index}: no recordings in split {split!r} for this selection")
    m = d.mixture if split == d.train_split else None
    passes = (
        mixture.passes(
            mixture.dataset_tokens(selection, t.samples_per_token), m.clusters,
            m.budget_tokens, m.max_epochs,
        )
        if m is not None
        else None
    )  # fmt: skip
    source = ExampleSource(
        selection, factory, cache_directory=d.cache_directory, cache_bytes=d.cache_bytes,
        max_tokens=t.max_tokens, samples_per_token=t.samples_per_token,
        max_channels=t.max_channels, block_s=d.block_s, shuffle_buffer=d.shuffle_buffer,
        reads_in_flight=d.reads_in_flight, seed=config.run.seed, passes=passes,
    )  # fmt: skip
    datasets = sorted({r.dataset_id for r in selection.recordings})
    return BatchStream(source, config, datasets, rank, ranks, rows)


def _loader(stream: BatchStream, workers: int, device: torch.device) -> DataLoader[Batch]:
    # The objects made so far (the selection: every recording and its block index) leave
    # the garbage collector's care: a collection in a forked worker would otherwise write to
    # all of them, and the worker would copy their memory pages (each held ~5 GB).
    gc.collect()
    gc.freeze()
    return DataLoader(
        stream, batch_size=None, num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=False, prefetch_factor=2 if workers else None,
    )  # fmt: skip


def learning_rate(step: int, o: OptimSection) -> float:
    """The learning rate of a step (counted from 0)."""
    if step < o.warmup_steps:
        return o.learning_rate * (step + 1) / o.warmup_steps
    span = o.learning_rate - o.min_learning_rate
    if o.schedule == "cosine":
        progress = min(1.0, (step - o.warmup_steps) / max(1, o.steps - o.warmup_steps))
        return o.min_learning_rate + span * 0.5 * (1 + math.cos(math.pi * progress))
    if o.decay_steps is None or o.decay_shape is None:
        raise ValueError("schedule wsd needs decay_steps and decay_shape")
    start = o.steps - o.decay_steps
    if step < start:
        return o.learning_rate
    progress = min(1.0, (step - start) / max(1, o.decay_steps))
    left = 1 - progress if o.decay_shape == "linear" else 1 - math.sqrt(progress)
    return o.min_learning_rate + span * left


class _Totals:
    """Squared error and count, split by dataset and by signal type, for reports."""

    def __init__(self, datasets: list[str], device: torch.device) -> None:
        self.datasets = datasets
        self.by_dataset = torch.zeros(len(datasets), 2, device=device)
        self.by_type = torch.zeros(len(SIGNAL_TYPES), 2, device=device)

    def add(self, analysis: Analysis, per_token: torch.Tensor) -> None:
        for index, totals in ((analysis.dataset, self.by_dataset),
                              (analysis.signal_type, self.by_type)):  # fmt: skip
            keep = index >= 0
            totals.index_add_(0, index[keep], per_token[keep])

    def report(self, prefix: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for names, totals, kind in ((self.datasets, self.by_dataset, "dataset"),
                                    (SIGNAL_TYPES, self.by_type, "type")):  # fmt: skip
            for name, (error, count) in zip(names, totals.tolist(), strict=True):
                if count > 0:
                    out[f"{prefix}/loss_by_{kind}/{name}"] = error / count
        return out


class _Window:
    """What happened since the last log line."""

    def __init__(self, datasets: list[str], device: torch.device) -> None:
        self.totals = _Totals(datasets, device)
        self.steps = 0
        self.loss = self.tokens = self.padding = self.hidden = 0.0
        self.samples = self.examples = self.wait = 0.0
        self.start = time.perf_counter()

    def add(self, batch: Batch, loss: float, per_token: torch.Tensor, waited: float) -> None:
        real = batch.inputs.document >= 0
        self.steps += 1
        self.loss += loss
        self.tokens += real.sum().item()
        self.padding += (~real).sum().item()
        self.hidden += batch.inputs.hidden.sum().item()
        self.samples += batch.samples
        self.examples += batch.examples
        self.wait += waited
        self.totals.add(batch.analysis.to(per_token.device), per_token)

    def report(self, ranks: int) -> dict[str, float]:
        """The window's metrics over all ranks (every rank calls it at the same step)."""
        seconds = time.perf_counter() - self.start
        if ranks > 1:
            sums = torch.tensor(
                [self.tokens, self.padding, self.hidden, self.samples, self.examples,
                 self.wait / ranks], device=self.totals.by_type.device, dtype=torch.float64,
            )  # fmt: skip
            dist.all_reduce(sums)
            self.tokens, self.padding, self.hidden, self.samples, self.examples, self.wait = (
                sums.tolist()
            )
            dist.all_reduce(self.totals.by_dataset)
            dist.all_reduce(self.totals.by_type)
        return {
            "train/loss": self.loss / self.steps,
            "speed/tokens_per_s": self.tokens / seconds,
            "speed/examples_per_s": self.examples / seconds,
            "speed/samples_per_s": self.samples / seconds,
            "speed/data_wait_fraction": self.wait / seconds,
            "batch/padding_fraction": self.padding / max(1.0, self.tokens + self.padding),
            "batch/hidden_fraction": self.hidden / max(1.0, self.tokens),
            "batch/examples": self.examples / self.steps,
        } | self.totals.report("train")


def _optimizer(model: Encoder, o: OptimSection, device: torch.device) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": o.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=o.learning_rate, betas=(o.beta1, o.beta2), fused=device.type == "cuda",
    )  # fmt: skip


def _wandb_run(
    run_cfg: RunSection,
    described: dict[str, Any],
    wandb_id: str | None,
    model: Encoder,
    index_manifest: bytes,
    train_stream: BatchStream,
    validation_stream: BatchStream,
) -> Any:
    """The W&B run: a new one, or the resumed one (no artifacts in either)."""
    import wandb  # noqa: PLC0415 - the `train` extra

    return wandb.init(
        project=run_cfg.wandb_project, entity=run_cfg.wandb_entity, name=run_cfg.name,
        mode=run_cfg.wandb_mode, save_code=False,
        id=wandb_id if run_cfg.wandb_mode != "disabled" else None,
        resume="must" if wandb_id and run_cfg.wandb_mode != "disabled" else None,
        config=described | {
            "parameters": sum(p.numel() for p in model.parameters()),
            "index_manifest": json.loads(index_manifest),
            "train_recordings": len(train_stream.source.selection.recordings),
            "passes": train_stream.source.passes,
            "validation_recordings": len(validation_stream.source.selection.recordings),
        },
    )  # fmt: skip


def train(config: TrainConfig, factory: StoreFactory, code_commit: str) -> dict[str, float]:
    """Run the pretraining; returns the last logged metrics."""
    run_cfg, o = config.run, config.optim
    rank, ranks, device = _process_group(run_cfg.device, config.tokens.rows)
    first = rank == 0
    torch.manual_seed(run_cfg.seed)
    store = factory()
    index_manifest = store.get_bytes(f"{INDEX_ROOT}/{config.data.index}/_manifest.json")
    if index_manifest is None:
        raise FileNotFoundError(f"index {config.data.index} has no manifest")
    train_stream, validation_stream, validation_batches = _data(config, factory, rank, ranks)
    model = Encoder(config.model_settings()).to(device)
    optimizer = _optimizer(model, o, device)
    described = config.model_dump(mode="json") | {"code_commit": code_commit, "gpus": ranks}
    start, wandb_id = _resume(store, run_cfg, model, optimizer, described, write=first)
    trained = _parallel(model, device, ranks)
    run = (
        _wandb_run(
            run_cfg, described, wandb_id, model, index_manifest, train_stream, validation_stream
        )
        if first
        else _Silent()
    )
    autocast = torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=run_cfg.precision == "bf16"
    )
    window = _Window(train_stream.datasets, device)
    last: dict[str, float] = {}
    # A resumed run numbers its epochs from its first step: the data in a new order.
    batches = _steps(
        _epochs(
            train_stream,
            config.data.workers,
            device,
            first_epoch=start,
            pieces=config.data.pieces_per_epoch,
        ),
        config.tokens.rows // ranks,
    )
    bad_in_a_row, bad_total = 0, 0
    for step in range(start + 1, o.steps + 1):
        before = time.perf_counter()
        batch = next(batches)
        x = batch.inputs.to(device)
        waited = time.perf_counter() - before
        rate = learning_rate(step - 1, o)
        loss, per_token, grad_norm = _forward_backward(
            trained, optimizer, x, autocast, rate, o, ranks
        )
        # The same decision on every rank: the loss is the step's, the gradients are shared.
        if not (torch.isfinite(loss) and torch.isfinite(grad_norm)):
            bad_in_a_row, bad_total = bad_in_a_row + 1, bad_total + 1
            _bad_step(run_cfg, step, batch, loss, grad_norm, bad_in_a_row, rank, ranks)
            run.log({"train/bad_steps": bad_total}, step=step)
            optimizer.zero_grad(set_to_none=True)
            continue
        bad_in_a_row = 0
        optimizer.step()
        window.add(batch, loss.item(), per_token, waited)
        if step % run_cfg.log_every == 0 or step == o.steps:
            last = window.report(ranks) | {"train/grad_norm": float(grad_norm),
                                      "train/learning_rate": rate} | _memory(device)  # fmt: skip
            run.log(last, step=step)
            window = _Window(train_stream.datasets, device)
        if first and (step % run_cfg.validate_every == 0 or step == o.steps):
            validation = _validate(
                model, validation_batches, validation_stream.datasets, device, autocast
            )
            run.log(validation, step=step)
            last |= validation
            window.start = time.perf_counter()  # validation time is not training time
        if first and (step % run_cfg.checkpoint_every == 0 or step == o.steps):
            _checkpoint(store, run_cfg.checkpoint_prefix, step, model, optimizer, config, run.id)
            window.start = time.perf_counter()
    _finish(run, ranks)
    return last


def _finish(run: Any, ranks: int) -> None:
    run.finish()
    if ranks > 1:
        dist.destroy_process_group()


def _data(
    config: TrainConfig, factory: StoreFactory, rank: int, ranks: int
) -> tuple[BatchStream, BatchStream, list[Batch]]:
    """This process's training batches, the validation stream, and the validation batches.
    Validation runs on the first process only, with whole steps: the same batches as on
    one GPU."""
    # Each data worker packs single rows; a step joins rows of several workers (_steps).
    train_stream = _stream(config, factory, config.data.train_split, rank, ranks, 1)
    validation_stream = _stream(
        config, factory, config.data.validation_split, 0, 1, config.tokens.rows
    )
    validation_batches = (
        _validation_batches(validation_stream, config, factory()) if rank == 0 else []
    )
    return train_stream, validation_stream, validation_batches


def _process_group(device: str, rows: int) -> tuple[int, int, torch.device]:
    """Under torchrun with several processes (WORLD_SIZE > 1): join the process group;
    returns this process's rank, the number of ranks and its device (its own GPU). Else
    (0, 1, the configured device). The step's `rows` must split evenly over the ranks."""
    ranks = int(os.environ.get("WORLD_SIZE", "1"))
    if rows % ranks:
        raise ValueError(f"tokens.rows {rows} cannot be split over {ranks} GPUs")
    if ranks == 1:
        return 0, 1, torch.device(device)
    if device == "cuda":
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        return dist.get_rank(), ranks, torch.device("cuda", local)
    dist.init_process_group("gloo", timeout=timedelta(minutes=30))  # CPU: for tests
    return dist.get_rank(), ranks, torch.device(device)


def _parallel(model: Encoder, device: torch.device, ranks: int) -> torch.nn.Module:
    """The model as trained: wrapped in DDP (gradients averaged over the ranks) when there
    are several ranks; the model itself otherwise."""
    if ranks == 1:
        return model
    return DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None
    )


class _Silent:
    """The W&B run of the processes other than the first: they log nothing."""

    id = ""

    def log(self, *_: Any, **__: Any) -> None:
        pass

    def finish(self) -> None:
        pass


def _forward_backward(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: Inputs,
    autocast: Any,
    rate: float,
    o: OptimSection,
    ranks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loss, per-token errors and the gradient norm (before clipping) of one step; the
    gradients are left for the optimizer, which the caller applies or not. With several
    ranks, the loss is that of the whole step (all ranks' errors over all ranks' count),
    as on one GPU with all the rows."""
    for group in optimizer.param_groups:
        group["lr"] = rate
    with autocast:
        predicted = model(x)
    loss, per_token = masked_loss(predicted, x)
    optimizer.zero_grad(set_to_none=True)
    if ranks == 1:
        loss.backward()  # type: ignore[no-untyped-call]
    else:
        count = per_token[..., 1].sum().clamp(min=1)
        step = torch.stack([loss.detach() * count, per_token[..., 1].sum()])
        dist.all_reduce(step)
        everywhere = step[1].clamp(min=1)
        # DDP averages the gradients over the ranks: times `ranks` gives their sum.
        (loss * count / everywhere * ranks).backward()  # type: ignore[no-untyped-call]
        loss = step[0] / everywhere
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
    return loss, per_token, grad_norm


def _memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"system/max_memory_gb": torch.cuda.max_memory_allocated() / 2**30}


def _epochs(
    stream: BatchStream, workers: int, device: torch.device, first_epoch: int, pieces: int
) -> Iterator[Batch]:
    """The batches of epoch after epoch, each epoch in `pieces` pieces, each piece read by
    new data loader processes."""
    stream.pieces = pieces
    for epoch in itertools.count(first_epoch):
        stream.epoch = epoch
        for piece in range(pieces):
            stream.piece = piece
            yield from _loader(stream, workers, device)


def _steps(rows: Iterator[Batch], per_step: int) -> Iterator[Batch]:
    """Batches of `per_step` consecutive single-row batches. The data loader takes its
    workers' outputs in turn, so the rows of one step come from different workers, each
    reading its own blocks: a step mixes per_step times as many recordings as a batch
    packed by one worker (v1-pretrain: ~16 blocks a GPU step)."""
    while True:
        yield concat(list(itertools.islice(rows, per_step)))


VALIDATION_ROOT = "models/validation-sets"


def _validation_batches(stream: BatchStream, config: TrainConfig, store: Store) -> list[Batch]:
    """The validation batches: every validation uses the very same batches (examples and
    hidden tokens), so its values compare, and no reading holds up the GPU. They are read
    once and kept in the store under a key of everything that makes them (index, split,
    selection, tokens, number, seed, block length), so a resumed run and every other run
    with the same settings use them too. (Read again, they would differ: with several data
    workers the order in which examples come is not fixed; v1-pretrain's two resumes drew
    other batches, and its validation loss jumped.)"""
    d, t = config.data, config.tokens
    made_of = {
        "index": d.index, "split": d.validation_split, "signal_types": d.signal_types,
        "datasets": d.datasets, "block_s": d.block_s, "tokens": t.model_dump(mode="json"),
        "batches": config.run.validation_batches, "seed": config.run.seed,
    }  # fmt: skip
    digest = hashlib.sha256(json.dumps(made_of, sort_keys=True).encode()).hexdigest()[:16]
    key = f"{VALIDATION_ROOT}/{d.index}/{digest}.pt"
    data = store.get_bytes(key)
    if data is not None:
        log.info("validation batches from %s", key)
        found: list[Batch] = torch.load(io.BytesIO(data), weights_only=False)
        return found
    stream.epoch = 0
    batches = list(
        itertools.islice(
            _loader(stream, config.data.workers, torch.device("cpu")),
            config.run.validation_batches,
        )
    )
    out = io.BytesIO()
    torch.save(batches, out)
    store.put_bytes(key, out.getvalue())
    store.put_bytes(f"{VALIDATION_ROOT}/{d.index}/{digest}.json", json.dumps(made_of).encode())
    log.info("validation batches written to %s", key)
    return batches


@torch.no_grad()
def _validate(
    model: Encoder,
    batches: list[Batch],
    datasets: list[str],
    device: torch.device,
    autocast: Any,
) -> dict[str, float]:
    model.eval()
    totals = _Totals(datasets, device)
    error, count = 0.0, 0.0
    for batch in batches:
        x = batch.inputs.to(device)
        with autocast:
            predicted = model(x)
        _, per_token = masked_loss(predicted, x)
        totals.add(batch.analysis.to(device), per_token)
        error += per_token[..., 0].sum().item()
        count += per_token[..., 1].sum().item()
    model.train()
    return {"validation/loss": error / max(1.0, count)} | totals.report("validation")


def _checkpoint(
    store: Store,
    prefix: str,
    step: int,
    model: Encoder,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    wandb_id: str,
) -> None:
    out = io.BytesIO()
    torch.save(
        {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "config": config.model_dump(mode="json"), "wandb_id": wandb_id},
        out,
    )  # fmt: skip
    store.put_bytes(f"{prefix}/step-{step:08d}.pt", out.getvalue())


def _newest_checkpoint(store: Store, prefix: str) -> dict[str, Any] | None:
    keys = sorted(k for k in store.list(f"{prefix}/") if k.rsplit("/", 1)[-1].startswith("step-"))
    if not keys:
        return None
    data = store.get_bytes(keys[-1])
    if data is None:
        raise FileNotFoundError(keys[-1])
    found: dict[str, Any] = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if "wandb_id" not in found:  # written before resume existed: a new W&B run continues it
        log.warning("%s has no W&B run id; the resumed run gets a new one", keys[-1])
        found["wandb_id"] = None
    return found


def _resume(
    store: Store,
    run_cfg: RunSection,
    model: Encoder,
    optimizer: torch.optim.Optimizer,
    described: dict[str, Any],
    *,
    write: bool,
) -> tuple[int, str | None]:
    """The step to continue after and the W&B run id: from the newest checkpoint when the
    run resumes and there is one; else (0, None), and the run's config is written (by the
    first process: `write`)."""
    resumed = (
        _newest_checkpoint(store, run_cfg.checkpoint_prefix) if run_cfg.resume == "newest" else None
    )
    if resumed is None:
        if write:
            store.put_bytes(
                f"{run_cfg.checkpoint_prefix}/config.json",
                json.dumps(described, indent=2).encode(),
            )
        return 0, None
    model.load_state_dict(resumed["model"])
    optimizer.load_state_dict(resumed["optimizer"])
    log.warning("resuming from step %d of W&B run %s", resumed["step"], resumed["wandb_id"])
    return int(resumed["step"]), resumed["wandb_id"]


def _bad_step(
    run_cfg: RunSection,
    step: int,
    batch: Batch,
    loss: torch.Tensor,
    grad_norm: torch.Tensor,
    in_a_row: int,
    rank: int,
    ranks: int,
) -> None:
    """A step that is not applied: save its batch (each rank its own); too many in a row
    stop the run."""
    name = f"bad-step-{step:08d}" + (f"-rank{rank}" if ranks > 1 else "")
    _save_bad_batch(
        run_cfg.bad_batch_directory, name, step, batch, loss.detach().item(), grad_norm.item()
    )
    if in_a_row >= run_cfg.max_bad_steps:
        raise FloatingPointError(
            f"{in_a_row} steps in a row with a non-finite loss or gradient "
            f"(last at step {step}); batches in {run_cfg.bad_batch_directory}"
        )


def _save_bad_batch(
    directory: Path, name: str, step: int, batch: Batch, loss: float, norm: float
) -> None:
    """The batch of a step that was not applied, for finding why (inputs and analysis)."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"step": step, "loss": loss, "grad_norm": norm, "inputs": vars(batch.inputs),
         "analysis": vars(batch.analysis)},
        directory / f"{name}.pt",
    )  # fmt: skip
    log.warning("step %d: loss %s, gradient norm %s; not applied, batch saved", step, loss, norm)

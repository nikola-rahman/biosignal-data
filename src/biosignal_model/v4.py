"""The v4 model and its pretraining objective (docs/architecture.md; the simple plan of
2026-11-03: change only what was measured as broken in v0-v2).

Encoder, for a ChannelBatch:

1. the local stage (biosignal_model.local_stage) reads every sample of every channel at its
   own rate and gives one feature vector per sample;
2. each global token (a cell of `token_s` seconds of one channel, biosignal_model.chunking)
   pools the features of its samples by attention (a learned score per head, a softmax over
   the token's samples), so a token can pick out a QRS or a spike;
3. each token gets the embedding of its channel: its signal type, its random slot (tells the
   channels of one example apart), and its level and scale (asinh of the mean, log of the
   standard deviation of the visible samples: the amplitude the value transform removed);
4. transformer blocks over the tokens of each row, attention within one example, rotary
   angles from the token centre in seconds (float64), so every rate has one time axis.

The student reads only the visible tokens (the hidden ones are left out of its sequence, as
in MAE): with the masks on the samples (biosignal_model.channel_batch), nothing of a hidden
time reaches it. The teacher is an exponential moving average (EMA) of the student and reads
every token of the unmasked input (`ChannelBatch.teacher_inputs`).

Objective (data2vec 2.0 type): the target of a token is the mean over the teacher's top
`top_k` blocks of each block's output, each layer-normalized (no affine) per token; a narrow
decoder (its own blocks, attention within one example) gets the student's visible tokens and,
for each hidden token, a learned mask vector plus its channel embedding, and predicts the
targets of the hidden tokens; the loss is their mean squared error. The teacher's decay rises
linearly from `ema_start` to `ema_end` over `ema_steps` steps.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from biosignal_model.channel_batch import (
    SIGNAL_TYPES,
    ChannelBatch,
    ChannelInputs,
    ChannelTable,
    ChannelTokens,
    TokenTable,
)
from biosignal_model.local_stage import LocalStageSpec, local_stage
from biosignal_model.model import Block, LayerGroups, _attention

LENGTH_MULTIPLE = 128  # a row's sequence is padded to a multiple of the flex block size


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GlobalSpec(_Spec):
    width: int
    layers: int
    heads: int
    feed_forward: int
    pool_heads: int  # heads of the attention pooling of a token's samples
    max_channels: int  # slots
    min_wavelength_s: float  # rotary wavelengths, log-spaced between these (s)
    max_wavelength_s: float
    attention: Literal["flex", "dense"]  # flex: block-sparse kernel (GPU); dense: a full mask
    dropout: float
    drop_path: float


class Data2VecSpec(_Spec):
    kind: Literal["data2vec"]
    top_k: int  # teacher blocks averaged into the target
    decoder_width: int
    decoder_layers: int
    decoder_heads: int
    decoder_feed_forward: int
    ema_start: float  # the teacher's decay at step 0
    ema_end: float  # ... from `ema_steps` on
    ema_steps: int


class V4Spec(_Spec):
    """`model` of a v4 train config (tokens.format channels)."""

    format: Literal["v4"]
    local: LocalStageSpec
    global_: GlobalSpec = Field(alias="global")
    objective: Data2VecSpec


@dataclass
class Layout:
    """Where the selected tokens sit in the rows' padded sequences [rows, length]."""

    index: torch.Tensor  # [N]: the selected tokens (positions in the token table)
    flat: torch.Tensor  # [N]: their places in the flattened [rows x length]
    document: torch.Tensor  # [rows, length]: the example of each place; -1 is padding
    centre: torch.Tensor  # [rows, length] float64: the time of each place (0: padding)
    rows: int
    length: int

    def scatter(self, values: torch.Tensor) -> torch.Tensor:
        """[N, width] -> [rows, length, width], 0 at padding."""
        out = values.new_zeros(self.rows * self.length, values.shape[-1])
        out[self.flat] = values
        return out.view(self.rows, self.length, -1)

    def gather(self, h: torch.Tensor) -> torch.Tensor:
        """[rows, length, width] -> [N, width]."""
        return h.reshape(self.rows * self.length, -1)[self.flat]


def layout(tokens: TokenTable, channels: ChannelTable, rows: int, select: torch.Tensor) -> Layout:
    """The rows of the tokens `select` [T] (bool) picks, in token-table order (which is row by
    row), each row's tokens from place 0, padded to a multiple of LENGTH_MULTIPLE."""
    index = (select).nonzero(as_tuple=True)[0]
    row = tokens.row[index]
    counts = torch.bincount(row, minlength=rows)
    starts = torch.cumsum(counts, 0) - counts
    place = torch.arange(index.numel(), device=index.device) - starts[row]
    longest = int(counts.max()) if index.numel() else 1
    length = -(-max(longest, 1) // LENGTH_MULTIPLE) * LENGTH_MULTIPLE
    flat = row * length + place
    document = torch.full((rows * length,), -1, dtype=torch.int64, device=index.device)
    document[flat] = channels.document[tokens.channel[index]]
    centre = torch.zeros(rows * length, dtype=torch.float64, device=index.device)
    centre[flat] = tokens.centre[index]
    return Layout(index, flat, document.view(rows, length), centre.view(rows, length),
                  rows, length)  # fmt: skip


def rotary(
    time_s: torch.Tensor, head_width: int, low_s: float, high_s: float, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) [rows, 1, length, head_width / 2] of tokens at `time_s` [rows, length]: the
    angles 2 pi t / wavelength in float64 (float32 is off by ~1e-3 rad at minutes), the
    wavelengths log-spaced in [low_s, high_s]."""
    half = head_width // 2
    wavelengths = torch.logspace(math.log10(low_s), math.log10(high_s), half,
                                 dtype=torch.float64, device=time_s.device)  # fmt: skip
    angles = time_s.to(torch.float64)[..., None] * (2 * math.pi / wavelengths)
    return angles.cos().to(dtype)[:, None], angles.sin().to(dtype)[:, None]


class TokenPool(nn.Module):
    """Attention pooling of the samples of each token: per head a learned score of each
    sample's feature, a softmax over the token's samples, the weighted mean of a linear map of
    the features. [S, local] -> [T, width]."""

    def __init__(self, local: int, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width is a multiple of pool_heads")
        self.heads = heads
        self.score = nn.Linear(local, heads)
        self.value = nn.Linear(local, width)

    def forward(self, features: torch.Tensor, token: torch.Tensor, tokens: int) -> torch.Tensor:
        s, h = features.shape[0], self.heads
        score = self.score(features).float()  # [S, H]
        at = token[:, None].expand(s, h)
        top = score.new_full((tokens, h), -math.inf).scatter_reduce(
            0, at, score, reduce="amax", include_self=True
        )
        weight = torch.exp(score - top[token])
        weight = weight / weight.new_zeros(tokens, h).index_add(0, token, weight)[token]
        value = self.value(features).view(s, h, -1)
        pooled: torch.Tensor = value.new_zeros(tokens, h, value.shape[-1]).index_add(
            0, token, weight.to(value.dtype)[..., None] * value
        )
        return pooled.view(tokens, -1)


class V4Encoder(nn.Module):
    def __init__(self, spec: V4Spec) -> None:
        super().__init__()
        g = spec.global_
        if g.width % g.heads or (g.width // g.heads) % 2:
            raise ValueError("width / heads is an even integer")
        self.spec = g
        self.local = local_stage(spec.local)
        self.pool = TokenPool(spec.local.width, g.width, g.pool_heads)
        self.signal_type = nn.Embedding(len(SIGNAL_TYPES), g.width)
        self.slot = nn.Embedding(g.max_channels, g.width)
        self.level_scale = nn.Linear(2, g.width)
        self.blocks = nn.ModuleList(
            Block(g.width, g.heads, g.feed_forward, g.dropout, g.drop_path) for _ in range(g.layers)
        )
        self.norm = nn.RMSNorm(g.width)

    def channel_embedding(self, c: ChannelTable) -> torch.Tensor:
        """[C, width]: signal type + slot + (level, log scale)."""
        level_scale = torch.stack([c.level, c.log_scale], dim=-1).to(self.level_scale.weight.dtype)
        out: torch.Tensor = (
            self.signal_type(c.signal_type) + self.slot(c.slot) + self.level_scale(level_scale)
        )
        return out

    def token_features(self, x: ChannelInputs) -> torch.Tensor:
        """[T, width]: the pooled local features of every token, plus its channel embedding."""
        t = x.tokens
        local = self.local(x)
        pooled: torch.Tensor = self.pool(local, t.sample_token, t.channel.numel())
        return pooled + self.channel_embedding(x.channels)[t.channel]

    def forward(
        self, x: ChannelInputs, select: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], Layout]:
        """The tokens `select` picks, through the blocks: the output [N, width] (after the
        final norm), each block's output [N, width], and their layout."""
        where = layout(x.tokens, x.channels, x.rows, select)
        h = where.scatter(self.token_features(x)[where.index])
        g = self.spec
        angles = rotary(where.centre, g.width // g.heads, g.min_wavelength_s,
                        g.max_wavelength_s, h.dtype)  # fmt: skip
        attend = _attention(where.document, g.attention)
        layers = []
        for block in self.blocks:
            h = block(h, angles, attend)
            layers.append(where.gather(h))
        return where.gather(self.norm(h)), layers, where


class Data2Vec(nn.Module):
    """The student encoder, its EMA teacher, and the decoder of the latent objective."""

    def __init__(self, spec: V4Spec) -> None:
        super().__init__()
        o, g = spec.objective, spec.global_
        if not 1 <= o.top_k <= g.layers:
            raise ValueError("top_k is in 1..global.layers")
        if o.decoder_width % o.decoder_heads or (o.decoder_width // o.decoder_heads) % 2:
            raise ValueError("decoder_width / decoder_heads is an even integer")
        self.spec = spec
        self.student = V4Encoder(spec)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.into_decoder = nn.Linear(g.width, o.decoder_width)
        self.channel_into_decoder = nn.Linear(g.width, o.decoder_width)
        self.mask = nn.Parameter(torch.zeros(o.decoder_width))
        self.decoder = nn.ModuleList(
            Block(o.decoder_width, o.decoder_heads, o.decoder_feed_forward, 0.0, 0.0)
            for _ in range(o.decoder_layers)
        )
        self.decoder_norm = nn.RMSNorm(o.decoder_width)
        self.predict = nn.Linear(o.decoder_width, g.width)

    @torch.no_grad()
    def targets(self, batch: ChannelBatch) -> torch.Tensor:
        """[hidden tokens, width]: the teacher's mean of its top_k blocks' layer-normalized
        outputs, over the unmasked input, at the hidden tokens."""
        self.teacher.eval()
        x = batch.teacher_inputs()
        everything = torch.ones_like(x.tokens.hidden)
        _, layers, _ = self.teacher(x, everything)
        top = layers[-self.spec.objective.top_k :]
        target = torch.stack([F.layer_norm(h.float(), h.shape[-1:]) for h in top]).mean(0)
        return target[batch.inputs.tokens.hidden]

    def predictions(self, x: ChannelInputs) -> torch.Tensor:
        """[hidden tokens, width]: the decoder's predictions from the student's visible
        tokens. Reads only `x` (the masked input)."""
        t = x.tokens
        visible = self.student(x, ~t.hidden)[0]
        where = layout(t, x.channels, x.rows, torch.ones_like(t.hidden))
        channel = self.channel_into_decoder(self.student.channel_embedding(x.channels))[t.channel]
        # a hidden token: the mask vector and its channel; a visible one: the student's output
        d = (self.mask + channel).index_copy(
            0, (~t.hidden).nonzero(as_tuple=True)[0], self.into_decoder(visible).to(channel.dtype)
        )
        h = where.scatter(d)
        o = self.spec.objective
        g = self.spec.global_
        angles = rotary(where.centre, o.decoder_width // o.decoder_heads, g.min_wavelength_s,
                        g.max_wavelength_s, h.dtype)  # fmt: skip
        attend = _attention(where.document, g.attention)
        for block in self.decoder:
            h = block(h, angles, attend)
        out: torch.Tensor = self.predict(self.decoder_norm(where.gather(h)))
        return out[t.hidden]

    def loss(self, batch: ChannelBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The mean squared error over the hidden tokens, and collapse monitors: the standard
        deviation over tokens (mean over features) of the targets and of the predictions."""
        target = self.targets(batch)
        predicted = self.predictions(batch.inputs).float()
        loss = F.mse_loss(predicted, target)
        stats = {
            "target_std": target.std(0).mean().detach(),
            "prediction_std": predicted.std(0).mean().detach(),
            "hidden_tokens": torch.tensor(float(target.shape[0])),
        }
        return loss, stats

    def forward(self, batch: ChannelBatch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The training step's loss (the module's call, so data-parallel training wraps it)."""
        return self.loss(batch)

    def ema_decay(self, step: int) -> float:
        o = self.spec.objective
        done = min(1.0, step / o.ema_steps) if o.ema_steps > 0 else 1.0
        return o.ema_start + (o.ema_end - o.ema_start) * done

    @torch.no_grad()
    def update_teacher(self, step: int) -> None:
        """teacher <- decay x teacher + (1 - decay) x student, weights and buffers."""
        decay = self.ema_decay(step)
        for kept, live in zip(self.teacher.parameters(), self.student.parameters(), strict=True):
            kept.lerp_(live.detach(), 1.0 - decay)
        for kept_buffer, live_buffer in zip(self.teacher.buffers(), self.student.buffers(),
                                            strict=True):  # fmt: skip
            kept_buffer.copy_(live_buffer)


@dataclass(frozen=True)
class V4Config:
    """What a downstream task needs of a v4 run: the model's spec and its batch format."""

    spec: V4Spec
    tokens: ChannelTokens

    @property
    def width(self) -> int:
        return self.spec.global_.width


class V4Task(V4Encoder):
    """The pretrained student as a task model: the features of every token, and the layer
    groups for layer-wise learning rates (benchmarks). Its weights are the student's."""

    def __init__(self, config: V4Config) -> None:
        super().__init__(config.spec)
        self.config = config

    def features(self, x: ChannelInputs) -> torch.Tensor:
        """[T, width]: every token of the batch (nothing hidden), in token-table order."""
        out: torch.Tensor = self(x, torch.ones_like(x.tokens.hidden))[0]
        return out

    def layer_groups(self) -> LayerGroups:
        inputs = [self.local, self.pool, self.signal_type, self.level_scale]
        return LayerGroups(
            input=[p for m in inputs for p in m.parameters()],
            channel=list(self.slot.parameters()),
            layers=[list(b.parameters()) for b in self.blocks],
            output=list(self.norm.parameters()),
        )

    @staticmethod
    def pretrained(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The student's weights from a v4 checkpoint's model state (Data2Vec)."""
        prefix = "student."
        return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}

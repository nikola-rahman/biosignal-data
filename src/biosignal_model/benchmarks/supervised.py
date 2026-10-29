"""Recipe parts for supervised training from scratch on PTB-XL (used by benchmarks/s4.py): losses
for a macro-AUC target, an exponential moving average of the weights, ECG augmentation, the
learning-rate schedule. Each part names its source; every setting is a YAML field without a
default (S4Settings).

- class-balanced BCE (Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of
  Samples", arXiv 1901.05555): the weight of a class is 1 / E_n with E_n = (1 - beta^n) /
  (1 - beta), n its training positives, normalized so the weights sum to the number of classes.
  The paper's setting for its large data sets (iNaturalist, ImageNet) is beta = 0.999. In the
  multi-label case each label's column of the BCE (its positive and negative terms) gets the
  label's weight.
- asymmetric loss (Ridnik et al. 2021, "Asymmetric Loss For Multi-Label Classification", arXiv
  2009.14119; github.com/Alibaba-MIIL/ASL src/loss_functions/losses.py AsymmetricLoss, with the
  settings of its train.py: gamma_neg 4, gamma_pos 0, clip 0.05, the focusing weight without
  gradient). The reference returns the sum over the batch; here the mean over its elements, the
  same AdamW steps (the update does not change with the scale of the loss, eps aside).
- the moving average (Polyak averaging as in Tarvainen & Valpola 2017 / timm ModelEmaV2):
  ema <- decay * ema + (1 - decay) * weights after each step; buffers copied.
- augmentation (training crops only, one draw per crop): see `Augmentation`.
- schedule: linear warmup (Goyal et al. 2017) then a cosine to zero at the last epoch
  (Loshchilov & Hutter 2017, SGDR without restarts), per step.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the usual name
from pydantic import BaseModel, ConfigDict, model_validator
from torch import nn


class AsymmetricLossSettings(BaseModel):
    """ASL (Ridnik et al. 2021): 4, 0, 0.05 in its train.py."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gamma_neg: float
    gamma_pos: float
    clip: float


class AugmentationSettings(BaseModel):
    """ECG augmentation of each training crop, in this order (sources in `Augmentation`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stretch: tuple[float, float]  # time factor range (heart-rate change); 1: none
    scale_std: float  # one amplitude factor per crop ~ N(1, scale_std), every lead
    wander_mv: float  # baseline wander amplitude per lead ~ U(0, wander_mv), mV
    wander_hz: tuple[float, float]  # its frequency (one per crop) ~ U(range), Hz
    noise_mv: float  # Gaussian noise standard deviation, mV
    lead_dropout: float  # each lead zeroed with this probability (at least one is kept)


class DistillationSettings(BaseModel):
    """Born-again self-distillation (Furlanello et al. 2018): the target is
    (1 - teacher_fraction) * labels + teacher_fraction * the teacher's probabilities on the same
    (augmented) crop; the teacher is a model of the same recipe, its chosen weights in a file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    teacher_weights: str  # local path, written by a run with `save_chosen_to`
    teacher_fraction: float


class RecipeSettings(BaseModel):
    """The parts of this module; none has a default. The published S4 recipe: loss "bce",
    class_balanced_beta None, asymmetric None, ema_decay None, augmentation None, schedule
    "constant", warmup_epochs 0, patience_epochs None, distillation None."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: Literal["bce", "class_balanced_bce", "asymmetric"]
    class_balanced_beta: float | None  # only with class_balanced_bce (0.999)
    asymmetric: AsymmetricLossSettings | None  # only with asymmetric
    ema_decay: float | None  # None: no moving average; else its weights are validated and tested
    augmentation: AugmentationSettings | None
    schedule: Literal["constant", "warmup_cosine"]
    warmup_epochs: float  # of warmup_cosine; 0 with constant
    patience_epochs: int | None  # stop after this many epochs without a fold-9 gain; None: never
    distillation: DistillationSettings | None
    save_chosen_to: str | None  # local path for the chosen epoch's (scored) weights

    @model_validator(mode="after")
    def _consistent(self) -> RecipeSettings:
        if (self.loss == "class_balanced_bce") != (self.class_balanced_beta is not None):
            raise ValueError("class_balanced_beta is set exactly with loss class_balanced_bce")
        if (self.loss == "asymmetric") != (self.asymmetric is not None):
            raise ValueError("asymmetric is set exactly with loss asymmetric")
        if self.schedule == "constant" and self.warmup_epochs != 0:
            raise ValueError("warmup_epochs is 0 with the constant schedule")
        return self


def class_balanced_weights(positives: np.ndarray, beta: float) -> np.ndarray:
    """Cui et al. 2019: 1 / E_n per class, E_n = (1 - beta^n) / (1 - beta), summing to C."""
    n = np.maximum(positives.astype(np.float64), 1.0)
    w = (1.0 - beta) / (1.0 - np.power(beta, n))
    return w * len(w) / w.sum()


def asymmetric_loss(
    logits: torch.Tensor, target: torch.Tensor, s: AsymmetricLossSettings, eps: float = 1e-8
) -> torch.Tensor:
    """AsymmetricLoss of the ASL repository (disable_torch_grad_focal_loss True), averaged."""
    p = torch.sigmoid(logits)
    p_neg = 1 - p
    if s.clip > 0:
        p_neg = (p_neg + s.clip).clamp(max=1)
    loss = target * torch.log(p.clamp(min=eps)) + (1 - target) * torch.log(p_neg.clamp(min=eps))
    if s.gamma_neg > 0 or s.gamma_pos > 0:
        with torch.no_grad():
            pt = p * target + p_neg * (1 - target)
            w = torch.pow(1 - pt, s.gamma_pos * target + s.gamma_neg * (1 - target))
        loss = loss * w
    return -loss.mean()


def loss_function(
    s: RecipeSettings, train_targets: np.ndarray, device: torch.device
) -> nn.Module | None:
    """None for the plain BCE path; else a callable (logits, target) -> loss."""
    if s.loss == "bce":
        return None
    if s.class_balanced_beta is not None:  # loss class_balanced_bce (checked in RecipeSettings)
        w = class_balanced_weights(train_targets.sum(0), s.class_balanced_beta)
        return _WeightedBce(torch.as_tensor(w, dtype=torch.float32, device=device))
    if s.asymmetric is None:
        raise ValueError(f"loss {s.loss} needs its settings")
    return _Asymmetric(s.asymmetric)


class _WeightedBce(nn.Module):
    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (bce * self.weights).mean()


class _Asymmetric(nn.Module):
    def __init__(self, s: AsymmetricLossSettings) -> None:
        super().__init__()
        self.s = s

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return asymmetric_loss(logits, target, self.s)


class Ema:
    """The moving average of a model's weights (its own copy, eval mode)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for e, p in zip(self.model.parameters(), model.parameters(), strict=True):
            e.lerp_(p.detach(), 1.0 - self.decay)
        for e, b in zip(self.model.buffers(), model.buffers(), strict=True):
            e.copy_(b)


def warmup_cosine(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    """The lr factor at a step: linear from 1/warmup to 1, then a cosine to 0 at total_steps."""

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return factor


class Augmentation:
    """Training crops of a batch of records, augmented (GPU; one draw per crop). Steps and
    sources; the ranges are global, the same for every label:

    1. time stretch by f ~ U(stretch): a window of round(f * size) samples resampled (linear) to
       `size`; a heart-rate change. Domain bound: QRS 80-120 ms and the rate classes (SBRAD < 60,
       STACH > 100 bpm) move by f too, so the range stays small (0.95-1.05: QRS +-5 ms).
    2. amplitude factor ~ N(1, scale_std), one per crop for all leads (Um et al. 2017, "Data
       augmentation of wearable sensor data", scaling sigma 0.1): one factor keeps the ratios
       between leads (electrical axis) intact.
    3. baseline wander: a sinusoid per lead, amplitude ~ U(0, wander_mv) mV, random phase, one
       frequency per crop ~ U(wander_hz) (respiration 10-30 breaths/min, 0.167-0.5 Hz: Raghu et
       al. 2022, "Data Augmentation for Electrocardiograms", arXiv 2204.04360; C_max 0.1:
       Mehari & Strodthoff 2022, arXiv 2103.12676).
    4. Gaussian noise, std noise_mv (0.01 mV: Mehari & Strodthoff 2022).
    5. lead dropout: each lead zeroed with probability lead_dropout, one lead always kept
       (random lead masking of Oh et al. 2022, arXiv 2203.06889, at a lower rate)."""

    def __init__(self, s: AugmentationSettings, rate: float) -> None:
        self.s, self.rate = s, rate

    def crops(
        self, records: torch.Tensor, size: int, generator: torch.Generator
    ) -> torch.Tensor:  # (B, leads, T) -> (B, leads, size)
        s, dev = self.s, records.device
        b, leads, length = records.shape

        def uniform(lo: float, hi: float, *shape: int) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(*shape, generator=generator, device=dev)

        f = uniform(s.stretch[0], s.stretch[1], b)
        window = torch.clamp(torch.round(f * size), 2, length)  # samples taken
        start = torch.floor(torch.rand(b, generator=generator, device=dev) * (length - window + 1))
        pos = start[:, None] + torch.arange(size, device=dev)[None] * (
            (window[:, None] - 1) / (size - 1)
        )  # (B, size), within [start, start + window - 1]
        lo = pos.floor().long().clamp(0, length - 1)
        hi = (lo + 1).clamp(max=length - 1)
        frac = (pos - lo).unsqueeze(1)
        x = torch.gather(records, 2, lo.unsqueeze(1).expand(b, leads, size)) * (1 - frac)
        x = x + torch.gather(records, 2, hi.unsqueeze(1).expand(b, leads, size)) * frac
        scale = 1 + s.scale_std * torch.randn(b, 1, 1, generator=generator, device=dev)
        x = x * scale
        hz = uniform(s.wander_hz[0], s.wander_hz[1], b, 1, 1)
        amp = uniform(0.0, s.wander_mv, b, leads, 1)
        phase = uniform(0.0, 2 * math.pi, b, leads, 1)
        t = torch.arange(size, device=dev).view(1, 1, size) / self.rate
        x = x + amp * torch.sin(2 * math.pi * hz * t + phase)
        x = x + s.noise_mv * torch.randn(x.shape, generator=generator, device=dev)
        keep = torch.rand(b, leads, generator=generator, device=dev) >= s.lead_dropout
        one = torch.randint(0, leads, (b,), generator=generator, device=dev)
        keep[torch.arange(b, device=dev), one] = True
        return x * keep.unsqueeze(-1)

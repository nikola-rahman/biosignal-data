"""The supervised S4 baseline of Mehari & Strodthoff ("Advancing the state-of-the-art for ECG
analysis through structured state space models", arXiv 2211.07579; 100 Hz, PTB-XL "all":
0.9417 +- 0.0035), ported without its CUDA/pykeops Cauchy kernel to check our benchmark pipeline
against the best published model trained from scratch. Sources, github.com/tmehari/ssm_ecg
(main, read 2026-10-29), each choice below names its file:

- command (README, "B1. Supervised Training"): train_ecg_model.py --dataset ptb_xl_fs100
  --label_class label_all --model s4; every other setting is a default of train_ecg_model.py
  parse_args: batch_size 128, epochs 100, lr 1e-3, input_size 250 (2.5 s), d_state 8,
  d_model 512, n_layers 4, s4_dropout 0.2, bn False, normalize False, test_folds [9, 10];
- model (train_ecg_model.py create_s4; dl_models/s4_model.py S4Model): Conv1d(12, 512, 1)
  encoder, 4 residual blocks [S4 -> channel dropout 0.2 -> + input -> LayerNorm] (post-norm,
  prenorm False), mean over time, Linear(512, classes); S4(bidirectional=True, postact "glu",
  dropout 0.2, transposed, d_state 8, l_max 250);
- the S4 layer (dl_models/s4.py, the "standalone" S4 of HazyResearch/state-spaces): HiPPO-LegS
  NPLR kernel, rank 1, N = 8 (4 conjugate pairs), log dt uniform in [log 1e-3, log 0.1] per
  feature, A, B, dt and P not trained (trainable=None: buffers), C (2 channels: the two
  directions) and D trained, "length correction" of C at L = 250, the kernel computed at the
  roots of unity through the bilinear transform and a Woodbury correction, then an FFT
  convolution of length 2L; GELU, channel dropout, a 1x1 linear to 2 x 512 and GLU;
- data (clinical_ts/ecg_dataset_wrapper.py): folds 1-8 train, 9 validation, 10 test; no input
  normalization (normalize False); training: one random crop of 250 samples per record and
  epoch, start randint(0, T - 251) (timeseries_utils.py TimeseriesDatasetCrops), shuffled,
  the last partial batch dropped; validation and test: crops of 250 every 250 samples;
- training (train_ecg_model.py ECGLightningModel): AdamW on every weight, lr 1e-3 constant,
  weight decay 1e-3 (its default wd), torch defaults betas (0.9, 0.999), eps 1e-8; binary
  cross-entropy on logits; fp32;
- model selection: the epoch with the best validation macro-AUC of the crops' logits averaged
  per record (ModelCheckpoint monitor "val/val_macro_agg"); its test score is the macro-AUC of
  the crops' sigmoid probabilities averaged per record ("test_macro_agg_sig", what the script
  saves as test_scores).

One deviation in the data: their PTB-XL 1.0.1 records are resampled from the 500 Hz files to
100 Hz (ecg_utils.py prepare_data_ptb_xl); ours are the 100 Hz records of PTB-XL 1.0.3. Their
nn.Dropout2d on (batch, features, time) zeroes whole features of a sample (PyTorch 1.10), which
is what `_channel_dropout` does.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from biosignal_model.benchmarks.supervised import (
    Augmentation,
    Ema,
    RecipeSettings,
    loss_function,
    warmup_cosine,
)
from biosignal_model.benchmarks.xresnet1d import crop_starts


class S4Settings(RecipeSettings):
    """The recipe of reference_s4; none has a default (see the module docstring). The fields of
    RecipeSettings (benchmarks/supervised.py: loss, moving average, augmentation, schedule,
    stop, distillation) take the published recipe's values in its config."""

    input_size_s: float  # 2.5 (input_size 250 at 100 Hz)
    valid_stride_fraction: float  # 1.0: validation/test crops every input_size
    validation_on: Literal["logits", "probabilities"]  # what is averaged per record to choose
    test_on: Literal["logits", "probabilities"]  # what is averaged per record to score
    batch_size: int  # 128
    epochs: int  # 100
    learning_rate: float  # 1e-3, constant
    weight_decay: float  # 1e-3
    betas: tuple[float, float]  # (0.9, 0.999)
    eps: float  # 1e-8
    d_model: int  # 512
    d_state: int  # 8
    n_layers: int  # 4
    dropout: float  # 0.2
    dt_min: float  # 1e-3
    dt_max: float  # 0.1


class S4ChannelsSettings(S4Settings):
    """reference_s4_channels: the recipe of reference_s4 (every field above) and the settings of
    the channel-set variant (see S4ChannelModel); none has a default."""

    heads: int  # of the attention across channels after each S4 block
    base_rate_hz: float  # the rate at which the S4 time steps are the published ones (100)
    identity_std: float  # initial standard deviation of the lead-identity embedding


# --- HiPPO-LegS NPLR (dl_models/s4.py transition, rank_correction, nplr) -----------------------


def _legs(n: int) -> tuple[np.ndarray, np.ndarray]:
    q = np.arange(n, dtype=np.float64)
    col, row = np.meshgrid(q, q)
    r = 2 * q + 1
    m = -(np.where(row >= col, r, 0) - np.diag(q))
    t = np.sqrt(np.diag(2 * q + 1))
    a = t @ m @ np.linalg.inv(t)
    b = np.diag(t)[:, None].copy()
    return a, b


def nplr_legs(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """w (N/2), P (1, N/2), B (N/2): A = V[diag(w) - P P*]V*, one of each conjugate pair."""
    a, b = _legs(n)
    a_t = torch.as_tensor(a, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)[:, 0]
    p = torch.sqrt(0.5 + torch.arange(n, dtype=torch.float)).unsqueeze(0)  # (1, N)
    ap = a_t + torch.sum(p.unsqueeze(-2) * p.unsqueeze(-1), dim=-3)
    w, v = torch.linalg.eig(ap)
    w, v = w[..., 0::2].contiguous(), v[..., 0::2].contiguous()
    v_inv = v.conj().transpose(-1, -2)
    b_c = torch.einsum("ij,j->i", v_inv, b_t.to(v))
    p_c = torch.einsum("ij,...j->...i", v_inv, p.to(v))
    return w, p_c, b_c


def _conj(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.conj()], dim=-1)


def _power(length: int, a: torch.Tensor) -> torch.Tensor:
    """A^length by repeated squaring (dl_models/s4.py power without v)."""
    out = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device).expand_as(a).clone()
    base = a
    while True:
        if length % 2 == 1:
            out = base @ out
        length //= 2
        if length == 0:
            return out
        base = base @ base


class NPLRKernel(nn.Module):
    """SSKernelNPLR of dl_models/s4.py for rank 1, rate 1 and a fixed length L: A (as w, P, Q),
    B and dt are buffers, C (`channels` x H x N/2, complex) the only parameter."""

    def __init__(
        self, h: int, n: int, length: int, channels: int, dt_min: float, dt_max: float,
        dt_scale: float,
    ) -> None:  # fmt: skip
        """`dt_scale`: every time step is multiplied by it (1: the published model; for a channel
        at rate fs, base_rate / fs, so the steps are the same in seconds at every rate)."""
        super().__init__()
        self.H, self.L, self.dt_scale = h, length, dt_scale
        w, p, b = nplr_legs(n)
        self.N = w.size(-1)
        log_dt = torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        c = torch.randn(channels, h, self.N, dtype=torch.cfloat)
        self.register_buffer("log_dt", log_dt)
        self.register_buffer("B", torch.view_as_real(b.expand(1, h, self.N).clone()))
        self.register_buffer("P", torch.view_as_real(p.unsqueeze(1).expand(1, h, self.N).clone()))
        self.register_buffer("w", torch.view_as_real(w.expand(h, self.N).clone()))
        q = p.unsqueeze(1).expand(1, h, self.N).conj().clone()
        self.register_buffer("Q", torch.view_as_real(q))
        omega = torch.tensor(np.exp(-2j * np.pi / length), dtype=torch.cfloat)
        omega = omega ** torch.arange(0, length // 2 + 1)
        self.register_buffer("omega", torch.view_as_real(omega))
        self.register_buffer("z", torch.view_as_real(2 * (1 - omega) / (1 + omega)))
        self.C = nn.Parameter(torch.view_as_real(c))
        self._length_correction()

    def _parts(self) -> tuple[torch.Tensor, ...]:
        r2c = torch.view_as_complex
        dt = torch.exp(self.log_dt) * self.dt_scale
        return r2c(self.w), r2c(self.B), r2c(self.P), r2c(self.Q), dt

    def discrete(self) -> tuple[torch.Tensor, torch.Tensor]:
        """dense dA (H, 2N, 2N) and dB (H, 2N) of the bilinear discretization, through the
        linear step of the DPLR form (_setup_linear, _step_state_linear, _setup_state)."""
        w, b, p, q, dt = self._parts()
        d = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        r = torch.eye(1, dtype=w.dtype, device=w.device) + 2 * torch.einsum(
            "rhn,hn,shn->hrs", q, d, p
        ).real
        q_d = (q * d).permute(1, 0, 2)  # (H, r, N)
        r = torch.linalg.solve(r.to(q_d), q_d).permute(1, 0, 2)  # (r, H, N)
        e = 2.0 / dt.unsqueeze(-1) + w

        def step(
            state: torch.Tensor, u: torch.Tensor, full: bool
        ) -> torch.Tensor:  # one step of the linear recurrence
            if full:  # the state has both conjugate halves
                dd, ee, rr, pp, qq, bb = (_conj(t) for t in (d, e, r, p, q, b))

                def contract(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                    return torch.einsum("rhn,rhm,...hm->...hn", x, y, s)
            else:
                dd, ee, rr, pp, qq, bb = d, e, r, p, q, b

                def contract(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                    return torch.einsum(
                        "rhn,rhm,...hm->...hn", _conj(x), _conj(y), _conj(s)
                    )[..., : self.N]
            new = ee * state - contract(pp, qq, state)
            new = new + 2.0 * bb * u.unsqueeze(-1)
            return dd * (new - contract(pp, rr, new))

        eye = torch.eye(2 * self.N, dtype=w.dtype, device=w.device).unsqueeze(-2)  # (2N, 1, 2N)
        zero_u = torch.zeros(self.H, dtype=w.dtype, device=w.device)
        da = step(eye, zero_u, full=True).permute(1, 2, 0)  # (H, 2N, 2N): "n h m -> h m n"
        zero_s = torch.zeros(self.H, self.N, dtype=w.dtype, device=w.device)
        db = _conj(step(zero_s, torch.ones_like(zero_u), full=False))[0]  # (H, 2N)
        return da, db

    @torch.no_grad()
    def _length_correction(self) -> None:
        """C <- C (I - dA^L) on both conjugate halves, then one half again (_setup_C)."""
        da, _ = self.discrete()
        da_l = _power(self.L, da)
        c = _conj(torch.view_as_complex(self.C))
        prod = torch.einsum("hmn,chn->chm", da_l.transpose(-1, -2), c)
        self.C.copy_(torch.view_as_real((c - prod)[..., : self.N].contiguous()))

    def forward(self) -> torch.Tensor:
        """The kernel (channels, H, L), real."""
        w, b, p, q, dt = self._parts()
        c = torch.view_as_complex(self.C)
        omega, z = torch.view_as_complex(self.omega), torch.view_as_complex(self.z)
        w = w * dt.unsqueeze(-1)
        bp = torch.cat([b, p], dim=-3)  # (2, H, N)
        cq = torch.cat([c, q], dim=-3)  # (channels + 1, H, N)
        v = bp.unsqueeze(-3) * cq.unsqueeze(-4)  # (2, channels + 1, H, N)
        # the Cauchy kernel over both conjugate halves (cauchy_conj / cauchy_mult symmetric)
        zz = z.view(1, 1, 1, 1, -1)
        vv, ww = v.unsqueeze(-1), w.view(1, 1, *w.shape, 1)
        r = (vv / (zz - ww) + vv.conj() / (zz - ww.conj())).sum(dim=-2)  # (2, c + 1, H, L/2+1)
        r = r * dt[None, None, :, None]
        k_f = r[:-1, :-1] - r[:-1, -1:] * r[-1:, :-1] / (1 + r[-1:, -1:])  # Woodbury, rank 1
        k_f = k_f * 2 / (1 + omega)
        k = torch.fft.irfft(k_f)[..., : self.L]
        return k[-1]  # (channels, H, L)


def _channel_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """nn.Dropout2d of PyTorch 1.10 on (batch, features, time): whole features of a sample."""
    if not training or p == 0.0:
        return x
    keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= p
    return x * keep / (1 - p)


class S4Layer(nn.Module):
    """The S4 block of dl_models/s4.py: bidirectional SSM convolution + D skip, GELU, channel
    dropout, 1x1 linear to 2H, GLU."""

    def __init__(self, s: S4Settings, length: int, dt_scale: float) -> None:
        super().__init__()
        h = s.d_model
        self.dropout = s.dropout
        self.D = nn.Parameter(torch.randn(1, h))
        self.kernel = NPLRKernel(h, s.d_state, length, 2, s.dt_min, s.dt_max, dt_scale)
        self.output_weight = nn.Parameter(torch.empty(2 * h, h))
        nn.init.kaiming_uniform_(self.output_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(h)
        self.output_bias = nn.Parameter(torch.empty(2 * h, 1).uniform_(-bound, bound))

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # (B, H, L)
        length = u.size(-1)
        k = self.kernel()[:, :, :length]  # (2, H, L)
        k = F.pad(k[:1], (0, length)) + F.pad(k[1:].flip(-1), (length, 0))  # (1, H, 2L)
        y = torch.fft.irfft(
            torch.fft.rfft(u, n=2 * length).unsqueeze(1) * torch.fft.rfft(k, n=2 * length),
            n=2 * length,
        )[..., :length]  # (B, 1, H, L)
        y = (y + u.unsqueeze(1) * self.D.unsqueeze(-1)).flatten(1, 2)  # (B, H, L)
        y = _channel_dropout(F.gelu(y), self.dropout, self.training)
        y = torch.einsum("bul,vu->bvl", y, self.output_weight) + self.output_bias
        return F.glu(y, dim=-2)


class S4Model(nn.Module):
    """S4Model of dl_models/s4_model.py (post-norm, LayerNorm, bidirectional)."""

    def __init__(self, s: S4Settings, inputs: int, classes: int, length: int) -> None:
        super().__init__()
        self.dropout = s.dropout
        self.encoder = nn.Conv1d(inputs, s.d_model, 1)
        self.layers = nn.ModuleList(S4Layer(s, length, 1.0) for _ in range(s.n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(s.d_model) for _ in range(s.n_layers))
        self.decoder = nn.Linear(s.d_model, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, leads, L) -> (B, classes)
        x = self.encoder(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            z = _channel_dropout(layer(x), self.dropout, self.training)
            x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)
        out: torch.Tensor = self.decoder(x.mean(dim=-1))
        return out


class S4ChannelModel(nn.Module):
    """reference_s4_channels: S4Model with the leads as a set (user, 2026-10-29). Each lead is its
    own sequence through ONE shared stack:

    - a per-sample Linear(1, d) (the encoder of S4Model mixes the 12 leads in a fixed order:
      here no lead order exists), plus a learned identity vector per lead (one per lead name,
      the fixed slots of our model), small at start (`identity_std`);
    - the S4 blocks of S4Model (S4 -> channel dropout -> + input -> LayerNorm), with the time
      steps scaled by base_rate / fs, so the same weights take another rate in seconds;
    - after each block, attention across the leads at each sample time (pre-norm LayerNorm,
      multi-head self-attention over the C leads, residual): the only place leads meet;
    - the head of S4Model on the mean over time and leads.

    Without the identity vectors the model is the same for any order of the leads."""

    def __init__(
        self, s: S4ChannelsSettings, leads: int, classes: int, length: int, rate_hz: float
    ) -> None:
        super().__init__()
        d = s.d_model
        self.dropout = s.dropout
        self.encoder = nn.Linear(1, d)
        self.identity = nn.Parameter(torch.randn(leads, d) * s.identity_std)
        dt_scale = s.base_rate_hz / rate_hz
        self.layers = nn.ModuleList(S4Layer(s, length, dt_scale) for _ in range(s.n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(s.n_layers))
        self.mix_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(s.n_layers))
        self.mixers = nn.ModuleList(
            nn.MultiheadAttention(d, s.heads, batch_first=True) for _ in range(s.n_layers)
        )
        self.decoder = nn.Linear(d, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, leads, L) -> (B, classes)
        b, c, length = x.shape
        h = self.encoder(x.unsqueeze(-1)) + self.identity[None, :, None, :]  # (B, C, L, d)
        h = h.flatten(0, 1).transpose(-1, -2)  # (B C, d, L)
        for layer, norm, mix_norm, mixer in zip(
            self.layers, self.norms, self.mix_norms, self.mixers, strict=True
        ):
            z = _channel_dropout(layer(h), self.dropout, self.training)
            h = norm((z + h).transpose(-1, -2)).transpose(-1, -2)
            t = h.view(b, c, -1, length).permute(0, 3, 1, 2).reshape(b * length, c, -1)
            q = mix_norm(t)
            t = t + mixer(q, q, q, need_weights=False)[0]  # across the leads, per sample time
            h = t.view(b, length, c, -1).permute(0, 2, 3, 1).reshape(b * c, -1, length)
        pooled = h.view(b, c, -1, length).mean(dim=(1, 3))  # (B, d): over leads and time
        out: torch.Tensor = self.decoder(pooled)
        return out


def train_and_score(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, train: tuple[int, ...],
    validation: tuple[int, ...], test: tuple[int, ...], rate: float, s: S4Settings,
    seed: int, device: torch.device, log: Callable[[dict[str, float]], None], prefix: str,
    macro_auc: Callable[[np.ndarray, np.ndarray], tuple[float, int]],
) -> tuple[dict[str, float], tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """Train with the recipe; the epoch with the best validation macro-AUC is scored on the test
    folds (fold 10 only for the epochs fold 9 chooses). `x`: [records, leads, samples], mV.
    S4ChannelsSettings: the channel-set model (S4ChannelModel), otherwise S4Model.
    Also returned: the chosen epoch's predictions on the validation and test folds (as scored:
    `validation_on`, `test_on`)."""
    random.seed(seed)
    torch.manual_seed(seed)
    tr, va, te = (np.flatnonzero(np.isin(folds, f)) for f in (train, validation, test))
    data = torch.from_numpy(x.astype(np.float32)).to(device)  # normalize False: as recorded
    target = torch.from_numpy(y.astype(np.float32)).to(device)
    size = int(round(s.input_size_s * rate))

    def build() -> nn.Module:
        return (
            S4ChannelModel(s, x.shape[1], y.shape[1], size, rate)
            if isinstance(s, S4ChannelsSettings)
            else S4Model(s, x.shape[1], y.shape[1], size)
        ).to(device)

    model = build()
    log({f"{prefix}/parameters": float(sum(p.numel() for p in model.parameters()))})
    optimizer = torch.optim.AdamW(model.parameters(), lr=s.learning_rate, betas=s.betas,
                                  eps=s.eps, weight_decay=s.weight_decay)  # fmt: skip
    per_epoch = len(tr) // s.batch_size  # drop_last
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer, warmup_cosine(int(round(s.warmup_epochs * per_epoch)), s.epochs * per_epoch)
        )
        if s.schedule == "warmup_cosine"
        else None
    )
    criterion = loss_function(s, y[tr], device)
    ema = Ema(model, s.ema_decay) if s.ema_decay is not None else None
    scored = ema.model if ema is not None else model  # the weights validated and tested
    augment = Augmentation(s.augmentation, rate) if s.augmentation is not None else None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    teacher = None
    if s.distillation is not None:
        teacher = build().eval()
        teacher.load_state_dict(torch.load(s.distillation.teacher_weights, map_location=device,
                                           weights_only=True))  # fmt: skip
    length = x.shape[2]
    starts = crop_starts(length, size, int(round(size * s.valid_stride_fraction)))

    def predict(net: nn.Module, ids: np.ndarray, on: str) -> np.ndarray:
        net.eval()
        out = []
        with torch.no_grad():
            for k in range(0, len(ids), s.batch_size):
                batch = data[torch.from_numpy(ids[k : k + s.batch_size]).to(device)]
                crops = torch.stack([batch[:, :, a : a + size] for a in starts], 1)
                p = net(crops.flatten(0, 1)).view(len(batch), len(starts), -1)
                out.append((torch.sigmoid(p) if on == "probabilities" else p).mean(1).cpu().numpy())
        return np.concatenate(out)

    best_val, best_test, best_epoch, labels_n = -1.0, 0.0, 0, 0
    best_p_val, best_p_test = np.zeros((0, y.shape[1])), np.zeros((0, y.shape[1]))
    for epoch in range(1, s.epochs + 1):
        model.train()
        order = tr[torch.randperm(len(tr)).numpy()]
        losses = []
        for k in range(per_epoch):
            ids = order[k * s.batch_size : (k + 1) * s.batch_size]
            if augment is None:  # the published crops
                first = [0 if length == size else random.randint(0, length - size - 1) for _ in ids]  # noqa: S311
                batch = torch.stack(
                    [data[i, :, a : a + size] for i, a in zip(ids, first, strict=True)]
                )
            else:
                batch = augment.crops(data[torch.from_numpy(ids).to(device)], size, generator)
            goal = target[ids]
            if teacher is not None and s.distillation is not None:
                with torch.no_grad():
                    soft = torch.sigmoid(teacher(batch))
                a = s.distillation.teacher_fraction
                goal = (1 - a) * goal + a * soft
            logits = model(batch)
            loss = (
                F.binary_cross_entropy_with_logits(logits, goal)
                if criterion is None
                else criterion(logits, goal)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())
        p_val = predict(scored, va, s.validation_on)
        auc_val, _ = macro_auc(y[va], p_val)
        record = {f"{prefix}/epoch": float(epoch),
                  f"{prefix}/epoch_train_loss": float(np.mean(losses)),
                  f"{prefix}/epoch_validation_macro_auc": auc_val}  # fmt: skip
        if scheduler is not None:
            record[f"{prefix}/learning_rate"] = float(scheduler.get_last_lr()[0])
        if ema is not None:  # the trained weights themselves, for comparison only
            raw, _ = macro_auc(y[va], predict(model, va, s.validation_on))
            record[f"{prefix}/epoch_validation_macro_auc_raw"] = raw
        if auc_val > best_val:  # ModelCheckpoint(monitor="val/val_macro_agg", mode="max")
            best_val, best_epoch = auc_val, epoch
            best_p_val, best_p_test = p_val, predict(scored, te, s.test_on)
            best_test, labels_n = macro_auc(y[te], best_p_test)
            record[f"{prefix}/chosen_epoch"] = float(epoch)
            record[f"{prefix}/chosen_test_macro_auc"] = best_test
            if s.save_chosen_to is not None:
                torch.save(scored.state_dict(), s.save_chosen_to)
        log(record)
        if s.patience_epochs is not None and epoch - best_epoch >= s.patience_epochs:
            log({f"{prefix}/stopped_epoch": float(epoch)})
            break
    return {"macro_auc": best_test, "labels": float(labels_n), "epoch": float(best_epoch),
            "validation_macro_auc": best_val}, (best_p_val, best_p_test)  # fmt: skip

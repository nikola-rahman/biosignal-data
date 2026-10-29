# v3 tokenizer bench

Why: the v0-v2 token is 64 consecutive samples of one channel through one linear layer, so a
token lasts 0.64 s at 100 Hz (about a whole beat) and 0.128 s at 500 Hz. Trained from scratch
on PTB-XL "all" at 100 Hz (fold-10 macro-AUC, 2026-10-29): 64 samples 0.895, 16 samples 0.915,
8 samples 0.917; the CNN xresnet1d101 through the same pipeline 0.927; supervised S4 0.942
(Mehari & Strodthoff 2023, arXiv 2308.15291, Table III: 0.941-0.942 for every train/test pair
of 100/200/500 Hz). Small sample tokens help but are no fix: 8 samples are 0.08 s at 100 Hz and
0.016 s at 500 Hz, and a fixed number of samples still means a different duration per rate.

The bench compares two tokenizers whose units are seconds, trained from scratch (no
pretraining needed to choose between them), against the sample tokens.

## The model's setting

`model.tokenizer` (train config; `from_scratch_tokenizer` in a PTB-XL benchmark config) is one
of three kinds. Every setting has no default. Checkpoints written before the field existed read
as `{kind: samples}` (`ADDED_LATER` in `train.py`).

| kind | what it is |
|---|---|
| `samples` | v0-v2: `tokens.samples_per_token` samples per token, one linear map. Unchanged. |
| `duration` (D1) | tokens of a fixed duration with overlap, embedded by a continuous-kernel stem |
| `hierarchical` (D2) | frames of a fixed duration through the same stem, a local bidirectional S4D stage over each channel's frames, attention pooling of `frames_per_token` frames into one global token |

## Frames (both D1 and D2)

A frame is the unit the stem reads. For a channel at rate `fs`:

- samples per frame `n = max(1, round(frame_s * fs))`, hop `h = max(1, round(frame_hop_s * fs))`
  (`frame_s >= frame_hop_s`, so frames overlap or touch). Both are whole samples, so every frame
  of a channel has the same sample offsets, and the stem's kernels depend only on `fs`.
- frame `k` covers samples `[k h - (n - h) // 2, ... + n)`: centred on its core
  `[k h, (k + 1) h)`. Samples outside the channel are not observed (NaN → mask 0).
- a channel of `S` samples has `M = ceil(S / h)` frames; `F = frames_per_token` frames make one
  token (D1: `F = 1`), so it has `ceil(M / F)` tokens; missing frames at the end are not observed.
- a token's time is the centre of its frames' cores, `start_s + (g F + F / 2) h / fs`; its
  length `F h / fs` (the `log_length_s` input).
- values are scaled per channel as in v0-v2: `asinh((x - mean) / std)`, mean and std of the
  channel's samples whose token is not hidden; `asinh(mean)`, `log(std)` go in as inputs.

For D1, `frame_s = token_s` and `frame_hop_s = hop_s`. Bench: D1 0.08 s / 0.04 s; D2 0.04 s /
0.02 s and 12 frames per token (0.24 s).

| | 100 Hz | 500 Hz |
|---|---|---|
| D1 frame (0.08 s / 0.04 s) | n = 8, h = 4 | n = 40, h = 20 |
| D2 frame (0.04 s / 0.02 s) | n = 4, h = 2 | n = 20, h = 10 |

The packer stores a batch as `values`, `observed` `[R, L, F, P]` with `P` the largest `n` in the
batch (the rest of a frame is not observed), and per token `rate_hz` (the stem builds its
kernels from it), `channel_index` (the channel within the row) and `position` (the token's
order within its channel: D2 rebuilds each channel's frame sequence from them). The
`samples` kind keeps its arrays `[R, L, P]` and none of the new fields.

## The stem: continuous-time kernels (D1 and D2)

The stem is a strided convolution whose kernel is a function of time in seconds, so one weight
set works at every rate (continuous-kernel convolutions: CKConv, arXiv 2102.02611; FlexConv,
arXiv 2110.08059, whose kernel frequencies are controlled to avoid aliasing and which run at
resolutions not seen in training).

- `stem_windows` sub-windows `K` per frame, centred at `c_k = -T/2 + (k + 1/2) T/K` (`T = n/fs`),
  each a Hann window of half-width `w = T/K` (50 % overlap).
- kernel `φ(τ) = hann(τ / w) · B(τ) W`: `B(τ) = [1, cos 2πf_m τ, sin 2πf_m τ]` for
  `kernel_frequencies` frequencies `f_m` log-spaced from `kernel_min_hz` to `kernel_max_hz`;
  **Nyquist limit**: at rate `fs` the columns with `f_m >= fs/2` are zero, so no kernel holds a
  frequency the samples cannot carry. `W` is learned, the same for every sub-window (a
  convolution) and every rate.
- sub-window output: `z_k = Σ_j (u_j φ(τ_j - c_k) + m_j φ'(τ_j - c_k)) / (fs w) + b`, a Riemann
  sum of `∫ u(t) φ(t - c_k) dt / w`, so its size does not depend on the rate; `m` is the observed
  mask with its own kernel `φ'`.
- `GELU`, the `K` outputs side by side, one linear layer to the frame's width.

Bench: `K = 4` (D1) / `2` (D2), 64 channels, 24 frequencies from 1 to 240 Hz (below the
500 Hz Nyquist frequency).

## D1 `duration`

Token = one frame. Embedding = stem output (width `model.width`) + the linear map of
(log length, channel scale) + the slot embedding, as v0-v2; hidden tokens get the learned
vector. Then the global transformer of v0-v2 (rotary time in seconds, attention within an
example). Pretraining output: each token's samples, by a continuous-time output kernel
`ŷ_j = h · (B(τ_j) W_out)` with the same Nyquist limit, compared on observed samples of hidden
tokens (the v0-v2 loss). Open for pretraining: with overlapping tokens a hidden token's samples
are partly seen in its neighbours, so masking must hide spans of time (not part of this bench,
where nothing is hidden).

## D2 `hierarchical`

1. Stem → frame features (width `local_width`).
2. Each channel's frames in time order (from `channel_index`, `position`) through
   `local_layers` residual blocks: RMSNorm → bidirectional S4D (diagonal state-space layer,
   arXiv 2206.11893: `local_states` states, S4D-Lin initialization `A_n = -1/2 + iπn`, ZOH, the
   kernel computed for the sequence length and applied as an FFT convolution, one kernel per
   direction) → GELU → linear with GLU. The frame step is a fixed 20 ms whatever the rate, so the
   S4D step needs no rate term. Frames that are not observed are zero before and after each
   block, so padding and other channels never enter a sequence.
3. Attention pooling: `pool_heads` learned queries over the token's `F` observed frames → one
   vector → linear to `model.width`. Tokens with no observed frame give 0.
4. + (log length, channel scale) + slot, hidden vector for hidden tokens, the global transformer.
5. Output features = RMSNorm(global output + linear(mean of the token's local frame features)):
   the local stage reaches the task head directly, an encoder-to-decoder residual as in H-Net
   (arXiv 2507.07955) and the Byte Latent Transformer (arXiv 2412.09871), so short-time detail
   does not have to pass the 0.24 s tokens.

Pretraining with D2 (a decoder from tokens and frames to samples, masking of local frames) is not
part of this bench: `Encoder.forward` raises for it.

## Bench protocol

PTB-XL "all", from scratch, the recipe of `runs/bench-ptbxl-from-scratch-tokens8-100hz.yaml`:
AdamW lr 1e-4, betas (0.9, 0.95), warmup 2.5 epochs, cosine to 0 over 50 epochs, clip 1.0,
weight decay 1e-3, 64 records per step, 2.5 s crops, fixed slots per lead, the global
transformer of v2 (width 512, 8 layers, 8 heads). Fold 9 chooses the epoch, fold 10 is scored
once for it.

| run | tokenizer | rate |
|---|---|---|
| `d1-100hz`, `d1-500hz` | D1 0.08 s / 0.04 s | 100, 500 Hz |
| `d2-100hz`, `d2-500hz` | D2 0.04 s / 0.02 s × 12 | 100, 500 Hz |
| `samples64-500hz`, `samples40-500hz` | samples, 64 and 40 per token | 500 Hz |

Earlier, the same recipe at 100 Hz: samples 64 / 16 / 8 → 0.895 / 0.915 / 0.917.

## Costs (2.5 s crop, 12 leads)

| tokenizer | 100 Hz | 500 Hz |
|---|---|---|
| samples 64 | 48 tokens | 240 tokens |
| samples 40 | – | 384 tokens |
| samples 8 | 384 tokens | 1,884 tokens |
| D1 0.08 / 0.04 s | 756 tokens | 756 tokens |
| D2 0.04 / 0.02 s × 12 | 132 global tokens, 1,500 frames | 132 global tokens, 1,500 frames |

D1 has the same number of tokens at every rate; its attention costs about 3.9× that of 8-sample
tokens at 100 Hz and 0.16× at 500 Hz (L²).
D2's global sequence is about a third of 8-sample tokens at 100 Hz and a fourteenth at 500 Hz; its
local stage costs grow linearly with frames (FFT convolutions over each channel).

## Proxy bench (2026-10-29)

Small proxy models (user: "small proxy models, successive halving"): every candidate gets a
global transformer of width 256, 4 layers, 4 heads (`from_scratch_model`), its feed-forward
size set so that every proxy has 4.2-4.3 M parameters (without the pretraining heads); recipe
lr 1e-4, betas (0.9, 0.95), warmup 1.5 epochs, cosine over 25 epochs, clip 1.0, weight decay
1e-3, fixed slots, 2.5 s crops, no dropout (`dropout`, `drop_path` 0; 0.1 is the second
level). Task "all" is also scored per statement group (diagnostic, form, rhythm; fold 9 alone
chooses the epoch). Every method (also reference_xresnet1d101) stores in the result JSON each
label's AUC on fold 9 and fold 10 for the chosen epoch (`<task>/<method>/label/<code>/
{validation,test}_auc`) and its positives per fold (`..._positives`): to find the labels that
make our runs lose ~1.3 points from fold 9 to fold 10 (S4 and xresnet lose ~0). First the
proxies must rank as the full-size models did.

| proxy (config `bench-ptbxl-proxy-*-100hz`) | global tokens per 2.5 s, 12 leads, 100 Hz | params (global ff) |
|---|---|---|
| `samples64`: 64 samples | 48 | 4.24 M (1024) |
| `samples8`: 8 samples | 384 | 4.21 M (1024) |
| `d1`: D1, 80 ms tokens every 40 ms | 756 | 4.28 M (1024) |
| `d1fine`: D1, 40 ms tokens every 20 ms | 1,500 | 4.28 M (1024) |
| `d2`: D2 as in the first bench (local 128 × 4, 0.24 s tokens, mean residual) | 132 (+1,500 local frames) | 4.20 M (896) |
| `d2prime`: D2' (local 256 × 6, 0.12 s tokens, attention residual) | 252 (+1,500) | 4.32 M (512) |
| `d2second`: D2'' (D2' local stage, 0.08 s tokens, attention residual) | 384 (+1,500) | 4.32 M (512) |

### What each candidate can resolve, checked against the ECG scales

The scales a 12-lead diagnosis needs (computational cardiology): QRS 80-120 ms (> 120 ms is
abnormal), delta waves and notching 20-40 ms, PR 120-200 ms, QT 350-450 ms with timing to
about ±10 ms, ST level and T shape relative to the baseline (amplitude), rhythm over seconds,
territory from lead identity. At 100 Hz one sample is 10 ms, so the data itself limits timing
to about ±5-10 ms and 20 ms features are 2 samples.

| candidate | resolution of the global model (token / hop) | the token's own features | local stage capacity | QRS 80-120 ms | notching 20-40 ms | intervals ±10 ms |
|---|---|---|---|---|---|---|
| samples 64 | 640 ms / 640 ms | one linear map of 64 samples | none | inside one token: only a linear code | linear code only | global positions every 640 ms: no |
| samples 8 | 80 ms / 80 ms | one linear map of 8 samples | none | 1-2 tokens | inside a token, linear code | 80 ms grid, finer only linearly |
| D1 80/40 | 80 ms / 40 ms | nonlinear stem: 4 Hann sub-windows of 20 ms, 64 channels | stem only (~0.07 M) | 2-3 tokens | in the 20 ms sub-windows, nonlinear | 40 ms grid; sub-token timing through the stem |
| D1 40/20 | 40 ms / 20 ms | stem: 2 sub-windows of 20 ms | stem only | 4-6 tokens | at the token scale | 20 ms grid with overlap: about the data limit |
| D2 (first bench) | 240 ms / 240 ms | attention pool of 12 local frames | 4 × S4D, width 128 (~0.3 M), each lead's whole crop | the local stage only | local frames (20 ms hop) | local only; the global grid is 240 ms |
| D2' | 120 ms / 120 ms | attention pool of 6 frames | 6 × S4D, width 256 (~1.2 M + pools ~0.4 M) | 1 token + local stage | local frames | local 20 ms, global 120 ms |
| D2'' | 80 ms / 80 ms | attention pool of 4 frames | as D2' | 1-2 tokens + local stage | local frames | local 20 ms, global 80 ms |

The S4D stage of D2 runs over each lead's whole 2.5 s crop in both directions, so its
receptive field covers every interval above. Rhythm (seconds) is in the global model's
context for all candidates (the whole 2.5 s crop; the record is scored over its crops).

**Amplitude** (the same for all candidates; ST level and T shape need it): each lead in each
crop is scaled as asinh((x - mean) / std), so the shape relative to the lead's mean stays in
the waveform; the lead's level and size reach the model only as (asinh(mean), log std), one
additive input per token (`context`). Ratios between leads (axis, voltage criteria) and the
absolute baseline therefore come only through that scalar path; the std of a 2.5 s crop also
depends on how many beats it holds. This is a known weak point, not changed in this bench
(one factor at a time).

**Lead identity** (territory): fixed slots per lead name in every proxy; the slot embedding is
added to every global token. D2's local stage is shared by all leads and does not see the
lead: morphology is read lead-agnostically, and territory is decided by the global model.

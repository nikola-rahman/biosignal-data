# Biosignal foundation model: the v4 architecture

The design of the v4 model (revision 2, 2026-10-29). Revision 1 was reviewed by 8 experts (ECG,
EEG/EMG, information theory, LLM, vision, audio, foundation-model architect, code); this
revision takes their consensus and the user's decisions. Each change names its reason.
"Ladder" = the scaling ladder in section 9. The code is built in the order of section 11; what
is built is marked **(built)**.

**The first v4 runs (user decision 2026-11-03: "elegant and simple, without over-engineering and
guessing").** Only the three problems measured in v0-v2 change: the tokens (the local stage at
every sample, 40 ms global tokens), the mask leak (time spans over all channels, in whole grid
cells) and the objective (latent targets, data2vec 2.0 type). Everything else stays as in v0-v2:
the value transform per window (section 2), the signal type as the only descriptor (section 5),
random channel slots. The other parts of this design wait until the ladder shows they are needed.

User decisions (2026-10-29):

- the model uses channel metadata when it exists and works without it;
- the 50 % error cut is the north star on every task;
- loader transforms are the lead's choice;
- more data (TUEG, NSRR) later, a cap per subject now;
- **no hand-set values per signal type anywhere** (no per-type tables of token durations, mask
  spans, scales or bands in code or config): one global rule, a value computed from the corpus,
  or a learned part.

## 1. Data and the unit of data

- The unit is the **subject**, not the hour: i-care has ~607 patients and gives > half of all
  samples; hd-semg has 20 subjects.
  - Cap per subject per epoch: random windows (blocks), at most a set number of hours per
    subject per epoch (start: 4 h). **(built)**: `data.subject_hours_per_epoch`.
  - UniMax over (type x dataset), sized by subjects; at most 4 epochs of a dataset.
- The loss, the mixture and the compute budget count the same unit: **global tokens** (section 4).
  The v0-v3 loss is a mean over samples, so 2 kHz EMG weighs 8x EEG per second. **(built)**: the
  mixture counts global tokens (`data.mixture.unit: global_tokens`).
- Quality masks (flat line, lead-off, clipping) from the quality layer: masked out of the loss.
- Checks before use: hd-semg raw and preprocessed files must not both be in the index; benchmark
  test subjects held out (exists for PTB-XL; check for every new task).

## 2. Input values

**Built for the first runs (as v0-v2):** per channel of an example, x' = asinh((x - m) / s) with m
and s the mean and standard deviation of its **visible** valid samples (s = 1 if constant);
asinh(m) and log(s) are inputs of the model (the channel embedding), so the amplitude is kept.
The teacher reads the same transform of every valid sample. The corpus scale table below is not
built: 2025-26 practice has no shared cross-modality scheme (PhysioOmni: one fixed constant;
EEG models: fixed constants; ECG/PPG and time-series models: per-window normalization), so the
simpler, already tested transform stays until the ladder shows a need.

The earlier plan (not built):

- Per channel: x' = (a / s) * asinh((x - c) / a). The scale s is **computed from the corpus per
  (type, unit)**; the knee is one **global knee factor** times s (a = k * s), the same for every
  type. No table of hand-set scales or knees.
- c is the **median of the visible samples of the crop**; it is removed and given to the model as
  a **separate input**, so the absolute level (DC: blood pressure, SpO2, temperature) is not lost.
- Channels without a unit (most EEG, SpO2, PPG): **one scale per (dataset, type)**, never per
  recording or crop. Measured 2026-10-29: i-care "nu" is already rescaled (not ADC counts), the 5
  hospitals differ by only ~20 % (3.5 % of the variance; within a hospital a factor 2.6), and its
  amplitude matches calibrated EEG: nu ~ uV, factor ~1 (to confirm in the i-care documentation). A
  gain per hospital fitted to the background would remove patient and outcome effects: not used.
  The descriptor says "physical", "dataset-calibrated" or "unknown" (section 5).
- Measured on the data (2026-10-29): EMG of surface / HD grids has a MAD of 0.027 mV (grabmyo);
  PSG chin / leg EMG 1-7 uV; one centre for all pressures put CVP 95 % in the nonlinear part.
  These are why the scale must come from the corpus per (type, unit), not from one constant.
- Data-layer fixes before training: hd-semg is labelled V but is very probably mV (MAD 0.023 "V" vs
  0.027 mV grabmyo; confirm in the source); flat channels (all sampled i-care and siena SpO2, 12 %
  of i-care EEG, 20 % of i-care EMG, 27 % of mimic4wdb resp) masked by the quality layer;
  sleep-edfx "EMG" at 1 Hz is an envelope (own type).
- Why: per-crop or per-recording scaling removes absolute amplitude (i-care prognosis, voltage
  criteria) and ratios between channels. The S4 winner (0.940) reads raw mV.
- Not built yet: the v4 batch carries the loader's physical values; the transform is a later step.

## 3. Local stage: every sample, channels mixed early **(built)**

- Per channel, at its own rate: a bidirectional diagonal complex SSM (S4D) defined in continuous
  time (lambda = a / Delta, Delta in seconds), discretized with dt = 1/fs by **centred ZOH**
  (decided 2026-10-29 from the measurement below): the poles of ZOH, exact at every rate, and the
  kernel integrated over [(l - 1/2) dt, (l + 1/2) dt]. Plain ZOH delays by half a sample (5 ms at
  100 Hz, 1 ms at 500 Hz), and that delay differs between channels of different rates that the
  model mixes. One kernel per distinct rate in the batch; kernel length capped at 5 decay times of
  the slowest unmasked mode.
- Mode time scales initialized log-uniform between two set bounds (start: 1 ms to 10 s); a mode is
  masked when |lambda| dt > pi (above Nyquist or decaying faster than one sample), and a masked
  mode gives its **static gain** -C/a at lag 0 (decided 2026-10-29): its response to in-band
  input, and the ZOH limit of a mode that decays within one sample. Rates above any seen in
  training: the same rule at test time.
- Measured at random init (2026-10-29, `scripts/v4_local_checks.py`; width 128, 4 layers, 32
  modes, 3 channels of sinusoids 0.1-10 Hz, 10 s; relative feature error against 500 Hz at the
  common times, 1 s from each end):

  | masked modes, discretization | 100 Hz | 128 Hz | 173 Hz | 250 Hz |
  |---|---|---|---|---|
  | dropped (first design), ZOH | 0.398 | 0.213 | 0.125 | 0.092 |
  | static gain, ZOH | 0.0123 | 0.0086 | 0.0056 | 0.0033 |
  | static gain, centred ZOH | **0.0038** | **0.0022** | **0.0014** | **0.0007** |
  | static gain, bilinear | 0.0196 | 0.0136 | 0.0091 | 0.0049 |

  Dropping masked modes fails: a fast mode's static gain does not depend on its time scale, so
  a low rate loses a gain that a high rate keeps. Frequency response of one layer against the
  continuous system (median over features, no mode masked): ZOH at 100 Hz has phase +18 deg at
  10 Hz and +72 deg at 40 Hz (the half-sample delay), magnitude 1.30 at 40 Hz; centred ZOH has
  phase 0 and a **known droop near Nyquist**: magnitude 0.98 at 10 Hz, 0.88 at 20 Hz, 0.44 at
  40 Hz at 100 Hz (0.98 at 40 Hz at 500 Hz). Bilinear is like ZOH in the median (+17.5 deg at
  10 Hz at 100 Hz).
- **Early channel mixing** (the hybrid's lesson: attention across leads at every sample): in a set
  subset of the layers (start: 2 of 4), attention across co-located channels at each sample time,
  for channels of one example with a shared time base (ECG leads, EEG electrodes, PSG). For grids
  (HD-EMG) a 3 x 3 neighbour convolution instead, once grid positions are descriptors (section 5).
- Width 128-256, 4 layers, activation checkpointing per layer; FFT in fp32 at fast sizes.
- **Hidden samples are zeroed before this stage**, with a mask channel. The stage sees the whole crop,
  so masking after it would leak (3 reviewers).
- Interface: `LocalStage` (samples -> one feature vector per sample); a continuous-kernel
  convolution or the identity can replace the S4D stage.
- Throughput and memory measured 2026-10-29 (`scripts/bench_local_stage.py`, one L40S, a
  training step with bf16 autocast, checkpointing, 4 layers, 32 modes, mixing in 2 layers;
  median of the per-second rates; width 256 at half the batch, width 128 runs out of memory
  there at ~2 M samples):

  | case | width 128: M samples/s, peak GiB (samples) | width 256 |
  |---|---|---|
  | ECG 12 x 500 Hz x 10 s | 1.14, 19.0 (1.92 M) | 0.58, 18.9 (0.96 M) |
  | EEG 19 x 256 Hz x 30 s | 1.11, 24.1 (2.33 M) | 0.54, 24.1 (1.17 M) |
  | EEG 64 x 256 Hz x 30 s | 1.18, 20.3 (1.97 M) | 0.55, 20.3 (0.98 M) |
  | HD-EMG 64 x 2048 Hz x 2 s | 1.43, 20.7 (2.10 M) | 0.69, 20.1 (1.05 M) |
  | PSG 30 s, 15 channels at 1-512 Hz | 1.37, 14.7 (1.52 M) | 0.76, 14.7 (0.76 M) |

  The stage is bound by memory traffic, not FLOPs: the rate halves and the memory doubles per
  sample from width 128 to 256 (~10-11 KB a sample at width 128). One ECG step: 2.40 s with a
  gather per layer, 1.65 s with the contiguous bucket layout and the memory-efficient attention
  kernel.

## 4. Length reduction: a global token duration, learned chunking as the end state

- Now: **one global token duration in seconds** for every channel and rate (the same grid at every
  rate), swept on the ladder. **(built)**: `tokens.token_s`. A channel's tokens are the cells of
  the grid that hold at least one of its samples: a channel slower than one sample a cell has one
  token a sample, not empty cells (to confirm).
- End state: a **learned chunker** (H-Net type) that places the boundaries from the local features.
- Interface **(built)**: encoder -> `boundaries(local features, visible only)` -> global model ->
  upsampling + residual. The fixed-duration chunker is the first implementation; an H-Net-type
  chunker plugs in at the same place. Before building one: measure its possible gain (peak vs
  mean local information rate) with no training.
- Pooling: attention pooling over the token window with 50 % overlap.
- The local residual carries the timing finer than one token.

## 5. Channel descriptors (metadata when present, nothing when absent)

**First runs:** the signal type only (one embedding), with the random slot and the level and
scale. The fields below need index columns and come after the first runs; remote-sensing models
(DOFA, Panopticon) condition the input embedding on such metadata, a candidate then.

A descriptor is a **set of optional (field, value) pairs**; each present field adds one embedding,
an absent field adds nothing. No "unknown" special case is needed.

| field | value | example |
|---|---|---|
| type | category | ecg |
| ECG electrode weights | vector over (RA, LA, LL, V1-V6, other) | I = LA - RA; V1 = V1 - Wilson terminal |
| ECG lead placement | category | standard, Mason-Likar, monitoring (CM5, MCL1) |
| EEG electrodes | Fourier features of xyz (10-05 template) of the active and the reference electrode, 0.25 cm noise | Fp1-avg |
| reference | category | average, linked ears, bipolar, as recorded |
| grid position | (row, column, inter-electrode mm, muscle) | HD-EMG |
| calibration | physical / site-calibrated / unknown | |
| device filters | high-pass, low-pass, notch (Hz) when known | 0.5-40 Hz monitor ECG |

- Field dropout 20-30 % per field in pretraining, so the model works with any subset.
- Shortcut check: a dataset-identity probe on the descriptors alone (diagnostic, not a result).
- Needs new index columns (standard name, electrode, reference, filters, site).
- **(built)**: the channel table has the type and an empty placeholder for the optional fields.

## 6. Global model

- Transformer over the global tokens of all channels; rotary embedding on continuous time in
  seconds, angles in **float64**, wavelengths log-spaced with enough frequencies per decade at
  0.1-1 s (QRS, RR).
- Full attention. Examples are capped at 4-8 k global tokens: random channel subsets (k uniform in
  1..C, then a random subset; for grids and EEG spatially coherent subsets), and crop length.
  Channel subsets also train "any subset at test time".
- Montages above ~24 channels: per time window, fuse channels into 8 latents (Perceiver/LUNA type).
  Later, not in the first version.
- Register tokens (4). QK-norm.
- Long context (monitoring ECG 30 s-5 min, sleep context, ICU hours): a second reduction to summary
  tokens per window. Second version.

## 7. Pretraining objective (fixed while the architecture is selected)

**Built (biosignal_model.v4):** data2vec 2.0 type. The student sees only the visible tokens; an
EMA teacher reads the unmasked input; the target of a hidden token is the mean of the teacher's
top-K block outputs, each layer-normalized; a narrow decoder predicts it (MSE). Masks: **time
spans over all channels only**, hidden in whole grid cells. Single-token and random channel-group
masks leak again (ECG: II = I + III, so a hidden lead follows from the visible ones), so they stay
off until channel groups come from descriptors (which channels share electrodes). Tests:
tests/test_masking.py, tests/test_v4_model.py.

**Cycle combs (2026-11-04, ECG-only phase).** Spans leave the copy from another beat open: on
PTB-XL (500 Hz, 8 independent leads, fold 9) a plain copy of the beat before explains 88 % of a
QRS core's variance, the neighbour beats 93 %, the beat's own P wave, QRS onset and ST-T 60 %
(scratchpad qrs_sources.py); in v4-small-b's latent targets a ridge map from the same phase one
or two beats away plus the neighbours explains 0.27, the model 0.36. Harder spans alone (80 % in
0.2-2 s, v4-small-m80) lowered the copy share but gave worse PTB-XL features (10 % of the labels,
seed 0: 0.821 vs 0.845). A comb hides the same phase of every cycle (VideoMAE's tube, ST-MTM's
period comb): the period from the autocorrelation of the example's energy envelope (|first
difference|, a moving mean over one token), averaged over its channels (`cycle_period`; none
below `cycle_strength`: the share goes to spans), a uniform random phase, a tooth width a
uniform part of the period; combs stay whole (the last one gets a narrower tooth). On PTB-XL
(folds 1-8, 500 Hz): a period of strength >= 0.3 in 91 % of the records (non-AF 96 %), within
40 ms of the median RR in 91 % (the first difference's own autocorrelation: 37 %, noise
dominates it). A copy from the next detected beat finds a visible cell for 27 % of the
comb-hidden tokens (spans: 72 %; with the true median RR 22 %; teeth anchored at each detected
beat 9 %: the heart rate drifts within 10 s; anchored teeth would show the beat times in the
layout, a rhythm leak to test first). The layout shows the period and nothing more of the
content (the layout of another draw of the same design predicts the targets as well).

**The teacher's input does not change with the mask** (fixed 2026-11-04): the student's values
are transformed with its visible samples' mean and standard deviation (nothing hidden reaches
it); the teacher's with those of all valid samples. Before, the teacher used the student's, so
two masks of one example gave targets that differed by up to 7.7 (unit-variance targets).

The earlier plan: MAE type, asymmetric: the global model sees only visible tokens; a narrow
decoder (1/4 width, 4 blocks) gets the mask tokens.

- Masks, a set total ratio of the global tokens (start: 75 %), three kinds, their shares a setting
  (start: 50 / 30 / 20 %) **(built, on samples)**:
  - time spans hidden **over all channels together**;
  - channel groups: groups from a descriptor field when the channels have it, else random
    channel subsets, hidden over a time window;
  - single tokens.
- Span lengths: **one global rule**, log-uniform between two global bounds in seconds (for the
  time spans and, separately, for the channel-group windows). No per-type span table.
- The masks are applied to the **samples** (zeroed, with a hidden flag) before any model part.
- Targets: the values of section 2; **no per-token normalization** (amplitude carries diagnoses).
  ECG loss only on the independent leads (I, II, V1-V6). Loss per global token, weighted equally
  across types by the mixture, not by samples.
- No local residual into the decoder in pretraining (it would solve the task).
- Challengers on the ladder after the architecture is fixed: data2vec 2.0 (top-K teacher layers),
  BEST-RQ (no teacher).

## 8. Augmentations in the loader (the data layer stays native)

- **Rate augmentation**: in 50 % of examples, resample a channel to a rate drawn from the **rates
  present in the data for its type** (no band per type is set by hand). Trains rate invariance.
- **No decimation** of oversampled channels: every channel stays at its native rate.
- **EEG re-reference at random**: as recorded / average (>= 32 channels) / bipolar pairs, with the
  descriptor set to match. Exact linear maps; removes the unknown hospital reference as a cue.
- ECG: realistic filters (high-pass 0.05/0.5/0.67 Hz, low-pass 40/100/150 Hz, notch 50/60 Hz),
  baseline wander 0.1-0.5 Hz up to 1 mV, EMG and electrode-motion noise; EEG: mains injection. So
  device filters and mains do not identify the dataset.

## 9. How the architecture is selected (the ladder)

- Global model sizes 5, 15, 50 M non-embedding parameters (150 M on rented GPUs), muP, separate
  learning rate for the local stage, AdamW beta2 0.95, clip 1.0, no weight decay on SSM A and dt.
- WSD schedule with cooldown branches at 3 points per size.
- Metric 1 (does not depend on the tokenizer): masked prediction error in physical units per type
  and frequency band, on held-out subjects, with the same masks for every candidate.
- Metric 2: fine-tuning suite with 3 seeds and 1 / 10 / 100 % of the labels: PTB-XL all + QRS/QT/PR
  regression + PACE / WPW / LVH; a sleep task; an EEG task; PPG-DaLiA heart rate; an EMG task.
- Decide by slopes over sizes, never by one point. A proxy counts only if ranks agree at 2 sizes.
- The global token duration is one of the swept settings.

## 10. Before any pretraining: tests whose answers hold at scale

Correctness (unit tests): gradient of a hidden prediction with respect to hidden samples = 0;
packing invariance; channel permutation invariance without descriptors; rotary precision.

Architecture properties (small runs, no downstream ranking):
1. Trivial-solver baselines for each mask kind (spline in time, linear map from the other leads,
   neighbour interpolation): the masks must not be solvable by them.
2. Synthetic resolution: a 0.5 ms 2-10 mV pacing spike at 500 Hz and 1 kHz (timing +-2 ms), a
   20 ms notch, a +-5 ms shift; EEG spikes 20-70 ms, 1-3 Hz discharges, bursts at 5 and 50 uV.
   Autoencode through sections 3-4; error per band against the noise floor.
3. Rate invariance at 100 / 128 / 173 / 250 / 500 Hz on band-limited signals.
4. Real token counts, memory and throughput per type.
5. III ~ II - I after section 2 (linearity check).

Data analyses (pod CPU): i-care amplitude per hospital (site scales); hd-semg duplicates; effective
band per type (spectral knee); share of values in the nonlinear part of asinh per type.

## 11. Order of the code changes

1. Bugs and tests on the current code: float64 rotary angles, the tests in section 10. (done)
2. One benchmark interface (`features()`, `layer_groups()`); merge the three ptbxl.py copies; xresnet,
   S4 and the hybrid stay as references. (done)
3. Batch format: a flat sample buffer per channel, a channel table, a token table (replaces the
   [R, L, P] grid). Loader: two limits per span (local samples, global tokens), a span per group,
   caps per subject, the mixture in global tokens. (built; the old [R, L, P] path stays for the
   old configs)
4. Local stage (section 3) (built), reducer (section 4), descriptors (section 5, with the index
   columns), global model (section 6), MAE (section 7), heads.

# Biosignal foundation model: the v4 architecture

The design of the v4 model (revision 2, 2026-10-29). Revision 1 was reviewed by 8 experts (ECG,
EEG/EMG, information theory, LLM, vision, audio, foundation-model architect, code); this
revision takes their consensus and the user's decisions. Each change names its reason.
"Ladder" = the scaling ladder in section 9. The code is built in the order of section 11.

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
    subject per epoch (start: 4 h).
  - UniMax over (type x dataset), sized by subjects; at most 4 epochs of a dataset.
- The loss, the mixture and the compute budget count the same unit: **global tokens** (section 4).
  The v0-v3 loss is a mean over samples, so 2 kHz EMG weighs 8x EEG per second.
  mixture counts global tokens (`data.mixture.unit: global_tokens`).
- Quality masks (flat line, lead-off, clipping) from the quality layer: masked out of the loss.
- Checks before use: hd-semg raw and preprocessed files must not both be in the index; benchmark
  test subjects held out (exists for PTB-XL; check for every new task).

## 2. Input values

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

## 3. Local stage: every sample, channels mixed early

- Per channel, at its own rate: a bidirectional diagonal complex SSM (S4D), discretized by **ZOH**
  with dt = 1/fs in seconds (not bilinear: bilinear warps frequency 3-15 % between 100 and 500 Hz).
  One kernel per distinct rate in the batch; kernel length capped near 5 x the longest mode time.
- Mode time scales initialized log-uniform between two set bounds (start: 1 ms to 10 s); a mode is
  masked when |lambda| dt > pi (above Nyquist or decaying faster than one sample). Rates above any
  seen in training: the same mask at test time.
- **Early channel mixing** (the hybrid's lesson: attention across leads at every sample): in a set
  subset of the layers (start: 2 of 4), attention across co-located channels at each sample time,
  for channels of one example with a shared time base (ECG leads, EEG electrodes, PSG). For grids
  (HD-EMG) a 3 x 3 neighbour convolution instead, once grid positions are descriptors (section 5).
- Width 128-256, 4 layers, activation checkpointing per layer; FFT in fp32 at fast sizes.
- **Hidden samples are zeroed before this stage**, with a mask channel. The stage sees the whole crop,
  so masking after it would leak (3 reviewers).
- Interface: `LocalStage` (samples -> one feature vector per sample); a continuous-kernel
  convolution or the identity can replace the S4D stage.

## 4. Length reduction: a global token duration, learned chunking as the end state

- Now: **one global token duration in seconds** for every channel and rate (the same grid at every
  rate), swept on the ladder.
- End state: a **learned chunker** (H-Net type) that places the boundaries from the local features.
- Interface
  upsampling + residual. The fixed-duration chunker is the first implementation; an H-Net-type
  chunker plugs in at the same place. Before building one: measure its possible gain (peak vs
  mean local information rate) with no training.
- Pooling: attention pooling over the token window with 50 % overlap.
- The local residual carries the timing finer than one token.

## 5. Channel descriptors (metadata when present, nothing when absent)

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
-

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

MAE type, asymmetric: the global model sees only visible tokens; a narrow decoder (1/4 width,
4 blocks) gets the mask tokens.

- Masks, a set total ratio of the global tokens (start: 75 %), three kinds, their shares a setting
  (start: 50 / 30 / 20 %)
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
   caps per subject, the mixture in global tokens.
4. Local stage (section 3), reducer (section 4), descriptors (section 5, with the index
   columns), global model (section 6), MAE (section 7), heads.

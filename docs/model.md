# Model v0

The first model: its purpose is to prove that the whole chain works (index, loader,
tokens, training, logs, checkpoints) and to measure the throughput. It is not the final
architecture; each choice below is a starting point for experiments.

    biosignal-model train run.yaml

Every setting is in the YAML file; nothing has a default. Metrics go to Weights & Biases
(no artifacts: no checkpoints, data or code); checkpoints go to the store under
`run.checkpoint_prefix`. A run starts only from a committed git tree, and logs the commit.

## What the model sees

Only the samples and their times (see [training-layer.md](training-layer.md)). For each
token: its samples, a flag per sample (observed or not), the time of its centre and its
length in seconds, a channel slot drawn at random per example (it only says "same
channel"), and the scale of its channel in this example.

## Choices, and what to try instead

| Part | v0 | Open questions for experiments |
|---|---|---|
| Token | `samples_per_token` consecutive samples of one channel | a token of fixed duration; tokens cut where the signal changes (as in the Byte Latent Transformer) |
| Values | asinh((x - mean) / std) with the channel's mean and standard deviation in the example (from visible tokens), both given to the network; exact to reverse | a mean and scale for each token, predicted as extra targets; other encodings of scale |
| Time | rotary embedding on continuous time (s), wavelengths on a log scale | learned time features |
| Channels | random slot embedding per example | attention that is the same for any order of channels by design |
| Attention | within each example only (packed rows) | |
| Objective | hide `mask_ratio` of the tokens of each channel at random; predict their samples (mean squared error on observed samples) | hiding spans of time or whole channels; predicting in latent space (JEPA); autoregressive prediction |
| Network | transformer encoder: RMSNorm, SwiGLU | sizes, depth |

## Logged metrics

- `train/loss`, `train/grad_norm`, `train/learning_rate`
- `train/loss_by_dataset/<id>`, `train/loss_by_type/<type>`: the loss split by the
  dataset and the signal type of each token. They come from the example's origin, which
  the model never sees.
- `validation/loss` and its splits: the same first batches of the validation split each
  time, so values compare.
- `speed/tokens_per_s`, `speed/examples_per_s`, `speed/samples_per_s`,
  `speed/data_wait_fraction` (the part of the time spent waiting for data)
- `batch/padding_fraction`, `batch/hidden_fraction`, `batch/examples`
- `system/max_memory_gb`, and the system metrics that W&B records itself

## Benchmarks

    biosignal-model benchmark apnea bench.yaml

Each benchmark uses the official split and protocol of its task, and its test records are
held out of pretraining (`held_out` in index.yaml), so results compare with published ones.
For one checkpoint it reports:

| Name | What | What it answers |
|---|---|---|
| `fine_tuning` | the pretrained encoder and a head, every weight trained on the labels | the comparison with published results, which also train every weight |
| `from_scratch` | the same, from random weights | what pretraining adds to the same model |
| `linear_probing`, `linear_probing_random` | a linear probe on the frozen encoder, pretrained or random | a diagnostic: did pretraining learn anything a task can use |

**apnea**: PhysioNet apnea-ecg, a label (apnea or not) for each minute of one ECG lead at
100 Hz; learning set a01-a20, b01-b05, c01-c10, test set x01-x35 (PhysioNet/CinC
Challenge 2000). Scores over all test minutes: accuracy, sensitivity, specificity, AUC.

**ptbxl**: PTB-XL, 12-lead ECG of 10 s, the protocol of Strodthoff et al. 2021 (arXiv
2004.13701) and its code: the labels of each task (`all`, `diagnostic`, `subdiagnostic`,
`superdiagnostic`, `form`, `rhythm`) from the records' `scp_codes`; folds 1-8 train, 9
validation (settings and the epoch are chosen here), 10 test; macro-AUC on fold 10. The
paper's main tables use 100 Hz (`rate: 100`); `rate: 500` is the original, a different
input from the published one. No patient is in two folds, and folds 9 and 10 are held out
of pretraining.

**isruc**: ISRUC-Sleep cohort I (100 subjects, one night each), 5 sleep stages (W, N1, N2,
N3, REM) of the first expert, by subject: 1-80 train, 81-90 validation, 91-100 test (CBraMod,
arXiv 2412.07236, appendix E.1: "we set subject 1 to 80 as training set, subject 81 to 90 as
validation set" and "subject 91 to 100 as test set"; its code checked on 2026-10-29). Each
night in sequences of 20 epochs of 30 s, the rest of a night unused; each epoch through the
encoder, a Transformer layer over the 20 epoch features; the epoch of training chosen by the
validation kappa. Balanced accuracy, Cohen's kappa and weighted F1 over the test epochs.
`channels: eeg6` takes CBraMod's 6 EEG channels (F3-A2, C3-A2, O1-A2, F4-A1, C4-A1, O2-A1),
`channels: all` every channel of the recording; both unfiltered, in physical units at their
native rates (CBraMod filters 0.3-35 Hz and notches 50 Hz). Published on this split (CBraMod,
Table 7; mean ± std): CBraMod 0.7865 ± 0.0110 / 0.7442 ± 0.0152 / 0.8011 ± 0.0099, LaBraM-Base
0.7633 / 0.7231 / 0.7810, USleep 0.7586 / 0.7209 / 0.7805 (balanced accuracy / kappa /
weighted F1). Subjects 81-100 are held out of pretraining (index.yaml).

**dalia**: PPG-DaLiA (15 subjects, about 2.5 h each), the heart rate of each 8 s window
(shift 2 s; the ECG-based label of the dataset) from the wrist PPG (`inputs: ppg`) or the PPG
and the wrist accelerometer (`inputs: ppg+acc`). Leave one subject out (Reiss et al. 2019,
"Deep PPG", Sensors 19:3079): each subject the test subject once; two other subjects choose
the epoch (the run's `folds`). The score is the mean of the 15 subjects' mean absolute errors
in bpm (their "All"). Published on this protocol (their Table 10): CNN ensemble 7.65 ± 4.2
(PPG and accelerometer, time-frequency spectra), single CNN 8.82 ± 3.8, SpaMaPlus 11.06 ± 4.8.
Later supervised results on a 4-fold split of subjects (PULSE 4.03, KID-PPG 3.79, EnhancePPG
3.54; arXiv 2412.17860) and the PPG foundation models' linear probes (PaPaGei, MOMENT) use
other splits. Every subject is held out of pretraining (index.yaml).

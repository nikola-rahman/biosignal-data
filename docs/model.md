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
| `finetune` | the pretrained encoder and a head, every weight trained on the labels | the comparison with published results, which also train every weight |
| `supervised` | the same, from random weights | what pretraining adds to the same model |
| `probe`, `random` | a linear probe on the frozen encoder, pretrained or random | a diagnostic: did pretraining learn anything a task can use |

**apnea**: PhysioNet apnea-ecg, a label (apnea or not) for each minute of one ECG lead at
100 Hz; learning set a01-a20, b01-b05, c01-c10, test set x01-x35 (PhysioNet/CinC
Challenge 2000). Scores over all test minutes: accuracy, sensitivity, specificity, AUC.

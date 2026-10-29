# The training layer

The training layer is a view: it is computed while a model trains, and only its index is
stored. The signals stay in the standard layer; a data loader reads windows from its
blocks and applies the transforms of the experiment (window length, sampling rate,
normalization, tokenization). A change to those is a change of code, not a rewrite of data.

## The index (stored)

    index/<date>/
        recordings.parquet   every recording with signals: dataset, subject, split, duration
        channels.parquet     every channel: name as written, signal type, what decided it
        _manifest.json       the standard versions used, the rules, counts, unknown names
        _SUCCESS

Build it with:

    biosignal-data index --all --dest s3://biosignal-data

Only complete standard versions are indexed; the newest of each dataset. An index is
immutable once complete (`--replace` makes it again).

### Subjects and splits

A split is made by person, so that no person is in two splits. The rules are in
[`index.yaml`](../index.yaml), per dataset, each with its source:

- `pattern`: a regex on the recording id; group 1 is the subject (e.g. `chb01/chb01_03`).
- `table`: a table of the raw dataset gives the subject of each recording (ephnogram's
  `ECGPCGSpreadsheet.csv`); `pattern` group 1 is the recording's key in the table.
- `one_per_recording`: the documentation says each recording is a different person.
- `unknown`, or no entry: each recording is its own subject, marked as not certain.
- `same_person`: subject ids that the documentation says are one person.

`exclude` leaves recordings out of the index: copies of, or extracts from, other
recordings of the same dataset, which would otherwise be in the data twice, maybe in two
splits. Each needs a quote (`evidence`) or a sample comparison (`checked`): the 23
`x_mitdb` records of mitdb are the first 10 minutes of the records of the same number, and
apnea-ecg `c05` is `c06` from 80 s on. The manifest counts what was left out, per reason.

Subject ids are qualified by their dataset (`physionet/chbmit:chb01`). The split of a
subject comes from a SHA-256 of that id: 80 % train, 10 % validation, 10 % test. It never
changes, whatever is added to the index later.

### Signal types

Each channel gets one of `ecg`, `eeg`, `eog`, `emg`, `resp`, `spo2`, `ppg`, `bp`, `hr`,
`temp`, `position`, `accel`, `scg`, `pcg`, `sound`, `other`, or `unknown`.
`signal_type_from` says what decided it: `dataset` (a channel rule of the dataset in
index.yaml), `name` (a general rule in `src/biosignal_data/index/signals.py`) or `none`.
A name no rule matches stays `unknown`, never a guess; the manifest lists those names.

## The data loader (computed)

**The model gets the samples and their times, and nothing else**: no channel names, no
signal types, no units, no dataset or subject. The index keeps those for us, to select
data and to analyse results; they never go into a batch.

`biosignal_data.training`:

    selection = Selection.load(store, "2026-10-20", split="train", signal_types=None)
    source = ExampleSource(selection, store_factory, cache_directory=..., cache_bytes=...,
                           limit=SampleAndTokenLimit(max_samples=400_000, max_tokens=8192,
                                                     token_s=0.04),
                           spans="per_group", subject_hours_per_epoch=4.0, max_channels=32,
                           block_s=1800.0, shuffle_buffer=4096, reads_in_flight=8, seed=0,
                           passes=None)
    for example in source.examples(epoch, part, parts):
        example.signals  # the model input
        example.origin   # for analysis only

An example is a set of signals from one recording and one time span. Each signal is one
channel at its native rate:

- `values`: physical values (stored integer x scale + offset) as float32; the stored
  integers if the channel has no calibration; NaN where a sample is invalid. Units stay as
  the source wrote them.
- `sampling_rate`, and `start_s`: the time of the first sample from the start of the
  example (relative time, never the date or the clock time).

The signals are in random order, so the order of channels in a file (a montage
convention) means nothing. Nothing is resampled, filtered, re-referenced or normalized.

**How a recording becomes examples.** A time cursor moves through the recording. At each
step the selected channels are put in random order and cut into groups of random size. The
longest span of a group comes from `limit`:

- `SampleTokenLimit` (v0-v3): `max_tokens` tokens of `samples_per_token` samples;
- `SampleAndTokenLimit` (v4, [architecture](architecture.md)): at most `max_samples` samples
  (what the local stage reads) and at most `max_tokens` global tokens (cells of `token_s`
  seconds that hold a sample).

With `spans: shared` (v0-v3) every group is one example over the same span, the shortest of
the groups' spans; with `per_group` each group cuts spans as long as its own limit allows, so
a group with many fast channels covers a short time and one slow channel a long time, also in
the same step. The first step of a block is a random part of a full one, so the cuts differ
from epoch to epoch. Every sample of every selected channel of a block read is in exactly one
example per reading.

`subject_hours_per_epoch` caps what one subject gives to one pass over its dataset: random
blocks, drawn anew in each epoch and pass, up to that many hours (none: no cap). The mixture
(`training.mixture`) counts each dataset in the unit of the limit (sample tokens or global
tokens) after this cap.

The limits and the token size are settings of the model, given to the sampler; the tokenizer
itself is part of the model.

**How it reads.** The block index of each standard version (`blocks.parquet`) is loaded
once. A span is one ranged read per block it touches (8 MiB at most before compression),
through a local disk cache of blocks (`BlockCache`); several recordings are read at the
same time, because one read from the S3 storage takes 80-240 ms; examples go through a
shuffle buffer. Memory stays bounded, whatever the length of a recording. The recordings are dealt
over `parts` (data loader workers times distributed ranks) after a shuffle fixed by `seed`
and `epoch`. `TorchExamples` in `biosignal_data.training.torch_dataset` wraps a source as
a PyTorch `IterableDataset` (install the `train` extra).

**Evaluation** uses the official split and protocol of each benchmark, so that results
compare with published ones; the 80/10/10 split above is for pretraining only.

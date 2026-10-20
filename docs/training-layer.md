# The training layer

The training layer is a view: it is computed while a model trains, and only its index is
stored. The signals stay in the standard layer; a data loader reads windows from its Zarr
arrays and applies the transforms of the experiment (window length, sampling rate,
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

`biosignal_data.training`:

    selection = Selection.load(store, "2026-10-20", split="train", signal_types={"ecg", "eeg"})
    source = WindowSource(selection, store_factory, cache_directory=..., cache_bytes=...,
                          window_s=30.0, min_window_s=10.0, shuffle_buffer=4096,
                          reads_in_flight=8, seed=0)
    for window in source.windows(epoch, part, parts): ...

A window is one array (one sampling rate) of one recording, at its native rate, in
physical units (float32, NaN where a sample is invalid or a channel has no calibration),
with the name, signal type and unit of each channel. Nothing is resampled or normalized:
those are choices of the experiment, made in its own code.

A unit of work is one array of one recording. Its shards (64 MiB at most) are read whole
through a local disk cache (`WholeObjectCache`) and cut into windows from a random offset
per epoch; pieces shorter than `min_window_s` at the ends are left out, and a recording
shorter than one window is one window. Several units are read at the same time, because
one read from the S3 storage takes 80-240 ms; the windows go through a shuffle buffer.
Memory stays bounded, whatever the length of a recording.

The units are dealt over `parts` (data loader workers times distributed ranks) after a
shuffle fixed by `seed` and `epoch`. `TorchWindows` in
`biosignal_data.training.torch_dataset` wraps a source as a PyTorch `IterableDataset`
(install the `train` extra).

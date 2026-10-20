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
- `one_per_recording`: the documentation says each recording is a different person.
- `unknown`, or no entry: each recording is its own subject, marked as not certain.
- `same_person`: subject ids that the documentation says are one person.

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

Reads whole shards (64 MiB) of the Zarr arrays and cuts many windows from each, through
a shuffle buffer, with a local disk cache: one read from the S3 storage takes 80-240 ms,
so windows are never fetched one by one.

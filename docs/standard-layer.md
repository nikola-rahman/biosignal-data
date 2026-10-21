# The standard layer

The standard layer holds the recordings of the raw layer in one form: samples as compressed
blocks in large objects, metadata as Parquet tables. It is made from the raw layer only,
never from a source.

## Rules

- **Exact.** Signals are the integers the source stored, with a scale and an offset per
  channel (`physical = stored * scale + offset`). Nothing is filtered, resampled, rereferenced,
  cut or renamed. Every block is read by two independent readers of the raw files, which
  must agree. (One reader alone is not enough: pyedflib, used before, returned another
  file's samples without an error.) Every block is decoded again and compared with the
  source before it is kept; every object is uploaded with its MD5 (`Content-MD5`), which
  the storage checks against what it received. The channels table has the SHA-256 of each
  channel's stored samples, so exactness can be checked again without the raw files.
- **Native sampling rates.** The channels of a recording are grouped by sampling rate; each
  group is one array (e.g. `250Hz`). Choosing a common rate is a decision for training, not
  for storage.
- **Source names.** Channel names, units and labels are kept as the source wrote them. The
  `channels` table has a `standard_name` column for a mapping added later.
- **Lineage.** Every recording lists the raw files it was made from, with their SHA-256.
  Every raw file of the version is either used or listed as not converted, with the reason.
- **Immutable.** A version is complete when `_SUCCESS` exists, and is not changed after that.

## Layout

    standard-blocks/<repository>/<dataset>/<raw version>/
        recordings.parquet     one row per recording
        channels.parquet       one row per channel
        events.parquet         one row per annotation or event
        blocks.parquet         one row per block: where it is and what it holds
        data/<batch>-<n>.bin   the blocks, one after another, at most 64 MiB per object
        _manifest.json         lineage, files not converted and why, the layout settings
        raw_manifest.json      the raw version's manifest: every file with its SHA-256
        files.parquet          the raw files that hold no signals, up to 16 MiB each, packed
        files/pack-<n>.bin     as published into objects of at most 64 MiB, one row each in
                               files.parquet (path, object, offset, nbytes, sha256)
                               (tables such as ptbxl_database.csv, documentation)
        _parts/                progress of an unfinished run (removed on success)
        _SUCCESS               written last

The `<raw version>` is the raw version the data was made from. A record whose signals the
source does not publish (only its annotations) is still a recording: it has no blocks, and
its events are kept. The recording id is the path of its raw file without the extension
(`100` for `100.hea`, `S001/S001R01` for `S001/S001R01.edf`).

`standard/` holds the earlier layout (one Zarr v3 group per recording), kept until the
block layout replaces it.

## Blocks

A block is one span of time of one recording's channels at one sampling rate: an array
`(channels, samples)` in the smallest integer type of the source format (int16 for EDF and
most WFDB formats, int32 for BDF and WFDB formats 24 and 32), compressed with Blosc (zstd
level 9, bit shuffle; lossless). On mitdb, chbmit and sleep-edfx it was 4-17 % smaller than
delta + zstd on EEG and PSG and 2 % larger on ECG.

- **Block:** at most 8 MiB before compression (a whole number of WFDB frames). A 10 s ECG is
  one block of 24 KB; a 23-channel EEG at 256 Hz has blocks of about 11 minutes.
- **Object:** blocks of the recordings of one conversion batch, one after another, closed
  before 64 MiB. Many short recordings share one object.
- **Index:** `blocks.parquet` gives each block's recording, array, first sample, samples,
  channels, type, object, byte offset, byte length and CRC-32. A reader loads it once.

Why: each request to the S3 storage costs 80-240 ms, whatever its size. The earlier layout
(a Zarr group per recording) made three objects of a few KB for a 10-second ECG, and a
reader paid four requests before the first sample: reading PTB-XL at 100 Hz (0.5 GB) took
87,000 requests and 30 minutes. This layout is how large training corpora are stored
(Megatron-LM indexed datasets: one large file and an offset index; MosaicML Streaming:
64 MiB shards); 8 MB is the usual size of an S3 byte-range read (AWS). A span is one ranged
GET per block; a whole dataset is about one GET per object.

## Formats

| Source | Recognised by | Read with |
| --- | --- | --- |
| WFDB record | the `RECORDS` lists, or every `.hea` file where there is no `RECORDS` | `wfdb` |
| WFDB annotation | `<record>.<annotator>`, annotators from `ANNOTATORS` | `wfdb` |
| EDF, EDF+, BDF | `.edf`, `.bdf` (and `RECORDS` entries naming them) | `edfio` |

Not converted yet, so listed in the manifest with the reason: multi-segment WFDB records,
discontinuous EDF+ (EDF+D), and every other format. Metadata files (CSV, TSV, JSON) are kept
in the raw layer and listed as not converted.

## Pipelines

The raw version is a build input. Each dataset can take its own way to it, from shared
parts:

- **raw layer** (default): the raw version in `<dest>/raw/`, one object per published file
  (see raw-layer.md). Good for sources with large files.
- **physionet-zip** (`--from-zip`): the version's ZIP from PhysioNet's AWS copy
  (`s3://physionet-open/<slug>/<slug>-<version>.zip`, no credentials), staged on local disk:
  checked against its S3 ETag, every member against SHA256SUMS.txt (and the ZIP's CRC-32),
  files that PhysioNet added later on the copy (e.g. apnea-ecg's test answers) copied and
  checked by size; an incomplete ZIP is refused (challenge-2018 1.0.0). PTB-XL: about 220
  requests for 1.8 GB instead of 87,000; planning from local disk takes seconds. On
  2026-10-20 it gave the same samples (SHA-256 of every channel) and events as the raw layer
  for PTB-XL, mitdb and apnea-ecg.

Because `raw_manifest.json` and `files/` stay with the standard version, the raw version can
be deleted once the standard version is checked: the index reads its subject and held-out
tables from `files/`, the benchmarks their labels. Making a version again means staging the
raw version again.

## Run it

    biosignal-data standardize physionet/mitdb --dest s3://biosignal-data
    biosignal-data standardize --repository physionet --dest s3://biosignal-data --dry-run

`--dest` holds both layers: the raw version is read from `<dest>/raw/`, the result is
written to `<dest>/standard-blocks/`. Without `--version`, the newest complete raw version is used
and the run says so. A stopped run continues where it stopped.

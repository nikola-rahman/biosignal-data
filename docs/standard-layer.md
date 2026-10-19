# The standard layer

The standard layer holds the recordings of the raw layer in one form: signals as Zarr v3
arrays, metadata as Parquet tables. It is made from the raw layer only, never from a source.

## Rules

- **Exact.** Signals are the integers the source stored, with a scale and an offset per
  channel (`physical = stored * scale + offset`). Nothing is filtered, resampled, rereferenced,
  cut or renamed. Every array is read back after writing and compared with the source.
- **Native sampling rates.** The channels of a recording are grouped by sampling rate; each
  group is one array. Choosing a common rate is a decision for training, not for storage.
- **Source names.** Channel names, units and labels are kept as the source wrote them. The
  `channels` table has a `standard_name` column for a mapping added later.
- **Lineage.** Every recording lists the raw files it was made from, with their SHA-256.
  Every raw file of the version is either used or listed as not converted, with the reason.
- **Immutable.** A version is complete when `_SUCCESS` exists, and is not changed after that.

## Layout

    standard/<repository>/<dataset>/<raw version>/
        recordings.parquet     one row per recording
        channels.parquet       one row per channel
        events.parquet         one row per annotation or event
        signals/<recording>.zarr/
            <rate>Hz           int16 or int32, shape (channels, samples)
        _manifest.json         lineage, files not converted and why, the layout settings
        _parts/                progress of an unfinished run (removed on success)
        _SUCCESS               written last

The `<raw version>` is the raw version the data was made from. The recording id is the
path of its raw file without the extension (`100` for `100.hea`, `S001/S001R01` for
`S001/S001R01.edf`).

## Arrays

Each array is `(channels, samples)`, in the smallest integer type of the source format
(int16 for EDF and most WFDB formats, int32 for BDF and WFDB formats 24 and 32), compressed
with Blosc (zstd level 9, bit shuffle). It is lossless, and it is a codec of the Zarr v3
specification, so any Zarr v3 reader can open the arrays. On mitdb, chbmit and sleep-edfx it
was 4-17 % smaller than delta + zstd on EEG and PSG and 2 % larger on ECG; delta is not in
the specification.

- **Chunk:** all channels, a whole number of 30-second epochs, at least 64 KiB
  uncompressed. A channel at 1 Hz therefore gets chunks of several hours, not of 30 samples.
- **Shard:** as many chunks as fit in 64 MiB uncompressed. One shard is one S3 object.

Array attributes repeat the channel names, units, scales and offsets, so an array can be
read without the tables. The group metadata is consolidated, so opening a recording is one
request.

## Formats

| Source | Recognised by | Read with |
| --- | --- | --- |
| WFDB record | the `RECORDS` lists, or every `.hea` file where there is no `RECORDS` | `wfdb` |
| WFDB annotation | `<record>.<annotator>`, annotators from `ANNOTATORS` | `wfdb` |
| EDF, EDF+, BDF | `.edf`, `.bdf` (and `RECORDS` entries naming them) | `pyedflib` |

Not converted yet, so listed in the manifest with the reason: multi-segment WFDB records,
discontinuous EDF+ (EDF+D), and every other format. Metadata files (CSV, TSV, JSON) are kept
in the raw layer and listed as not converted.

## Run it

    biosignal-data standardize physionet/mitdb --dest s3://biosignal-data
    biosignal-data standardize --repository physionet --dest s3://biosignal-data --dry-run

`--dest` holds both layers: the raw version is read from `<dest>/raw/`, the result is
written to `<dest>/standard/`. Without `--version`, the newest complete raw version is used
and the run says so. A stopped run continues where it stopped.

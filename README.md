# biosignal-data

A catalog of public biosignal datasets (EEG, iEEG, ECG, EMG, sleep PSG, PPG, MEG, fNIRS, ...)
and the code to copy them, unchanged, into the **raw layer** of an S3 bucket.

- **Datasets**: 873 of them. [`catalog.yaml`](catalog.yaml) lists what a download needs for
  each (source, identifier, version, license); [`docs/datasets.yaml`](docs/datasets.yaml)
  describes them (signals, access, links, sizes). Browse them in
  [`docs/datasets/`](docs/datasets/README.md).
- **Download**: each dataset is planned (version and file list, read once), then copied
  complete and unchanged, its files in parallel, to
  `s3://<bucket>/raw/<dataset id>/<version>/`, with checksums verified and a manifest.
  Interrupted runs resume.
- **Read**: open the raw files with the established library for each format
  (MNE, mne-bids, wfdb, pynwb, h5py, ...), with local caching.

Nothing is converted or standardized yet. The raw layer is a faithful, versioned copy of each
source, which later layers can build on.

## Quick start

```bash
uv sync                                    # Python 3.11+, installs everything
uv run biosignal-data list --signal ecg
uv run biosignal-data info physionet/mitdb
uv run biosignal-data download physionet/mitdb --dest ./lake --dry-run  # the plan: files, size
uv run biosignal-data download physionet/mitdb --dest ./lake            # copy it
uv run biosignal-data download --all --dest s3://my-bucket              # everything
uv run biosignal-data status --all --dest s3://my-bucket                # what is stored
uv run biosignal-data check --all --latest --dest s3://my-bucket        # sizes vs manifests
uv run biosignal-data import <id> <folder> --version <label> --dest s3://my-bucket
```

| Command | What it does |
| --- | --- |
| `list` | find datasets by their descriptions |
| `info <id>` | one dataset: what it is, how to get access, where it is downloaded from |
| `download` | copy datasets from their sources, unchanged, to `--dest` |
| `import` | copy a dataset you downloaded by hand, from a folder |
| `status` | what is stored in `--dest` |
| `check` | stored files against their manifests (`--checksums` re-reads every file) |

- **Which datasets** (`download`, `status`, `check`): ids, `--ids-file`, `--signal` /
  `--repository`, or `--all`, exactly one of these.
- **Settings** that do not change between runs are in [`biosignal-data.yaml`](biosignal-data.yaml):
  the S3 storage (endpoint, region, addressing), parallel downloads, the staging folder for
  TUH, and the read cache. Nothing has a default.
- **Complete datasets only:** the raw layer holds each dataset as its source publishes it.
  A copy in which no file can hold recordings (only metadata) is reported, not counted as
  done.
- **Versions:** a version in the catalog is downloaded exactly, or the download fails; an
  entry without one gets the latest, with a warning ([`docs/access.md`](docs/access.md)).
- **Secrets** come only from `BIOSIGNAL_*` variables: `BIOSIGNAL_ACCESS_KEY_ID` and
  `BIOSIGNAL_SECRET_ACCESS_KEY` for your storage, and one or two per gated source
  ([`docs/access.md`](docs/access.md)).
- **Datasets that need you first** (credentials, download by hand) are skipped with a
  warning and listed at the end, with a warning for each source type whose code did not run.
- **Warnings** show every choice the tool makes and everything it skips. Every run prints
  where the data flows: from the sources through the machine that runs the command to
  `--dest`.

Reading back:

```python
from pathlib import Path
from biosignal_data.read import RawDataset, read_wfdb

ds = RawDataset.open(
    "./lake", "physionet/mitdb", "latest", cache_dir=Path("~/.cache/biosignal-data")
)
ds.local_paths("100.*")  # 100.hea, 100.dat, 100.atr, ...
record = read_wfdb(ds.local_path("100.hea"))
```

See [`docs/reading.md`](docs/reading.md).

## How a dataset gets in

| Source | How | Access |
| --- | --- | --- |
| PhysioNet | files and SHA-256 from each version's `SHA256SUMS.txt`; open data from PhysioNet's AWS copy where it has the files, else physionet.org; credentialed data from physionet.org | open or PhysioNet credentialed |
| OpenNeuro | OpenNeuro's public S3 bucket (latest snapshot) | open |
| NEMAR | official `nemar-py` client, version-pinned with checksums | open |
| DANDI | official `dandi` client | open |
| Zenodo, figshare | public REST APIs, MD5 checksums | open |
| Hugging Face | official `huggingface_hub` client, pinned to a commit | open or token |
| BDSP (Harvard) | credentialed S3 access points | approval + DUA |
| NSRR (sleepdata.org) | the web API of the official `nsrr` gem, MD5 checksums | approval |
| Kaggle | official Python SDK (`kagglesdk`) | account |
| TUH EEG | `rsync` over SSH into a staging folder, then copied | approval |
| Everything else | download by hand, then `biosignal-data import` | varies |

## Layout of the raw layer

```
s3://<bucket>/raw/<repository>/<dataset>/<version>/
    data/...                 files exactly as the source names and nests them
    _manifest.json           every file with size, SHA-256, MD5 and the source's checksum
    _SUCCESS                 written last: the version is complete and immutable
```

Details and the manifest schema: [`docs/raw-layer.md`](docs/raw-layer.md).

## The standard layer

`biosignal-data standardize <id> --dest s3://<bucket>` converts a raw version: signals as
Zarr v3 arrays at their native sampling rates with the exact stored values, metadata as
Parquet tables, every raw file either used or listed with the reason it was not.

```
s3://<bucket>/standard/<repository>/<dataset>/<raw version>/
    recordings.parquet, channels.parquet, events.parquet
    signals/<recording>.zarr/<rate>Hz
    _manifest.json, _SUCCESS
```

Details: [`docs/standard-layer.md`](docs/standard-layer.md).

## Repository map

```
biosignal-data.yaml     settings that do not change between runs (no defaults)
catalog.yaml            what a download needs, per dataset
docs/datasets.yaml      the description of each dataset, for people
docs/                   design notes; docs/datasets/ is generated from the two files
scripts/                import_eegdash.py (OpenNeuro/NEMAR entries), check_each.py (test runs)
src/biosignal_data/
    catalog/            schemas and loader of the two files
    fetchers/           one module per kind of source
    storage/            raw-layer layout, manifest, S3 and local stores
    read/               raw-layer browsing and format openers
    ingest.py           plan and copy one dataset: verify, checkpoint, resume
    batch.py            many datasets at once, one shared download pool
    paths.py            globs, and which files can hold recordings
    config.py           biosignal-data.yaml
    console.py          progress display and logging
    cli.py              the `biosignal-data` command
tests/
```

## Status

The code is tested against mocked sources, and real downloads have been tested for a part
of the catalog. Each description carries `download_verified`; set it to `true` once a
complete copy of that dataset has been checked. Known open points are listed in
[`docs/open-questions.md`](docs/open-questions.md).

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
uv run biosignal-data dev docs    # after changing catalog.yaml or docs/datasets.yaml
uv run pre-commit install         # run the checks on every commit
```

Adding or fixing a dataset: [`docs/adding-a-dataset.md`](docs/adding-a-dataset.md).

# biosignal-data

A catalog of public biosignal datasets (EEG, iEEG, ECG, EMG, sleep PSG, PPG, MEG, fNIRS, ...)
and the code to copy them, unchanged, into the **raw layer** of an S3 bucket.

- **Catalog**: 873 datasets, one YAML file each under [`catalog/`](catalog), saying where the
  data lives, how to get access, and how big it is. Browse it in
  [`docs/datasets/`](docs/datasets/README.md).
- **Ingest**: one command per dataset copies every file from its source to
  `s3://<bucket>/raw/<dataset id>/<version>/`, verifies checksums, and writes a manifest.
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
uv run biosignal-data download physionet/mitdb --dest ./lake --limit 5   # 5 recordings, for tests
uv run biosignal-data download physionet/mitdb --limit 5 --dry-run      # what that would download
uv run biosignal-data check physionet/mitdb --dest ./lake --latest

# S3 storage: settings in biosignal-data.yaml, keys in BIOSIGNAL_ACCESS_KEY_ID / _SECRET_ACCESS_KEY
uv run biosignal-data download --all --dest s3://bucket
```

- **Which datasets:** ids, `--ids-file`, `--signal`, `--repository` or `--all`. Many
  datasets download at the same time, sharing `--workers` parallel file downloads.
- **Where to:** `--dest` (required) takes a local folder or `s3://bucket[/prefix]`. For an S3
  destination, the storage's endpoint, region and addressing style come from
  [`biosignal-data.yaml`](biosignal-data.yaml) (no defaults), and the keys only from
  `BIOSIGNAL_ACCESS_KEY_ID` and `BIOSIGNAL_SECRET_ACCESS_KEY`.
- **Test subsets:** `--limit N` copies the first N recordings (each with its companion files
  and the metadata it needs); `--include <glob>` only matching paths. The version stays
  incomplete; a later run without them continues from it.
- **Datasets that need you first** (credentials, download by hand) are skipped with a
  warning and listed at the end. Each dataset's page in
  [`docs/datasets/`](docs/datasets/README.md) lists the steps, and
  [`docs/access.md`](docs/access.md) explains each repository's credentials.
- **Warnings** show every choice the tool makes for you and everything it skips.

## Where the data flows

The data passes through the machine that runs the command; every run prints this path at
the start. To keep it off your computer, run the command on another machine.

Reading back:

```python
from biosignal_data.read import RawDataset, read_wfdb

ds = RawDataset.open("s3://my-bucket", "physionet/mitdb")  # latest complete version
ds.local_paths("100.*")  # 100.hea, 100.dat, 100.atr, ...
record = read_wfdb(ds.local_path("100.hea"))
```

See [`docs/reading.md`](docs/reading.md).

## How a dataset gets in

| Source | How | Access |
| --- | --- | --- |
| PhysioNet | HTTPS, files and SHA-256 from each version's `SHA256SUMS.txt` | open or PhysioNet credentialed |
| OpenNeuro | OpenNeuro's public S3 bucket (latest snapshot) | open |
| NEMAR | official `nemar-py` client, version-pinned with checksums | open |
| DANDI | official `dandi` client | open |
| Zenodo, figshare | public REST APIs, MD5 checksums | open |
| Hugging Face | official `huggingface_hub` client, pinned to a commit | open or token |
| BDSP (Harvard) | credentialed S3 access points | approval + DUA |
| NSRR (sleepdata.org) | official `nsrr` Ruby gem | approval |
| TUH EEG | `rsync` over SSH | approval |
| Kaggle | official `kaggle` CLI | account |
| Everything else | download by hand, then `download --from-dir` | varies |

## Layout of the raw layer

```
s3://<bucket>/raw/<repository>/<dataset>/<version>/
    data/...                 files exactly as the source names and nests them
    _manifest.json           every file with size, SHA-256, MD5 and the source's checksum
    _SUCCESS                 written last: the version is complete and immutable
```

Details and the manifest schema: [`docs/raw-layer.md`](docs/raw-layer.md).

## Repository map

```
catalog/                one YAML per dataset (the source of truth)
docs/                   design notes; docs/datasets/ is generated from the catalog
scripts/                import_eegdash.py regenerates OpenNeuro/NEMAR entries
src/biosignal_data/
    catalog/            schema and loader
    fetchers/           one module per kind of source
    storage/            raw-layer layout, manifest, S3 and local stores
    read/               raw-layer browsing and format openers
    ingest.py           copy, verify, checkpoint, resume
    cli.py              the `biosignal-data` command
tests/
```

## Status

The code is tested against mocked sources; **no dataset has been downloaded completely end to end yet**.
Every catalog entry carries `download_verified: false` until someone does. Set it to `true`
in the entry once a download of it succeeds. Known open points are listed in
[`docs/open-questions.md`](docs/open-questions.md).

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
uv run biosignal-data dev docs    # after changing the catalog
uv run pre-commit install         # run the checks on every commit
```

Adding or fixing a dataset: [`docs/adding-a-dataset.md`](docs/adding-a-dataset.md).

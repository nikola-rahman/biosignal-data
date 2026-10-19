# Adding or fixing a dataset

The catalog is the source of truth. Each dataset is one YAML file at `catalog/<id>.yaml`, and
the file path must match the `id`.

## Choose the id

`<repository>/<the repository's own identifier>`, all of it taken from the source, nothing
invented:

| Where it lives | id |
| --- | --- |
| PhysioNet | `physionet/<slug>` e.g. `physionet/ptb-xl` |
| OpenNeuro | `openneuro/<accession>` e.g. `openneuro/ds004789` |
| NEMAR | `nemar/<accession>` e.g. `nemar/nm000103` |
| DANDI | `dandi/<dandiset>` e.g. `dandi/000055` |
| Zenodo / figshare | `zenodo/<record>`, `figshare/<article or collection>` |
| Hugging Face | `huggingface/<owner>/<repo>` |
| Kaggle | `kaggle/<competition or dataset slug>` |
| NSRR / BDSP / TUH | `nsrr/<slug>`, `bdsp/<project slug>`, `isip/<corpus folder name>` |
| No repository | `<host organisation>/<dataset name as published>`, lowercase, hyphens |

## Write the entry

```yaml
id: physionet/mitdb
name: MIT-BIH Arrhythmia Database
signals: [ecg]                     # free-form, lowercase; see existing entries for usage
source:
  type: physionet                  # selects the fetcher; fields depend on the type
  slug: mitdb
  version: 1.0.0                   # pin when you can; omit for "latest at download time"
access:
  tier: open                       # open | registration | application | credentialed | unavailable
  license: ODC-By 1.0
  steps: []                        # manual steps before a download can run, in order
  credentials: []                  # environment variables the fetcher reads
homepage: https://physionet.org/content/mitdb/
stats:                             # informational only
  subjects: 47
  recordings: 48
  hours: 24
  hours_estimated: false
related: []                        # other catalog ids this overlaps with or contains
notes: []
download_verified: false           # set true after a successful complete download
```

The fields of each `source.type` are defined in `src/biosignal_data/catalog/models.py`.
Use `type: manual` with a `url` when there is no programmatic download; the data then comes
in with `biosignal-data import <id> <folder> --version <label>`.

## Check and regenerate

```bash
uv run biosignal-data dev validate
uv run biosignal-data dev docs      # regenerates docs/datasets/
uv run pytest
```

## OpenNeuro and NEMAR entries

These are generated from the [EEGDash](https://github.com/eegdash/EEGDash) catalog:

```bash
git clone --depth 1 https://github.com/eegdash/EEGDash /tmp/eegdash
uv run python scripts/import_eegdash.py /tmp/eegdash/eegdash/dataset/dataset_summary.csv \
    --commit "$(git -C /tmp/eegdash rev-parse HEAD)"
```

The script only overwrites files it generated (they carry its marker note). To curate a
generated entry by hand, edit it and delete the marker note.

## Adding a new kind of source

1. Add a source model to `catalog/models.py` and to the `Source` union.
2. Write a `Fetcher` in `fetchers/`: `resolve_version()` and `list_files()` yielding
   `RemoteFile`s that open as byte streams. Use the repository's official client where one
   exists. If its tool can only write to disk, subclass `StagedFetcher`.
3. Register it in `fetchers/__init__.py` and add tests with recorded responses.

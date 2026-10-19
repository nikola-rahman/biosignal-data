# Reading the raw layer

The raw layer holds each dataset in its original formats. `biosignal_data.read` helps find and
open those files; it does not convert them.

## Find and fetch files

```python
from pathlib import Path
from biosignal_data.read import RawDataset
from biosignal_data.storage.stores import S3Settings

cache = Path("~/.cache/biosignal-data")  # read.cache_dir in biosignal-data.yaml
s3 = S3Settings.from_env("https://s3.hippius.com", "decentralized", "path")  # the config's s3

ds = RawDataset.open("s3://my-bucket", "openneuro/ds005505", "latest", cache_dir=cache, s3=s3)
ds = RawDataset.open("s3://my-bucket", "openneuro/ds005505", "1.0.1", cache_dir=cache, s3=s3)

ds.manifest.license  # provenance travels with the data
eeg = ds.glob("sub-*/eeg/*_eeg.set")  # manifest entries: path, size, sha256, ...
path = ds.local_path(eeg[0].path)  # downloaded once to the cache, SHA-256 checked
uri = ds.uri(eeg[0].path)  # s3://... for tools that read S3 directly
```

`version` is a label or `"latest"` (the newest complete version); `cache_dir` is where
`local_path` keeps fetched files. Both are required.

Patterns are standard globs: `*` stays inside one folder, `**` crosses folders. Formats split
across files need all parts cached: `100.hea` and `100.dat` for a WFDB record, the `.vhdr`,
`.vmrk` and `.eeg` for BrainVision, or `ds.local_paths("sub-01/**")` for a whole BIDS
subject.

## Open files

| Format | Function | Returns |
| --- | --- | --- |
| EDF/EDF+, BDF, GDF, BrainVision, EEGLAB, FIF, CNT, SNIRF, ... | `read_mne(path)` | `mne.io.Raw` |
| BIDS dataset (OpenNeuro, NEMAR) | `read_bids(root, subject=..., task=..., datatype=...)` | `mne.io.Raw` with sidecars applied |
| PhysioNet WFDB | `read_wfdb(record)`, `read_wfdb_annotation(record, "atr")` | `wfdb.Record`, `wfdb.Annotation` |
| NWB (DANDI) | `read_nwb(path)` | `pynwb.NWBHDF5IO` |
| HDF5 | `read_hdf5(path)` | `h5py.File` |
| MATLAB | `read_mat(path)` | `dict` (v4-v7) or `h5py.File` (v7.3) |
| CSV, TSV, Parquet, Excel | `read_table(path)` | `pandas.DataFrame` |
| XML annotations (NSRR, Compumedics) | `read_xml(path)` | `xml.etree.ElementTree.Element` |
| any of the above | `read_any(path)` | picks by suffix |

Every function passes extra keyword arguments to the underlying library.

```python
from biosignal_data.read import read_bids, read_mne

raw = read_mne(path)  # header only until you ask for data
data, times = raw[:, :1000]

ds.local_paths("**/*.json")
ds.local_paths("**/*.tsv")  # BIDS sidecars at every level
ds.local_paths("sub-01/**")  # one subject's recordings
raw = read_bids(ds.local_root, subject="01", task="rest", datatype="eeg")
```

## Read the standard layer

```python
from biosignal_data.read import StandardDataset

ds = StandardDataset.open("s3://my-bucket", "physionet/mitdb", "latest", s3=s3)
ds.recordings.to_pandas()  # one row per recording: start, duration, arrays, raw files
ds.channels.to_pandas()  # name, unit, sampling rate, scale, offset per channel
ds.events.to_pandas()  # annotations: onset, duration, label, text

group = ds.signals("100")  # zarr.Group, read from the store as needed
group["360Hz"].shape  # (channels, samples), the stored integers
window = ds.window("100", "360Hz", start_s=600.0, duration_s=30.0)  # physical units
```

Only the chunks a window needs are read. `physical=False` gives the stored integers. In
physical units, samples stored as the channel's invalid value and channels without a scale
are NaN. The format is described in [standard-layer.md](standard-layer.md).

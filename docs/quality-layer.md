# The quality layer

What each channel of the standard layer holds, measured once per standard version. It
measures and never deletes or changes anything: which channels go into training is a
setting of each run, visible in its configuration.

    biosignal-data quality --repository physionet --dest s3://biosignal-data

    quality/<repository>/<dataset>/<standard version>/
        channels.parquet   one row per channel
        _manifest.json     the standard version and its manifest's SHA-256, the settings
        _SUCCESS

A run that stops continues where it stopped (one part per recording under `_parts/`).
The settings are in the `quality` section of `biosignal-data.yaml`.

## Columns

| Column | What it counts |
|---|---|
| `samples`, `invalid` | samples; those stored as the channel's invalid value |
| `nonfinite` | samples whose physical value is not a finite float32: a broken calibration in the source (autonomic-aging-cardiovascular 0065 NIBP has a gain of 2.78e-152) |
| `constant` | every valid stored value is the same (a channel without a signal) |
| `longest_flat_s`, `flat_fraction` | runs of one stored value; a run counts as flat from `flat_s` seconds on (at least 2 samples) |
| `at_low`, `at_high`, `limits_from` | the share of samples at the recorder's limits: the EDF header's digital minimum and maximum (`header`), else the limits of the stored integer type (`dtype`) |
| `minimum`, `maximum`, `mean`, `std` | of the valid physical values, exact |
| `p01`, `p50`, `p99` | percentiles from a seeded uniform random sample of at most `percentile_sample` valid values per channel; exact percentiles would need a whole channel in memory |

## Flat channels and runs in training

The columns above are per channel and need a run over the whole standard version (on
2026-10-29 only 8 small datasets had one). Training does not wait for them:

- A run of one stored value much longer than the usual runs of its channel is marked as
  not observed by the data loader, on the span it reads (`flat_runs`, see
  [training-layer.md](training-layer.md)).
- A channel that holds a placeholder in every recording (i-care and siena SpO2: the stored
  value 0) gets the type `other` by a rule in [`index.yaml`](../index.yaml), with the
  evidence, so a run that selects types leaves it out.

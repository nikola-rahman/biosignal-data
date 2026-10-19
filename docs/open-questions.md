# Open questions

Things that are not settled yet, in rough order of how much they matter.

1. **No dataset has been downloaded completely yet.** The fetchers are tested against recorded and
   mocked responses only. The first real runs will find mismatches with live services; fix them
   and set `download_verified: true` on each entry that succeeds.
2. **Several hour figures are estimates** (`hours_estimated: true`), notably the Harvard EEG
   Database (~2M h, from one preprint), the Human Sleep Project (recordings x 8 h) and
   Icentia11k (segment count). They are informational and do not affect ingestion.
3. **OpenNeuro older snapshots.** OpenNeuro's S3 bucket only serves the latest snapshot. Older
   ones need DataLad; download them that way, then run `biosignal-data import <id> <folder> --version <tag>`.
4. **BDSP key prefixes.** The credentialed access point holds several datasets; the prefix of
   each is shared by BDSP after approval and written into the dataset's entry in catalog.yaml (`source.prefix`).
5. **Overlap between datasets.** Some catalog entries contain others (TUH subcorpora inside
   TUEG, PhysioNet Challenge 2021 inside several 12-lead sets, NEMAR copies of Meta's EMG
   data). The raw layer keeps them as separate datasets; deduplication belongs to a later
   layer. `related` records the known overlaps.
6. **Licenses.** Many sources are non-commercial or share-alike. The license is recorded per
   version in the manifest; deciding what may be used for what is outside this repository.
7. **Entries with uncertain details** say so in their `notes`, e.g. unconfirmed project ids.
8. **Very large file counts.** MIMIC-III Waveform has millions of files; the partial manifest
   is rewritten every 30 s, which gets slow at that scale.

9. **The standard layer converts WFDB and EDF/BDF only.** Multi-segment WFDB records and
   discontinuous EDF+ are listed as not converted; adding them must keep their gaps explicit.
   BIDS (OpenNeuro, NEMAR), NWB, MATLAB and the other formats come next. Subjects, and the
   mapping of channel and event names to a common vocabulary, are not done yet.

10. **BDF files are read whole.** edfio 0.4 has no lazy loading for BDF, so a BDF file is
    read into memory at once (about 4/3 of its size). Large BDF recordings (e.g. 128
    channels at 2 kHz for hours) need more memory than a worker has; read them in slices
    when BIDS datasets with BDF are converted.

## Research questions for the model

The data layers keep everything native (rates, values, units, channel metadata, times) so
that these stay open:

- **Standardize or not.** The usual practice (one sampling rate, a fixed montage, z-scored
  amplitudes, fixed windows) loses information. The alternative: positions in seconds,
  tokens at the native rate with dynamic patch sizes, one token stream per channel with a
  metadata embedding (type, position, reference, unit, filters), physical units with the
  scale as an input, and packed sequences of different lengths. Compare both on tasks that
  need high frequencies (e.g. high-frequency oscillations in EEG, EMG).
- **Time scales.** The finest detail (patch size and rate), the base context, and whether a
  long-context level over window embeddings (hours, days) is part of the foundation model.
- **Annotations.** Self-supervised pretraining on signals only; annotations for downstream
  tasks. Datasets used for both are split by subject.
- **The target.** At least 50 % less error than the best published results, on a fixed
  benchmark suite with the same labels and splits. Some tasks have a ceiling set by how
  well human experts agree (e.g. sleep staging).

## Backlog

- **Request access to the gated datasets** (95 need a person): NSRR data requests per
  dataset, the TUH form and SSH key, PhysioNet credentialing (CITI course and DUA), BDSP
  (CITI certificate and DUA), a Kaggle token, and the downloads by hand. Each dataset's page
  in `docs/datasets/` lists its steps; `docs/access.md` lists the variables to set.

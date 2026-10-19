# Open questions

Things that are not settled yet, in rough order of how much they matter.

1. **No dataset has been downloaded completely yet.** The fetchers are tested against recorded and
   mocked responses only. The first real runs will find mismatches with live services; fix them
   and set `download_verified: true` on each entry that succeeds.
2. **Several hour figures are estimates** (`hours_estimated: true`), notably the Harvard EEG
   Database (~2M h, from one preprint), the Human Sleep Project (recordings x 8 h) and
   Icentia11k (segment count). They are informational and do not affect ingestion.
3. **OpenNeuro older snapshots.** OpenNeuro's S3 bucket only serves the latest snapshot. Older
   ones need DataLad; download them that way, then run `download --from-dir <folder> --version <tag>`.
4. **BDSP key prefixes.** The credentialed access point holds several datasets; the prefix of
   each is shared by BDSP after approval and passed with `--source-prefix`.
5. **Overlap between datasets.** Some catalog entries contain others (TUH subcorpora inside
   TUEG, PhysioNet Challenge 2021 inside several 12-lead sets, NEMAR copies of Meta's EMG
   data). The raw layer keeps them as separate datasets; deduplication belongs to a later
   layer. `related` records the known overlaps.
6. **Licenses.** Many sources are non-commercial or share-alike. The license is recorded per
   version in the manifest; deciding what may be used for what is outside this repository.
7. **Entries with uncertain details** say so in their `notes`, e.g. unconfirmed project ids.
8. **Very large file counts.** MIMIC-III Waveform has millions of files; the partial manifest
   is rewritten every 30 s, which gets slow at that scale. PhysioNet also mirrors many
   datasets on AWS (`s3://physionet-open`), which would be faster for the biggest ones.

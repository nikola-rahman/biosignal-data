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
  tokens at the native rate with dynamic patch sizes, physical values with the scale as an
  input, and packed sequences of different lengths. Compare both on tasks that need high
  frequencies (e.g. high-frequency oscillations in EEG, EMG).
- **Samples and times only (decided 2026-10-20).** The model gets no channel names, types,
  units or datasets, at training or inference; channels of one recording come together, in
  random order, with random slots that only say "same channel". Metadata stays in the
  index, for selecting data and analysing results.
- **Spectral or raw input.** Audio models often use log-mel spectrograms: shorter
  sequences, no phase, log compression of loudness, a hearing-based frequency scale. The
  strongest self-supervised speech models (wav2vec 2.0, HuBERT) use the raw waveform with a
  learned front end. Spectral content is strong for EEG, EMG and heart sounds; ECG, blood
  pressure and PPG depend on morphology, timing and the phase between channels, which a
  magnitude spectrogram loses, and a fixed STFT is a human prior. To test, keeping the raw
  input: (1) a spectral term in the loss of hidden tokens (plain MSE in time punishes
  small shifts and noise), (2) STFT tokens with magnitude and phase (lossless), (3) a
  learned convolutional front end in place of the linear patch layer.
- **Time scales.** The finest detail (patch size and rate), the base context, and whether a
  long-context level over window embeddings (hours, days) is part of the foundation model.
- **Annotations.** Self-supervised pretraining on signals only; annotations for downstream
  tasks. Datasets used for both are split by subject.
- **The target.** At least 50 % less error than the best published results, on a fixed
  benchmark suite with the same labels and splits. Some tasks have a ceiling set by how
  well human experts agree (e.g. sleep staging).

## Backlog

- **Many tiny recordings in raw (needed before MIMIC-IV-ECG, 800,000 ECGs).** One object
  per published file costs a request per file to copy and to read. An explicit catalog
  option could store the source's official archive (e.g. PhysioNet's ZIP) as one object,
  every member checked against SHA256SUMS.txt and listed in the manifest, single members
  read by range requests. (The standard layer was changed on 2026-10-20: blocks in 64 MiB
  objects with one index, see standard-layer.md.)
- **Fine-tuning recipe (before any conclusion about pretraining).** PTB-XL `all`
  (2026-10-20, checkpoint v0-rtx6000-c2 step 30,000): finetune 0.861, supervised from
  scratch 0.867, the same architecture and recipe (AdamW lr 3e-4 on every weight from the
  first step, a random linear head, no warmup, no layer-wise decay, mean pooling). Finetune
  learns faster, then overfits from epoch 8. That the recipe destroys the pretrained
  features (catastrophic forgetting) is not ruled out; a better recipe comes first.
- **Mixture weights for pretraining.** The loader now reads every recording once per
  epoch, so each dataset counts by its size: in the test index (9 datasets) capslpdb is 77 %
  of the training tokens, adfecgdb 0.02 % (3 recordings of 5 min). The weight of each
  dataset or signal type should be a setting of the run (natural proportions, a sampling
  temperature as in LLM data mixtures, or explicit weights), because the mix is the
  research question itself: does mixing help scarce signals and hurt abundant ones.
- **A quality table over the standard layer** (agreed 2026-10-20): one row per channel,
  computed once and stored with the index: the share of invalid samples, of flat segments
  and of samples at the recorder's minimum or maximum; minimum, maximum and percentiles of
  the physical values; non-finite counts. It measures and never deletes: which limits put a
  channel into training is a setting of the run, as `signal_types` is.
- **The arousal labels of challenge-2018** (`tr*-arousal.mat`, one label per sample, 994
  files): not converted, because no WFDB header names them; needed for its benchmark.
- **Request access to the gated datasets** (95 need a person): NSRR data requests per
  dataset, the TUH form and SSH key, PhysioNet credentialing (CITI course and DUA), BDSP
  (CITI certificate and DUA), a Kaggle token, and the downloads by hand. Each dataset's page
  in `docs/datasets/` lists its steps; `docs/access.md` lists the variables to set.

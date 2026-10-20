# The raw layer

The raw layer (the "bronze" layer, in data-lake terms) is a faithful copy of each source. It is
the foundation every later, standardized layer is built from, so it optimizes for one thing:
being able to prove where every byte came from.

## Rules

1. **Bytes are never changed.** Files are stored exactly as the source serves them. Archives
   stay archives, formats stay formats, folder structure and file names are kept.
2. **Versions are immutable.** A version folder is written once. When a source publishes a
   new release, it goes into a new version folder next to the old one.
3. **Every version carries its own provenance.** `_manifest.json` records what was copied,
   from where, under which license, when, and with which checksums.
4. **Completeness is explicit.** `_SUCCESS` is written only after every file is copied and
   verified, and after the manifest. Readers ignore versions without it.
5. **A file the source lists but does not have does not block a version.** When the source
   answers 404 for a file in its own file list, the file is recorded under `missing` in the
   manifest, with the answer, and the run warns. Every other error fails the file, and the
   version stays incomplete until a run copies it.
6. **A file served other than its source lists it is downloaded twice.** When the size or
   checksum differs from the source's own list, the file is downloaded again. The same bytes
   twice mean the source serves them so: the file is kept as served, with the difference in
   its manifest entry (`differs_from_source`), and the run warns. Different bytes mean a
   transfer error, and the file fails.
7. **A complete version can get the files it lacks, and only those.** When the source
   holds files that a complete version lacks (PhysioNet added apnea-ecg's test-set answers
   after its `SHA256SUMS.txt`; until 2026-10-20 the PhysioNet fetcher read only that list),
   `download --add-missing` lists the source again and copies only those files. Stored
   files are not touched and none is removed; the manifest is written again at the end.
   This repairs a copy of the same version; it is not a new version (rule 2).

## Keys

```
<store root>/raw/<repository>/<dataset>/<version>/data/<path as published>
<store root>/raw/<repository>/<dataset>/<version>/_manifest.json
<store root>/raw/<repository>/<dataset>/<version>/_manifest.partial.json   (only while running)
<store root>/raw/<repository>/<dataset>/<version>/_SUCCESS
```

- **`<repository>/<dataset>`** is the catalog id: the repository where the data lives and that
  repository's own identifier, e.g. `physionet/ptb-xl`, `openneuro/ds004789`, `dandi/000055`.
  Repositories with owner namespaces add the owner: `huggingface/NeuroTec/SWEC_iEEG_Dataset`.
  Where a dataset has no repository identifier, the id is `<host organisation>/<dataset name
  as published>`, e.g. `kuleuven/seizeit1`.
- **`<version>`** is the source's own release label where one exists (PhysioNet `1.0.3`,
  OpenNeuro snapshot `1.0.1`, DANDI `0.220126.1903`, TUH `v2.0.2`, a Hugging Face commit
  hash, a Zenodo `record-<id>`). Where the source has no releases, it is the UTC download
  date, `YYYY-MM-DD`, and the manifest says so (`version_origin: retrieval-date`).
- **`data/`** keeps source files apart from our own metadata files, so a source file can never
  collide with `_manifest.json` or `_SUCCESS`.

The store root is the prefix in `--dest`: `--dest s3://lake` gives `s3://lake/raw/...`,
`--dest s3://lake/team-a` gives `s3://lake/team-a/raw/...`.

## Manifest

```json
{
  "schema_version": 1,
  "dataset_id": "physionet/mitdb",
  "version": "1.0.0",
  "version_origin": "source",
  "source": {"type": "physionet", "slug": "mitdb", "version": "1.0.0", "credentialed": false},
  "license": null,
  "retrieved_at": "2026-10-18T12:00:00Z",
  "completed_at": "2026-10-18T12:03:10Z",
  "tool": "biosignal-data 0.1.0",
  "files": [
    {
      "path": "100.dat",
      "size": 1950000,
      "sha256": "b2ea3c25...",
      "md5": "...",
      "source_checksum": "sha256:b2ea3c25...",
      "source_url": "https://physionet.org/files/mitdb/1.0.0/100.dat"
    }
  ],
  "file_count": 1,
  "total_bytes": 1950000
}
```

`sha256` and `md5` are computed by us while the bytes stream through. `source_checksum` is what
the source published, if it published one; the copy fails when the two disagree.

## Integrity

- Each file is hashed in flight and compared with the source's published size and checksum
  (PhysioNet SHA-256, NEMAR SHA-256/MD5, DANDI SHA-256, Zenodo/figshare MD5, Hugging Face
  LFS SHA-256). On a mismatch the object is deleted and the file is reported as failed.
- Sources without checksums (OpenNeuro's S3 bucket, plain HTTPS, manual downloads) are
  recorded with our own SHA-256 only.
- `biosignal-data check <id> --dest ... [--version V] [--checksums]` re-checks a stored
  version (default: the latest): sizes against the manifest, and with `--checksums` every
  SHA-256.

## Resuming

Progress is saved to `_manifest.partial.json` every 30 seconds. Re-running the same download
skips files already recorded there whose stored size matches, and copies the rest. The
partial manifest is deleted once `_SUCCESS` is written.

## Recommended bucket settings

These are outside the code, set once per bucket:

- Versioning on, plus Object Lock or a deny-delete policy on `raw/`, so the immutability rule
  is enforced by S3 and not only by convention.
- Default encryption (SSE-S3 or SSE-KMS). Several datasets require it under their data use
  agreements.
- Block all public access. Credentialed datasets (BDSP, PhysioNet credentialed, NSRR) must not
  be readable by anyone outside the approved users; consider a separate prefix or bucket for
  them with its own access policy.
- An S3 Intelligent-Tiering or lifecycle rule: raw data is written once and read rarely.

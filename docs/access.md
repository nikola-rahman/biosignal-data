# Access and credentials per repository

Every secret comes from a `BIOSIGNAL_*` environment variable, and from nothing else: the
code does not read `~/.aws`, `~/.kaggle`, `~/.ssh/config`, saved Hugging Face tokens or other
tools' variables. Never put secrets in the catalog, the config file, or a commit.

| Repository | What you need | Variables |
| --- | --- | --- |
| PhysioNet (open) | nothing; read only from PhysioNet's copy on AWS (see below) | |
| PhysioNet (credentialed) | a credentialed PhysioNet account (CITI training) and the dataset's DUA signed on its page; read only from physionet.org | `BIOSIGNAL_PHYSIONET_USERNAME`, `BIOSIGNAL_PHYSIONET_PASSWORD` |
| OpenNeuro | nothing | |
| NEMAR | nothing | |
| DANDI | nothing for public dandisets; an API key for embargoed ones | `BIOSIGNAL_DANDI_API_KEY` |
| Zenodo | nothing for public records; a token for restricted ones | `BIOSIGNAL_ZENODO_TOKEN` |
| figshare | nothing | |
| Hugging Face | nothing for public repos; a token for gated ones | `BIOSIGNAL_HF_TOKEN` |
| Kaggle | a Kaggle account, and the competition's rules accepted on its page | `BIOSIGNAL_KAGGLE_API_TOKEN` (Settings → API on kaggle.com) |
| NSRR (sleepdata.org) | an approved data request for each dataset | `BIOSIGNAL_NSRR_TOKEN` (from https://sleepdata.org/token) |
| BDSP (bdsp.io) | a credentialed BDSP account (CITI certificate) and the dataset's DUA; BDSP then issues S3 keys and the dataset's key prefix | `BIOSIGNAL_BDSP_ACCESS_KEY_ID`, `BIOSIGNAL_BDSP_SECRET_ACCESS_KEY`; the prefix goes into the catalog entry as `source.prefix` |
| TUH EEG (isip) | the signed TUH form emailed to help@nedcdata.org, then your SSH public key registered | `BIOSIGNAL_TUH_SSH_KEY` (path to the private key) |
| Manual sources | whatever the provider's page says | none: download into a folder, then `biosignal-data import` |

Your own S3 storage (the destination): `BIOSIGNAL_ACCESS_KEY_ID` and
`BIOSIGNAL_SECRET_ACCESS_KEY`; its endpoint, region and addressing are in
`biosignal-data.yaml`.

## PhysioNet: one source per dataset, no switching

- **Open datasets** come only from PhysioNet's own copy on AWS, `s3://physionet-open`
  (AWS Open Data program, no account). It is faster than physionet.org and gives file
  sizes. If the version, or a file of it, is not on the copy, the download fails with an
  error; it does not fall back to physionet.org.
- **Credentialed datasets** are not on that copy. They come only from physionet.org, with
  your PhysioNet account.
- The file list and SHA-256 checksums come from the version's `SHA256SUMS.txt`. Old versions
  publish none: their files are then listed from the AWS listing (open) or the physionet.org
  folder index (credentialed), and cannot be checked against published checksums (warning).
- For an entry without a pinned version, the latest version is read from physionet.org.

## Versions

- A version in the catalog entry is downloaded exactly; if the source does not have it, the
  download fails.
- An entry without a version gets the source's latest, with a warning naming it.
- Plain HTTP, rsync (TUH) and S3 prefixes (BDSP) cannot be asked for a version: there the
  catalog's version is only a label for what the source serves now (warning on every run).

## How each source is read

Every source streams from the source to the destination, without local disk, except TUH:

- PhysioNet, OpenNeuro, NEMAR, DANDI, Zenodo, figshare, Hugging Face and BDSP through their
  official clients or APIs.
- NSRR through the web API that the official `nsrr` gem uses (same requests, no gem).
- Kaggle through Kaggle's official Python SDK (`kagglesdk`, installed with the package).
- TUH EEG through `rsync` over SSH, which can only write into a folder: the files land in
  `staging.dir` from `biosignal-data.yaml` and are then copied. The folder is reused, so an
  interrupted rsync resumes; after a complete copy it is deleted unless `staging.keep` is
  true. The disk must hold the dataset (TUEG is several TB). `rsync` and `ssh` must be
  installed; ssh uses only the key in `BIOSIGNAL_TUH_SSH_KEY`, no `~/.ssh/config`.

## Not yet confirmed

These details come from documentation or the official tools' source code and have not been
exercised against the live service:

- NSRR: the listing and download requests (taken from the `nsrr` gem's source code).
- Kaggle: downloads through `kagglesdk` with a token.
- BDSP: the key prefix of each dataset inside the credentialed access point.
- PhysioNet's redirect from `/content/<slug>/` to the latest version (used only when an entry
  pins no version).

# Access and credentials per repository

Credentials are read from environment variables or the tool's own config files. Never put them
in the catalog or commit them.

| Repository | What you need | Where the code reads it |
| --- | --- | --- |
| PhysioNet (open) | nothing | |
| PhysioNet (credentialed) | a credentialed PhysioNet account (CITI training) and the dataset's DUA signed on its page | `PHYSIONET_USERNAME`, `PHYSIONET_PASSWORD` |
| OpenNeuro | nothing | |
| NEMAR | nothing | |
| DANDI | nothing for public dandisets; an API key for embargoed ones | `DANDI_API_KEY` |
| Zenodo | nothing for public records; a token for restricted ones | `ZENODO_TOKEN` |
| figshare | nothing | |
| Hugging Face | nothing for public repos; a token for gated ones | `huggingface-cli login` or `HF_TOKEN` |
| Kaggle | a Kaggle account, and the competition's rules accepted on its page | `kaggle auth login`, `KAGGLE_API_TOKEN` or `~/.kaggle/access_token` |
| NSRR (sleepdata.org) | an approved data request for each dataset; the Ruby gem `gem install nsrr` | `NSRR_TOKEN` (from https://sleepdata.org/token) |
| BDSP (bdsp.io) | a credentialed BDSP account (CITI certificate) and the dataset's DUA; BDSP then provides AWS credentials for its access points | an AWS profile, passed with `--aws-profile` |
| TUH EEG (isip) | the signed TUH form emailed to help@nedcdata.org, then your SSH public key registered | `TUH_SSH_KEY` (path to the private key); `rsync` and `ssh` installed |
| Manual sources | whatever the provider's page says | none: download into a folder, then `download --from-dir` |

## External tools

Python clients are installed with the package. Three sources use tools outside Python:

- `rsync` and `ssh` for TUH EEG (preinstalled on Linux and macOS).
- `nsrr` Ruby gem for NSRR: `gem install nsrr --no-document` (needs Ruby 2.7+).
- `kaggle` CLI is installed with the package.

## Staging

rsync, the NSRR gem and the Kaggle CLI can only write to local disk. Their downloads land in
`--staging-dir` (default: a temporary folder, deleted afterwards) and are then copied to the
store. Use a persistent `--staging-dir` for large datasets: the tools resume from what is
already there, and `--keep-staging` keeps the files after a successful copy. Make sure the
disk can hold the whole dataset (TUEG is several TB).

## Not yet confirmed

These details come from documentation and have not been exercised against the live service:

- The NSRR gem's `--token=` flag for non-interactive downloads.
- The key prefixes inside BDSP's credentialed access point for each dataset; BDSP shares them
  after approval (pass `--source-prefix`).
- PhysioNet's redirect from `/content/<slug>/` to the latest version (used only when an entry
  pins no version).

"""Schemas of the two dataset files.

- `catalog.yaml` (`Dataset`): what a download needs, per dataset: its id, its source (type,
  identifier at the source, version, the source's own settings) and its license, which every
  copied version records in its manifest.
- `docs/datasets.yaml` (`DatasetInfo`): descriptions for people (name, signals, access,
  links, sizes, notes). The docs and the dataset explorer are built from it; downloads do
  not read it.

Neither says how the data should look after copying: the raw layer keeps every file exactly
as the source publishes it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

# <repository>/<id> or <repository>/<owner>/<id> (for repositories with owner namespaces,
# e.g. huggingface/NeuroTec/SWEC_iEEG_Dataset).
DATASET_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*(/[A-Za-z0-9][A-Za-z0-9._-]*){1,2}$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AccessTier(StrEnum):
    """How hard it is to obtain the data. Ordered from easiest to hardest."""

    OPEN = "open"  # anonymous download
    REGISTRATION = "registration"  # free account, token or click-through license
    APPLICATION = "application"  # a request someone reviews (form, DUA, fee)
    CREDENTIALED = "credentialed"  # approval plus training or identity checks (CITI, DUA)
    UNAVAILABLE = "unavailable"  # known to exist but not obtainable (private, discontinued)


class Access(_Model):
    tier: AccessTier
    steps: list[str] = Field(
        default_factory=list,
        description="Manual steps a person must do before `ingest` can run, in order.",
    )
    credentials: list[str] = Field(
        default_factory=list,
        description="Environment variables (or files) the fetcher reads credentials from.",
    )


class Stats(_Model):
    """Informational size figures, as reported by the source or estimated."""

    subjects: int | None = None
    recordings: int | None = None
    hours: float | None = None
    hours_estimated: bool = False
    size_bytes: int | None = None
    note: str | None = None


# --- Sources -------------------------------------------------------------------------------
# One model per way of getting bytes. `type` selects the fetcher.


class PhysioNetSource(_Model):
    type: Literal["physionet"]
    slug: str
    version: str | None = Field(None, description="Pinned version; None = latest at ingest time.")
    credentialed: bool = False


class OpenNeuroSource(_Model):
    type: Literal["openneuro"]
    accession: Annotated[str, Field(pattern=r"^ds\d{6}$")]
    snapshot: str | None = None


class NemarSource(_Model):
    type: Literal["nemar"]
    accession: Annotated[str, Field(pattern=r"^nm\d{6}$")]
    version: str | None = None


class DandiSource(_Model):
    type: Literal["dandi"]
    dandiset: Annotated[str, Field(pattern=r"^\d{6}$")]
    version: str | None = Field(None, description="Published version; None = most recent.")


class ZenodoSource(_Model):
    type: Literal["zenodo"]
    record: int


class FigshareSource(_Model):
    type: Literal["figshare"]
    article: int | None = None
    collection: int | None = None
    api: HttpUrl = HttpUrl("https://api.figshare.com/v2")

    @model_validator(mode="after")
    def _one_of(self) -> FigshareSource:
        if (self.article is None) == (self.collection is None):
            raise ValueError("set exactly one of `article` or `collection`")
        return self


class HuggingFaceSource(_Model):
    type: Literal["huggingface"]
    repo_id: str
    repo_type: Literal["dataset", "model"] = "dataset"
    revision: str | None = None


class KaggleSource(_Model):
    type: Literal["kaggle"]
    kind: Literal["competition", "dataset"]
    ref: str


class S3Source(_Model):
    type: Literal["s3"]
    bucket: str = Field(description="Bucket name or S3 access point ARN.")
    prefix: str | None = Field(
        None,
        description="Key prefix. None = not known yet (e.g. given by BDSP after approval); "
        "set it here before downloading.",
    )
    region: str
    anonymous: bool
    key_id_env: str | None = Field(
        None, description="Not anonymous: variable holding the access key id."
    )
    secret_env: str | None = Field(
        None, description="Not anonymous: variable holding the secret access key."
    )
    version: str | None = None

    @model_validator(mode="after")
    def _keys_unless_anonymous(self) -> S3Source:
        if not self.anonymous and not (self.key_id_env and self.secret_env):
            raise ValueError("a source that is not anonymous needs key_id_env and secret_env")
        return self


class HttpFile(_Model):
    url: HttpUrl
    path: str | None = Field(None, description="Path under data/; defaults to the URL file name.")


class HttpSource(_Model):
    type: Literal["http"]
    files: list[HttpFile] = Field(min_length=1)
    version: str | None = None


class NsrrSource(_Model):
    type: Literal["nsrr"]
    slug: str
    subpath: str | None = None


class RsyncSource(_Model):
    type: Literal["rsync"]
    remote: str = Field(description="rsync remote, e.g. user@host:path")
    version: str | None = None
    ssh_identity_env: str = Field(
        description="Environment variable holding the path to the SSH private key."
    )


class UnavailableSource(_Model):
    """Known to exist, but there is no way to get it (private, withdrawn, discontinued)."""

    type: Literal["unavailable"]
    url: HttpUrl | None = None


class ManualSource(_Model):
    """No programmatic download. A person follows `access.steps` into a local folder,
    then `biosignal-data import <id> <folder>` publishes it to the raw layer."""

    type: Literal["manual"]
    url: HttpUrl | None = None
    version: str | None = None


Source = Annotated[
    PhysioNetSource
    | OpenNeuroSource
    | NemarSource
    | DandiSource
    | ZenodoSource
    | FigshareSource
    | HuggingFaceSource
    | KaggleSource
    | S3Source
    | HttpSource
    | NsrrSource
    | RsyncSource
    | ManualSource
    | UnavailableSource,
    Field(discriminator="type"),
]


def _check_dataset_id(value: str) -> str:
    if not DATASET_ID_RE.fullmatch(value):
        raise ValueError(
            f"dataset id {value!r} must look like 'repository/dataset' or "
            "'repository/owner/dataset'"
        )
    return value


class Dataset(_Model):
    """One entry of catalog.yaml: what a download needs."""

    id: str = Field(description="<repository>/<the repository's own identifier>")
    source: Source
    license: str | None = Field(None, description="Recorded in every copied version's manifest.")

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _check_dataset_id(value)

    @property
    def repository(self) -> str:
        return self.id.split("/", 1)[0]

    @property
    def pinned_version(self) -> str | None:
        """The version fixed in the catalog, if the source type carries one."""
        for attr in ("version", "snapshot", "revision"):
            value = getattr(self.source, attr, None)
            if value:
                return str(value)
        return None


class DatasetInfo(_Model):
    """One entry of docs/datasets.yaml: the description of a dataset, for people."""

    id: str
    name: str
    signals: list[str] = Field(min_length=1)
    access: Access
    homepage: HttpUrl | None = None
    references: list[HttpUrl] = Field(default_factory=list)
    stats: Stats = Field(default_factory=Stats)
    related: list[str] = Field(
        default_factory=list, description="Other dataset ids this overlaps with or contains."
    )
    notes: list[str] = Field(default_factory=list)
    download_verified: bool = Field(
        False, description="True once a complete copy of this dataset has been checked."
    )

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        return _check_dataset_id(value)

    @field_validator("signals")
    @classmethod
    def _lower_signals(cls, value: list[str]) -> list[str]:
        return [s.strip().lower() for s in value]

    @property
    def repository(self) -> str:
        return self.id.split("/", 1)[0]

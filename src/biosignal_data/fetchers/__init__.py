"""Map a catalog source type to the fetcher that reads it."""

from __future__ import annotations

from biosignal_data.catalog.models import Dataset
from biosignal_data.fetchers.archives import (
    DandiFetcher,
    FigshareFetcher,
    HttpFetcher,
    HuggingFaceFetcher,
    ZenodoFetcher,
)
from biosignal_data.fetchers.base import (
    AccessRestrictedError,
    Fetcher,
    FetchError,
    FetchOptions,
    LocalDirFetcher,
    ManualDownloadError,
    MissingCredentialsError,
    RemoteFile,
    ResolvedVersion,
)
from biosignal_data.fetchers.openneuro_nemar import NemarFetcher, OpenNeuroFetcher
from biosignal_data.fetchers.physionet import PhysioNetFetcher
from biosignal_data.fetchers.s3 import S3Fetcher
from biosignal_data.fetchers.staged import ManualFetcher, RsyncFetcher
from biosignal_data.fetchers.web_apis import KaggleFetcher, NsrrFetcher

FETCHERS: dict[str, type[Fetcher]] = {
    "physionet": PhysioNetFetcher,
    "openneuro": OpenNeuroFetcher,
    "nemar": NemarFetcher,
    "dandi": DandiFetcher,
    "zenodo": ZenodoFetcher,
    "figshare": FigshareFetcher,
    "huggingface": HuggingFaceFetcher,
    "kaggle": KaggleFetcher,
    "s3": S3Fetcher,
    "http": HttpFetcher,
    "nsrr": NsrrFetcher,
    "rsync": RsyncFetcher,
    "manual": ManualFetcher,
}


def fetcher_for(dataset: Dataset, options: FetchOptions | None = None) -> Fetcher:
    """The fetcher for `dataset`. With `options.from_dir` (`import`), files come from that
    folder instead, whatever the source type."""
    options = options or FetchOptions()
    if options.from_dir is not None:
        return LocalDirFetcher(dataset, options)
    return FETCHERS[dataset.source.type](dataset, options)


__all__ = [
    "FETCHERS",
    "AccessRestrictedError",
    "FetchError",
    "FetchOptions",
    "Fetcher",
    "ManualDownloadError",
    "MissingCredentialsError",
    "RemoteFile",
    "ResolvedVersion",
    "fetcher_for",
]

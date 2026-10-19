"""Download many datasets at once.

Several datasets are listed and copied at the same time, and all of them share one pool of
file-download threads, so the total number of parallel downloads stays at `workers`.

Every dataset ends with an `Outcome`. Datasets that need a person to act first (credentials
not set, download by hand, not available) are found before any network call and reported
apart from real failures.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal

from biosignal_data.catalog import AccessTier, Dataset
from biosignal_data.fetchers import (
    FetchError,
    FetchOptions,
    ManualDownloadError,
    MissingCredentialsError,
)
from biosignal_data.ingest import AlreadyComplete, IngestProgress, IngestReport, ingest
from biosignal_data.storage.stores import Store

Status = Literal[
    "complete",  # every file copied; the version is final
    "subset",  # the selected subset is copied (--limit / --include)
    "already",  # the version was already complete
    "incomplete",  # some files failed; running again retries them
    "credentials",  # needs credentials that are not set
    "manual",  # must be downloaded by hand, then --from-dir
    "unavailable",  # the catalog says there is no way to get it
    "failed",  # an error before copying (listing, network, ...)
]
DONE: frozenset[Status] = frozenset({"complete", "subset", "already"})
NEEDS_ACTION: frozenset[Status] = frozenset({"credentials", "manual", "unavailable"})

# Datasets listed and copied at the same time. Listing is mostly waiting on the network, so
# a few in parallel keep the shared download pool busy.
PARALLEL_DATASETS = 6


@dataclass
class Outcome:
    dataset: Dataset
    status: Status
    report: IngestReport | None = None
    message: str = ""


def download_many(
    datasets: Sequence[Dataset],
    store: Store,
    *,
    options: FetchOptions | None = None,
    root: str = "raw",
    workers: int = 16,
    include: Sequence[str] = (),
    limit: int | None = None,
    progress_for: Callable[[Dataset], IngestProgress | None] | None = None,
    on_done: Callable[[Outcome], None] | None = None,
) -> list[Outcome]:
    """Download `datasets`; returns one outcome per dataset, in the order they finished."""
    outcomes: list[Outcome] = []
    if not datasets:
        return outcomes
    files = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="file")
    runner = ThreadPoolExecutor(
        max_workers=min(PARALLEL_DATASETS, len(datasets)), thread_name_prefix="dataset"
    )

    def one(dataset: Dataset) -> Outcome:
        if dataset.access.tier is AccessTier.UNAVAILABLE:
            return Outcome(dataset, "unavailable", message="no way to get it (see notes)")
        progress = progress_for(dataset) if progress_for else None
        try:
            report = ingest(
                dataset,
                store,
                options=options,
                root=root,
                include=include,
                limit=limit,
                progress=progress,
                executor=files,
            )
        except AlreadyComplete as exc:
            return Outcome(dataset, "already", message=str(exc))
        except MissingCredentialsError as exc:
            return Outcome(dataset, "credentials", message=str(exc))
        except ManualDownloadError as exc:
            return Outcome(dataset, "manual", message=str(exc))
        except (FetchError, OSError, RuntimeError, ValueError) as exc:
            return Outcome(dataset, "failed", message=f"{type(exc).__name__}: {exc}")
        if report.complete:
            status: Status = "complete"
        elif report.failed:
            status = "incomplete"
        else:
            status = "subset"
        return Outcome(dataset, status, report=report)

    try:
        futures = [runner.submit(one, d) for d in datasets]
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            if on_done:
                on_done(outcome)
    finally:
        # On Ctrl+C: drop downloads that have not started (files first, so that the dataset
        # threads stop waiting), let files in progress finish, keep partial manifests.
        files.shutdown(wait=False, cancel_futures=True)
        runner.shutdown(cancel_futures=True)
        files.shutdown()
    return outcomes

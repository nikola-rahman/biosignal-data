"""Command-line interface: `biosignal-data --help`.

list       find datasets in the catalog
info       one dataset: what it is, how to get it, what is downloaded
download   copy a dataset (or a test subset) to a local folder or S3
check      check a download against its manifest
dev        maintainer commands (catalog validation, generated docs)
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Annotated

import typer

from biosignal_data.batch import DONE, NEEDS_ACTION, Outcome, download_many
from biosignal_data.catalog import AccessTier, Catalog, CatalogError, Dataset
from biosignal_data.console import ProgressBar, RunBar, console, download_progress, setup_logging
from biosignal_data.fetchers import FetchError, FetchOptions, RemoteFile, fetcher_for
from biosignal_data.ingest import verify
from biosignal_data.read.raw import stored_versions
from biosignal_data.selection import select
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import Manifest
from biosignal_data.storage.stores import S3Store, Store, open_store

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Find public biosignal datasets and download them unchanged, to a folder or S3.",
)
dev = typer.Typer(no_args_is_help=True, help="Maintainer commands.")
app.add_typer(dev, name="dev")

CatalogOpt = Annotated[
    Path | None,
    typer.Option(
        "--catalog",
        envvar="BIOSIGNAL_CATALOG",
        help="Catalog folder (default: the repository's catalog/).",
        hidden=True,
    ),
]
DestOpt = Annotated[
    str | None,
    typer.Option(
        "--dest",
        "-d",
        help="Where downloads go: a local folder or s3://bucket[/prefix]. "
        "Default: $BIOSIGNAL_DEST.",
    ),
]
EndpointOpt = Annotated[
    str | None,
    typer.Option(
        "--endpoint",
        help="URL of your S3-compatible storage, for s3:// destinations not on AWS S3. "
        "Default: $BIOSIGNAL_ENDPOINT.",
    ),
]
RootOpt = Annotated[
    str, typer.Option("--root", help="Layer folder inside the destination.", hidden=True)
]
IncludeOpt = Annotated[
    list[str],
    typer.Option(
        "--include",
        "-i",
        help="Only files whose path matches this glob (`*` also matches `/`). Repeatable.",
    ),
]
LimitOpt = Annotated[
    int | None,
    typer.Option(
        "--limit",
        "-n",
        min=1,
        help="Only the first N recordings, with the files and metadata each one needs.",
    ),
]


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.secho(message, fg="red", err=True)
    return typer.Exit(code)


def _catalog(path: Path | None) -> Catalog:
    try:
        return Catalog.load(path)
    except CatalogError as exc:
        raise _fail(str(exc), 2) from exc


def _dataset(catalog: Catalog, dataset_id: str) -> Dataset:
    try:
        return catalog.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc.args[0] if exc.args else exc)) from exc


def _store(dest: str | None, endpoint: str | None) -> Store:
    if not dest:
        dest = os.environ.get("BIOSIGNAL_DEST")
        if not dest:
            raise _fail("missing --dest (a local folder or s3://bucket); or set BIOSIGNAL_DEST")
        _warn(f"no --dest given; using BIOSIGNAL_DEST={dest}")
    if dest.startswith("s3://") and not endpoint:
        endpoint = os.environ.get("BIOSIGNAL_ENDPOINT")
        if endpoint:
            _warn(f"no --endpoint given; using BIOSIGNAL_ENDPOINT={endpoint}")
        else:
            _warn("no --endpoint given; using AWS S3")
    try:
        return open_store(dest, endpoint)
    except ValueError as exc:
        raise _fail(str(exc)) from exc


def _data_path(store: Store) -> str:
    """Where the bytes go, said plainly: they pass through the machine running this."""
    machine = f"this machine ({socket.gethostname()})"
    if isinstance(store, S3Store):
        target = f"{store.uri} at {store.endpoint} (credentials: {store.credentials_source})"
    else:
        target = store.uri
    return f"data flows: sources -> {machine} -> {target}"


def _warn(message: str) -> None:
    console.print(f"[yellow]warning:[/] {message}", highlight=False, soft_wrap=True)


def _size(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} TB"


# --- list ------------------------------------------------------------------------------


@app.command("list")
def list_(
    signal: Annotated[
        list[str], typer.Option("--signal", "-s", help="e.g. eeg, ecg, emg. Repeatable.")
    ] = [],  # noqa: B006
    access: Annotated[
        list[AccessTier], typer.Option("--access", "-a", help="Access tier. Repeatable.")
    ] = [],  # noqa: B006
    repository: Annotated[
        list[str], typer.Option("--repository", "-r", help="e.g. physionet. Repeatable.")
    ] = [],  # noqa: B006
    as_json: Annotated[bool, typer.Option("--json", help="Full entries as JSON.")] = False,
    catalog: CatalogOpt = None,
) -> None:
    """Find datasets in the catalog."""
    datasets = _catalog(catalog).filter(signals=signal, tiers=access, repositories=repository)
    if as_json:
        typer.echo(json.dumps([d.model_dump(mode="json") for d in datasets], indent=2))
        return
    for d in datasets:
        hours = f"{d.stats.hours:,.0f} h" if d.stats.hours is not None else "? h"
        typer.echo(f"{d.id:55} {d.access.tier.value:13} {','.join(d.signals):20} {hours:>12}")
    typer.echo(f"{len(datasets)} datasets", err=True)


# --- info ------------------------------------------------------------------------------


@app.command()
def info(
    dataset_id: Annotated[str, typer.Argument(help="Catalog id, e.g. physionet/mitdb.")],
    dest: DestOpt = None,
    endpoint: EndpointOpt = None,
    as_json: Annotated[bool, typer.Option("--json", help="The catalog entry as JSON.")] = False,
    root: RootOpt = "raw",
    catalog: CatalogOpt = None,
) -> None:
    """What a dataset is, how to get it, and (with --dest) what is downloaded."""
    d = _dataset(_catalog(catalog), dataset_id)
    if as_json:
        typer.echo(json.dumps(d.model_dump(mode="json", exclude_defaults=True), indent=2))
        return
    typer.secho(f"{d.id}  {d.name}", bold=True)
    for label, value in _info_rows(d):
        if value:
            typer.echo(f"  {label:10}{value}")
    if d.access.steps:
        typer.echo("\n  Before downloading:")
        for i, step in enumerate(d.access.steps, 1):
            typer.echo(f"    {i}. {step}")
    if d.access.credentials:
        typer.echo(f"  Credentials: {', '.join(d.access.credentials)}")
    if d.notes:
        typer.echo("\n  Notes:")
        for note in d.notes:
            typer.echo(f"    - {note.strip()}")
    if dest:
        _print_downloads(_store(dest, endpoint), d.id, root, dest)


def _info_rows(d: Dataset) -> list[tuple[str, str]]:
    stats = d.stats
    approx = "~" if stats.hours_estimated else ""
    size_parts = [
        f"{stats.subjects:,} subjects" if stats.subjects is not None else "",
        f"{stats.recordings:,} recordings" if stats.recordings is not None else "",
        f"{approx}{stats.hours:,.0f} h" if stats.hours is not None else "",
        _size(stats.size_bytes) if stats.size_bytes is not None else "",
    ]
    version = f" · version {d.pinned_version}" if d.pinned_version else ""
    return [
        ("signals", ", ".join(d.signals)),
        ("size", " · ".join(p for p in size_parts if p) or "?"),
        ("access", f"{d.access.tier.value} · license {d.access.license or '?'}"),
        ("source", d.source.type + version),
        ("homepage", str(d.homepage) if d.homepage else ""),
    ]


def _print_downloads(store: Store, dataset_id: str, root: str, dest: str) -> None:
    versions = stored_versions(store, dataset_id, root)
    typer.echo(f"\n  Downloaded to {dest}:" if versions else f"\n  Not downloaded to {dest}.")
    for version, complete in versions.items():
        location = VersionLocation(dataset_id, version, root)
        data = store.get_bytes(location.manifest_key if complete else location.partial_manifest_key)
        summary = ""
        if data:
            files = Manifest.from_json(data).files
            summary = f"{len(files):,} files, {_size(sum(f.size for f in files))}"
        state = "complete" if complete else "incomplete (subset or interrupted)"
        typer.echo(f"    {version:14}{state:36}{summary}")


# --- download --------------------------------------------------------------------------


@app.command()
def download(
    dataset_ids: Annotated[
        list[str] | None, typer.Argument(help="Catalog ids. Or select with the options below.")
    ] = None,
    dest: DestOpt = None,
    endpoint: EndpointOpt = None,
    all_: Annotated[bool, typer.Option("--all", help="Every dataset in the catalog.")] = False,
    signal: Annotated[
        list[str], typer.Option("--signal", "-s", help="Datasets with this signal. Repeatable.")
    ] = [],  # noqa: B006
    repository: Annotated[
        list[str],
        typer.Option("--repository", "-r", help="Datasets from this repository. Repeatable."),
    ] = [],  # noqa: B006
    ids_file: Annotated[
        Path | None,
        typer.Option(help="A file with one catalog id per line (# starts a comment)."),
    ] = None,
    limit: LimitOpt = None,
    include: IncludeOpt = [],  # noqa: B006
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be downloaded; download nothing.")
    ] = False,
    workers: Annotated[
        int, typer.Option(min=1, help="Parallel file downloads, shared by all datasets.")
    ] = 16,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="More log output.")] = False,
    from_dir: Annotated[
        Path | None,
        typer.Option(help="One dataset: take the files from this folder (downloaded by hand)."),
    ] = None,
    version: Annotated[
        str | None, typer.Option(help="One dataset: version label (default: latest).")
    ] = None,
    source_prefix: Annotated[
        str | None, typer.Option(help="One dataset: key prefix in the source bucket (BDSP).")
    ] = None,
    staging_dir: Annotated[
        Path | None,
        typer.Option(help="Folder for rsync/nsrr/kaggle downloads; reused to resume."),
    ] = None,
    keep_staging: Annotated[bool, typer.Option(help="Keep the staging folder.")] = False,
    aws_profile: Annotated[
        str | None, typer.Option(help="AWS profile for reading source buckets.")
    ] = None,
    root: RootOpt = "raw",
    catalog: CatalogOpt = None,
) -> None:
    """Download datasets, unchanged, to --dest.

    Select datasets by id, with --signal / --repository, with --ids-file, or with --all.
    Many datasets download at the same time. Datasets that need you to act first
    (credentials, download by hand) are skipped with a warning and listed at the end.

    Files go to <dest>/raw/<id>/<version>/data/, with a manifest of checksums. If a download
    stops, run the same command again: it continues where it stopped.

    --limit and --include download a test subset. The version then stays incomplete; the
    same command without them later completes it without downloading the subset again.
    """
    setup_logging(verbose)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids or [], all_, signal, repository, ids_file
    )
    if len(datasets) > 1 and (from_dir or version or source_prefix):
        raise _fail("--from-dir, --version and --source-prefix apply to one dataset only")
    options = FetchOptions(
        staging_dir=staging_dir,
        from_dir=from_dir,
        source_prefix=source_prefix,
        version=version,
        aws_profile=aws_profile,
        keep_staging=keep_staging,
    )
    if dry_run:
        for dataset in datasets:
            _preview(dataset, options, include, limit)
        return

    store = _store(dest, endpoint)
    console.print(f"[bold]{len(datasets)} datasets[/] -> {store.uri}/{root}", highlight=False)
    console.print(f"[yellow]{_data_path(store)}[/]", highlight=False, soft_wrap=True)
    with download_progress() as progress:
        run = RunBar(progress, len(datasets)) if len(datasets) > 1 else None

        def progress_for(dataset: Dataset) -> ProgressBar:
            return ProgressBar(progress, dataset.id, run=run, show_id=run is not None)

        bars: dict[str, ProgressBar] = {}

        def progress_for_tracked(dataset: Dataset) -> ProgressBar:
            bars[dataset.id] = bar = progress_for(dataset)
            return bar

        def on_done(outcome: Outcome) -> None:
            bar = bars.pop(outcome.dataset.id, None)
            if bar:
                bar.finish()
            if run:
                run.dataset_done(outcome.status in DONE)
            _print_outcome(outcome)

        outcomes = download_many(
            datasets,
            store,
            options=options,
            root=root,
            workers=workers,
            include=include,
            limit=limit,
            progress_for=progress_for_tracked,
            on_done=on_done,
        )
    _print_summary(outcomes)
    raise typer.Exit(0 if all(o.status in DONE for o in outcomes) else 1)


def _select_datasets(
    cat: Catalog,
    ids: list[str],
    all_: bool,
    signal: list[str],
    repository: list[str],
    ids_file: Path | None,
) -> list[Dataset]:
    if ids_file is not None:
        lines = ids_file.read_text(encoding="utf-8").splitlines()
        ids = ids + [ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()]
    if not (ids or all_ or signal or repository):
        raise _fail("say which datasets: ids, --ids-file, --signal, --repository or --all")
    chosen = {d.id: d for d in (_dataset(cat, d_id) for d_id in ids)}
    if all_ or signal or repository:
        for d in cat.filter(signals=signal, repositories=repository):
            chosen.setdefault(d.id, d)
    return list(chosen.values())


_OUTCOME_STYLE = {
    "complete": ("✓", "green"),
    "subset": ("✓", "green"),
    "already": ("•", "yellow"),
    "incomplete": ("✗", "red"),
    "credentials": ("!", "yellow"),
    "manual": ("!", "yellow"),
    "unavailable": ("!", "yellow"),
    "failed": ("✗", "red"),
}


def _print_outcome(outcome: Outcome) -> None:
    mark, color = _OUTCOME_STYLE[outcome.status]
    d, r = outcome.dataset, outcome.report
    if r is not None:
        text = (
            f"{d.id} {r.version}: {outcome.status} · {r.recordings:,} recordings · "
            f"{r.copied:,} files ({_size(r.bytes_copied)}) downloaded"
        )
        if r.skipped:
            text += f" · {r.skipped:,} already there"
        if r.failed:
            text += f" · {len(r.failed):,} files failed (run again to retry)"
    else:
        label = {
            "already": "skipped, already complete",
            "credentials": "skipped, needs credentials",
            "manual": "skipped, download by hand",
            "unavailable": "skipped, not available",
            "failed": "failed",
        }[outcome.status]
        text = f"{d.id}: {label}: {outcome.message}"
    console.print(f"[{color}]{mark}[/] {text}", markup=True, highlight=False, soft_wrap=True)


def _print_summary(outcomes: list[Outcome]) -> None:
    by_status: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_status.setdefault(o.status, []).append(o)
    done = sum(len(by_status.get(s, [])) for s in DONE)
    action = sum(len(by_status.get(s, [])) for s in NEEDS_ACTION)
    failed = len(by_status.get("failed", [])) + len(by_status.get("incomplete", []))
    console.print(
        f"\n[bold]Done:[/] {done} downloaded · [yellow]{action} need your action[/] · "
        f"[red]{failed} failed[/]",
        highlight=False,
    )
    groups = [
        ("Skipped, already complete", "already"),
        ("Need credentials (see `biosignal-data info <id>`)", "credentials"),
        ("Download by hand, then --from-dir (see `biosignal-data info <id>`)", "manual"),
        ("Not available", "unavailable"),
        ("Failed", "failed"),
        ("Incomplete, some files failed (run again to retry)", "incomplete"),
    ]
    for title, status in groups:
        items = by_status.get(status, [])
        if not items:
            continue
        color = "red" if status in ("failed", "incomplete") else "yellow"
        console.print(f"[{color}]{title}:[/] {len(items)}", highlight=False)
        for o in sorted(items, key=lambda o: o.dataset.id):
            detail = (
                o.message
                if status != "incomplete"
                else f"{len(o.report.failed) if o.report else 0} files failed"
            )
            console.print(f"  {o.dataset.id}  [dim]{detail}[/]", highlight=False, soft_wrap=True)


def _preview(
    dataset: Dataset, options: FetchOptions, include: list[str], limit: int | None
) -> None:
    fetcher = fetcher_for(dataset, options)
    try:
        fetcher.check_access()
        fetcher.prepare()
        resolved = fetcher.resolve_version()
        files = {f.path: f for f in fetcher.list_files()}
    except FetchError as exc:
        console.print(f"[yellow]![/] {dataset.id}: {exc}", highlight=False, soft_wrap=True)
        return
    finally:
        fetcher.cleanup()
    typer.secho(f"{dataset.id} {resolved.value}", bold=True)
    typer.echo(f"  whole dataset   {_describe(list(files.values()))}")
    if not include and limit is None:
        return
    selection = select(list(files), include, limit)
    chosen = [files[p] for p in selection.paths]
    typer.echo(f"  would download  {len(selection.recordings)} recordings, {_describe(chosen)}")
    for f in chosen:
        typer.echo(f"    {_size(f.size):>10}  {f.path}")


def _describe(files: list[RemoteFile]) -> str:
    sizes = [f.size for f in files if f.size is not None]
    unknown = f" ({len(files) - len(sizes)} sizes unknown)" if len(sizes) < len(files) else ""
    return f"{len(files):,} files, {_size(sum(sizes))}{unknown}"


# --- check -----------------------------------------------------------------------------


@app.command()
def check(
    dataset_id: Annotated[str, typer.Argument(help="Catalog id.")],
    dest: DestOpt = None,
    endpoint: EndpointOpt = None,
    version: Annotated[
        str | None, typer.Option(help="Version to check (default: the latest downloaded).")
    ] = None,
    checksums: Annotated[
        bool,
        typer.Option(
            "--checksums", help="Also read every file again and compare its SHA-256 (slow)."
        ),
    ] = False,
    root: RootOpt = "raw",
) -> None:
    """Check that the downloaded files are all there, with the right size (and checksum)."""
    store = _store(dest, endpoint)
    versions = stored_versions(store, dataset_id, root)
    if version is None:
        if not versions:
            raise _fail(f"{dataset_id} is not downloaded to {dest}")
        complete = [v for v, done in versions.items() if done]
        version = complete[-1] if complete else list(versions)[-1]
        kind = "latest complete" if complete else "latest (incomplete)"
        _warn(f"no --version given; checking the {kind} version {version}")
    problems = verify(store, dataset_id, version, root, deep=checksums)
    for path, problem in sorted(problems.items()):
        typer.secho(f"  {path}: {problem}", fg="red")
    note = "" if versions.get(version) else " (incomplete version: checked the files so far)"
    if problems:
        raise _fail(f"{dataset_id} {version}: {len(problems)} problems{note}")
    what = "sizes and SHA-256" if checksums else "sizes"
    typer.secho(f"{dataset_id} {version}: ok, {what} match the manifest{note}", fg="green")


# --- dev -------------------------------------------------------------------------------


@dev.command()
def validate(catalog: CatalogOpt = None) -> None:
    """Validate every catalog file."""
    typer.echo(f"{len(_catalog(catalog))} catalog entries are valid")


@dev.command()
def docs(
    catalog: CatalogOpt = None,
    out: Annotated[Path, typer.Option(help="Output folder.")] = Path("docs/datasets"),
    check: Annotated[bool, typer.Option(help="Fail if the generated docs are stale.")] = False,
) -> None:
    """Generate the dataset pages in docs/datasets/ from the catalog."""
    from biosignal_data.docsgen import write_docs  # noqa: PLC0415

    stale = write_docs(_catalog(catalog), out, check=check)
    if check and stale:
        raise _fail("stale docs: " + ", ".join(stale) + "; run `biosignal-data dev docs`")
    typer.echo(f"docs {'checked' if check else 'written'} in {out}")


def main() -> None:  # pragma: no cover
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(app())

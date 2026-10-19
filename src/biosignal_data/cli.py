"""Command-line interface: `biosignal-data --help`.

    list       find datasets in the catalog
    info       one catalog entry: what it is and how to get access
    download   copy datasets (or a test subset) from their sources to --dest
    import     copy a dataset you downloaded by hand from a folder to --dest
    status     what is stored in --dest
    check      check stored files against their manifests
    dev        maintainer commands (catalog validation, generated docs)

Settings that do not change between runs come from biosignal-data.yaml (see config.py);
secrets from BIOSIGNAL_* environment variables. Nothing has a default.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from rich.markup import escape

from biosignal_data.batch import DONE, NEEDS_ACTION, Outcome, download_many
from biosignal_data.catalog import AccessTier, Catalog, CatalogError, Dataset
from biosignal_data.config import (
    Config,
    ConfigError,
    DownloadConfig,
    default_config_path,
    load_config,
    need,
)
from biosignal_data.console import DownloadDisplay, ProgressBar, console, setup_logging
from biosignal_data.fetchers import FetchError, FetchOptions, RemoteFile, fetcher_for
from biosignal_data.ingest import verify
from biosignal_data.read.raw import stored_versions
from biosignal_data.selection import select
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import Manifest
from biosignal_data.storage.stores import S3Settings, S3Store, Store, open_store
from biosignal_data.text import plural

ROOT = "raw"  # the layer folder inside the destination
T = TypeVar("T")

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,  # help texts are plain: "[/prefix]" is not markup
    help="Find public biosignal datasets and copy them unchanged to a folder or S3 storage.",
)
dev = typer.Typer(no_args_is_help=True, help="Maintainer commands.")
app.add_typer(dev, name="dev")


# --- shared options ------------------------------------------------------------------------

CatalogOpt = Annotated[
    Path | None,
    typer.Option("--catalog", help="Catalog folder. Default: the repository's catalog/."),
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", help=f"Settings file. Default: {default_config_path()}."),
]
_DEST_HELP = "A local folder or s3://bucket[/prefix]."
DestOpt = Annotated[str, typer.Option("--dest", "-d", help=_DEST_HELP)]
IdsArg = Annotated[list[str] | None, typer.Argument(help="Catalog ids, e.g. physionet/mitdb.")]
AllOpt = Annotated[bool, typer.Option("--all", help="Every dataset in the catalog.")]
SignalOpt = Annotated[
    list[str], typer.Option("--signal", "-s", help="Datasets with this signal. Repeatable.")
]
RepositoryOpt = Annotated[
    list[str],
    typer.Option("--repository", "-r", help="Datasets from this repository. Repeatable."),
]
IdsFileOpt = Annotated[
    Path | None,
    typer.Option("--ids-file", help="A file with one catalog id per line (# starts a comment)."),
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="More log output.")]


def _fail(message: str, code: int = 1) -> typer.Exit:
    console.print(f"[red]error:[/] {escape(message)}", highlight=False, soft_wrap=True)
    return typer.Exit(code)


def _warn(message: str) -> None:
    console.print(f"[yellow]warning:[/] {escape(message)}", highlight=False, soft_wrap=True)


def _catalog(path: Path | None) -> Catalog:
    try:
        return Catalog.load(path)
    except CatalogError as exc:
        raise _fail(str(exc), 2) from exc


def _config(path: Path | None) -> tuple[Config, Path]:
    try:
        return load_config(path)
    except ConfigError as exc:
        raise _fail(str(exc), 2) from exc


def _need(section: T | None, name: str, path: Path) -> T:
    try:
        return need(section, name, path)
    except ConfigError as exc:
        raise _fail(str(exc), 2) from exc


def _dataset(catalog: Catalog, dataset_id: str) -> Dataset:
    try:
        return catalog.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc.args[0] if exc.args else exc)) from exc


def _select_datasets(
    cat: Catalog,
    ids: list[str] | None,
    ids_file: Path | None,
    signal: list[str],
    repository: list[str],
    all_: bool,
) -> list[Dataset]:
    """Exactly one way to select: ids (arguments and/or --ids-file), filters (--signal,
    --repository; all filters must match), or --all."""
    ids = list(ids or [])
    if ids_file is not None:
        lines = ids_file.read_text(encoding="utf-8").splitlines()
        ids += [ln.split("#", 1)[0].strip() for ln in lines if ln.split("#", 1)[0].strip()]
    ways = [bool(ids), bool(signal or repository), all_]
    if sum(ways) != 1:
        raise _fail(
            "select datasets one way: ids (or --ids-file), or --signal/--repository, or --all"
        )
    if ids:
        return list({d.id: d for d in (_dataset(cat, i) for i in ids)}.values())
    if all_:
        return list(cat)
    return cat.filter(signals=signal, repositories=repository)


def _store(dest: str, config: Config, config_path: Path, parallel_uploads: int = 1) -> Store:
    """Open --dest. An s3:// destination uses the config's `s3` section and the keys in
    BIOSIGNAL_ACCESS_KEY_ID / BIOSIGNAL_SECRET_ACCESS_KEY."""
    try:
        if not dest.startswith("s3://"):
            return open_store(dest)
        s3 = _need(config.s3, "s3", config_path)
        settings = S3Settings.from_env(s3.endpoint, s3.region, s3.addressing)
        return open_store(dest, settings, parallel_uploads)
    except ValueError as exc:
        raise _fail(str(exc)) from exc


def _describe_store(store: Store) -> str:
    if isinstance(store, S3Store) and store.settings:
        s3 = store.settings
        return f"{store.uri} at {s3.endpoint} (region {s3.region}, {s3.addressing} addressing)"
    return store.uri


def _size(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} TB"


# --- list ----------------------------------------------------------------------------------


@app.command("list")
def list_(
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    access: Annotated[
        list[AccessTier], typer.Option("--access", "-a", help="Access tier. Repeatable.")
    ] = [],  # noqa: B006
    as_json: Annotated[bool, typer.Option("--json", help="Full entries as JSON.")] = False,
    catalog: CatalogOpt = None,
) -> None:
    """Find datasets in the catalog. Without filters: all of them."""
    datasets = _catalog(catalog).filter(signals=signal, tiers=access, repositories=repository)
    if as_json:
        typer.echo(json.dumps([d.model_dump(mode="json") for d in datasets], indent=2))
        return
    for d in datasets:
        hours = f"{d.stats.hours:,.0f} h" if d.stats.hours is not None else "? h"
        typer.echo(f"{d.id:55} {d.access.tier.value:13} {','.join(d.signals):20} {hours:>12}")
    console.print(f"{len(datasets)} datasets", highlight=False)


# --- info ----------------------------------------------------------------------------------


@app.command()
def info(
    dataset_id: Annotated[str, typer.Argument(help="Catalog id, e.g. physionet/mitdb.")],
    as_json: Annotated[bool, typer.Option("--json", help="The catalog entry as JSON.")] = False,
    catalog: CatalogOpt = None,
) -> None:
    """One catalog entry: what the dataset is and how to get access to it."""
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


def _info_rows(d: Dataset) -> list[tuple[str, str]]:
    stats = d.stats
    approx = "~" if stats.hours_estimated else ""
    size_parts = [
        f"{stats.subjects:,} subjects" if stats.subjects is not None else "",
        f"{stats.recordings:,} recordings" if stats.recordings is not None else "",
        f"{approx}{stats.hours:,.0f} h" if stats.hours is not None else "",
        _size(stats.size_bytes) if stats.size_bytes is not None else "",
    ]
    version = f" · version {d.pinned_version}" if d.pinned_version else " · version not pinned"
    return [
        ("signals", ", ".join(d.signals)),
        ("size", " · ".join(p for p in size_parts if p) or "?"),
        ("access", f"{d.access.tier.value} · license {d.access.license or '?'}"),
        ("source", d.source.type + version),
        ("homepage", str(d.homepage) if d.homepage else ""),
    ]


# --- download / import ---------------------------------------------------------------------


@app.command()
def download(
    dataset_ids: IdsArg = None,
    dest: Annotated[
        str | None,
        typer.Option("--dest", "-d", help=_DEST_HELP + " Not with --dry-run."),
    ] = None,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            "-n",
            min=1,
            help="Test subset: the first N recordings of each dataset, with the files and "
            "metadata each one needs. A dataset published as one archive has one recording.",
        ),
    ] = None,
    include: Annotated[
        list[str],
        typer.Option(
            "--include",
            "-i",
            help="Test subset: only files whose path matches this glob (* within a folder, "
            "** across folders). Repeatable.",
        ),
    ] = [],  # noqa: B006
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be copied; copy nothing, no --dest."),
    ] = False,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
) -> None:
    """Copy datasets from their sources, unchanged, to --dest.

    Select datasets by ids, --ids-file, --signal/--repository, or --all. Many datasets copy
    at the same time. Datasets that need you to act first (credentials, download by hand)
    are skipped with a warning and listed at the end.

    Files go to <dest>/raw/<id>/<version>/data/ with a manifest of checksums. If a run
    stops, run the same command again: it continues where it stopped.

    A test subset (--limit, --include) stays an incomplete version; the same command
    without them later completes it without copying the subset again.
    """
    setup_logging(verbose)
    cfg, cfg_path = _config(config)
    settings = _need(cfg.download, "download", cfg_path)
    datasets = _select_datasets(_catalog(catalog), dataset_ids, ids_file, signal, repository, all_)
    staging = cfg.staging
    options = FetchOptions(
        staging_dir=staging.dir if staging else None,
        keep_staging=staging.keep if staging else None,
        connections=settings.workers,
    )
    console.print(
        f"settings from {cfg_path}: workers {settings.workers}, parallel_datasets "
        f"{settings.parallel_datasets}",
        highlight=False,
        soft_wrap=True,
    )
    if dry_run:
        if dest:
            raise _fail("--dry-run copies nothing: leave out --dest")
        for dataset in datasets:
            _preview(dataset, options, include, limit)
        return
    if not dest:
        raise _fail("missing --dest (a local folder or s3://bucket[/prefix])")
    store = _store(dest, cfg, cfg_path, settings.workers)
    _run(datasets, store, options, settings, include, limit)


@app.command("import")
def import_(
    dataset_id: Annotated[str, typer.Argument(help="Catalog id.")],
    folder: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, help="The folder you downloaded into."),
    ],
    dest: DestOpt,
    version: Annotated[
        str,
        typer.Option("--version", help="The version label, as the source names it."),
    ],
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
) -> None:
    """Copy a dataset you downloaded by hand from FOLDER, unchanged, to --dest."""
    setup_logging(verbose)
    cfg, cfg_path = _config(config)
    dataset = _dataset(_catalog(catalog), dataset_id)
    settings = _need(cfg.download, "download", cfg_path)
    store = _store(dest, cfg, cfg_path, settings.workers)
    options = FetchOptions(from_dir=folder, version=version)
    _run([dataset], store, options, settings, include=[], limit=None)


def _run(
    datasets: list[Dataset],
    store: Store,
    options: FetchOptions,
    settings: DownloadConfig,
    include: list[str],
    limit: int | None,
) -> None:
    console.print(
        f"[bold]{plural(len(datasets), 'dataset')}[/] -> {store.uri}/{ROOT}", highlight=False
    )
    console.print(
        f"[yellow]data flows: sources -> this machine ({socket.gethostname()}) -> "
        f"{_describe_store(store)}[/]",
        highlight=False,
        soft_wrap=True,
    )
    _warn_unfinished(store, store.uri)
    try:
        outcomes = _copy(datasets, store, options, settings, include, limit)
    except KeyboardInterrupt:
        console.print(
            "[yellow]stopped.[/] Files copied so far are kept; run the same command again "
            "to continue.",
            highlight=False,
        )
        raise typer.Exit(130) from None
    _print_summary(outcomes)
    raise typer.Exit(0 if all(o.status in DONE for o in outcomes) else 1)


def _copy(
    datasets: list[Dataset],
    store: Store,
    options: FetchOptions,
    settings: DownloadConfig,
    include: list[str],
    limit: int | None,
) -> list[Outcome]:
    with DownloadDisplay(len(datasets)) as display:
        bars: dict[str, ProgressBar] = {}

        def progress_for(dataset: Dataset) -> ProgressBar:
            bars[dataset.id] = bar = display.bar(dataset.id)
            return bar

        def on_done(outcome: Outcome) -> None:
            bar = bars.pop(outcome.dataset.id, None)
            if bar:
                bar.finish()
            display.dataset_done(outcome.status in DONE)
            _print_outcome(outcome)

        return download_many(
            datasets,
            store,
            options=options,
            root=ROOT,
            workers=settings.workers,
            parallel_datasets=settings.parallel_datasets,
            include=include,
            limit=limit,
            progress_for=progress_for,
            on_done=on_done,
        )


_OUTCOME_STYLE = {
    "complete": ("✓", "green"),
    "subset": ("✓", "green"),
    "already": ("•", "yellow"),
    "incomplete": ("✗", "red"),
    "credentials": ("!", "yellow"),
    "restricted": ("!", "yellow"),
    "manual": ("!", "yellow"),
    "unavailable": ("!", "yellow"),
    "failed": ("✗", "red"),
}


def _print_outcome(outcome: Outcome) -> None:
    mark, color = _OUTCOME_STYLE[outcome.status]
    d, r = outcome.dataset, outcome.report
    if r is not None:
        text = (
            f"{d.id} {r.version}: {outcome.status} · {plural(r.recordings, 'recording')} · "
            f"{plural(r.copied, 'file')} ({_size(r.bytes_copied)}) copied"
        )
        if r.skipped:
            text += f" · {r.skipped:,} already there"
        if r.failed:
            text += f" · {plural(len(r.failed), 'file')} failed (run again to retry)"
    else:
        label = {
            "already": "skipped, already complete",
            "credentials": "skipped, needs credentials",
            "restricted": "skipped, access restricted by the source",
            "manual": "skipped, download by hand",
            "unavailable": "skipped, not available",
            "failed": "failed",
        }[outcome.status]
        text = f"{d.id}: {label}: {outcome.message}"
    console.print(f"[{color}]{mark}[/] {text}", highlight=False, soft_wrap=True)


def _print_summary(outcomes: list[Outcome]) -> None:
    by_status: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_status.setdefault(o.status, []).append(o)
    _warn_untested(outcomes)
    done = sum(len(by_status.get(s, [])) for s in DONE)
    action = sum(len(by_status.get(s, [])) for s in NEEDS_ACTION)
    failed = len(by_status.get("failed", [])) + len(by_status.get("incomplete", []))
    console.print(
        f"\n[bold]Done:[/] {done} copied · [yellow]{action} need your action[/] · "
        f"[red]{failed} failed[/]",
        highlight=False,
    )
    groups = [
        ("Skipped, already complete", "already"),
        ("Need credentials (see `biosignal-data info <id>`)", "credentials"),
        ("Access restricted by the source (request access on its page)", "restricted"),
        ("Download by hand, then `biosignal-data import`", "manual"),
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
                f"{len(o.report.failed)} files failed"
                if status == "incomplete" and o.report
                else o.message
            )
            console.print(f"  {o.dataset.id}  [dim]{detail}[/]", highlight=False, soft_wrap=True)


def _warn_untested(outcomes: list[Outcome]) -> None:
    """Source types whose download code did not run at all, because every dataset of that
    type was skipped before the first request."""
    ran = {o.dataset.source.type for o in outcomes if o.status not in NEEDS_ACTION}
    skipped: dict[str, list[Outcome]] = {}
    for o in outcomes:
        if o.status in NEEDS_ACTION and o.dataset.source.type not in ran:
            skipped.setdefault(o.dataset.source.type, []).append(o)
    for source_type, items in sorted(skipped.items()):
        reasons = sorted({o.status for o in items})
        _warn(
            f"source type {source_type!r}: all {len(items)} datasets skipped "
            f"({', '.join(reasons)}); its download code was not tested in this run"
        )


def _preview(
    dataset: Dataset, options: FetchOptions, include: list[str], limit: int | None
) -> None:
    fetcher = fetcher_for(dataset, options)
    try:
        fetcher.check_access()
        fetcher.prepare()
        resolved = fetcher.resolve_version()
        files = {f.path: f for f in fetcher.list_files()}
    except FetchError as exc:  # needs action, or the source refused: say so, go on
        console.print(
            f"[yellow]![/] {dataset.id}: {escape(str(exc))}", highlight=False, soft_wrap=True
        )
        return
    except (OSError, RuntimeError, ValueError) as exc:  # as in a real run: report, go on
        console.print(
            f"[red]✗[/] {dataset.id}: {type(exc).__name__}: {escape(str(exc))}",
            highlight=False,
            soft_wrap=True,
        )
        return
    finally:
        fetcher.cleanup(complete=False)
    typer.secho(f"{dataset.id} {resolved.value}", bold=True)
    typer.echo(f"  whole dataset   {_describe(list(files.values()))}")
    if not include and limit is None:
        return
    selection = select(list(files), include, limit)
    chosen = [files[p] for p in selection.paths]
    typer.echo(
        f"  would copy      {plural(len(selection.recordings), 'recording')}, {_describe(chosen)}"
    )
    for f in chosen:
        typer.echo(f"    {_size(f.size):>10}  {f.path}")


def _describe(files: list[RemoteFile]) -> str:
    sizes = [f.size for f in files if f.size is not None]
    unknown = f" ({len(files) - len(sizes)} sizes unknown)" if len(sizes) < len(files) else ""
    return f"{plural(len(files), 'file')}, {_size(sum(sizes))}{unknown}"


# --- status / check ------------------------------------------------------------------------


@app.command()
def status(
    dest: DestOpt,
    dataset_ids: IdsArg = None,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
) -> None:
    """What is stored in --dest: versions, complete or not, files and size."""
    cfg, cfg_path = _config(config)
    datasets = _select_datasets(_catalog(catalog), dataset_ids, ids_file, signal, repository, all_)
    store = _store(dest, cfg, cfg_path)
    missing = 0
    for d in datasets:
        versions = stored_versions(store, d.id, ROOT)
        if not versions:
            missing += 1
            if not all_:
                console.print(f"  {d.id}  [dim]not stored[/]", highlight=False)
            continue
        for version, complete in versions.items():
            location = VersionLocation(d.id, version, ROOT)
            key = location.manifest_key if complete else location.partial_manifest_key
            data = store.get_bytes(key)
            files = Manifest.from_json(data).files if data else []
            state = "complete" if complete else "[yellow]incomplete (subset or interrupted)[/]"
            console.print(
                f"  {d.id}  {version}  {state}  {len(files):,} files, "
                f"{_size(sum(f.size for f in files))}",
                highlight=False,
            )
    console.print(f"{len(datasets) - missing} of {len(datasets)} datasets stored", highlight=False)
    _warn_unfinished(store, dest)


@app.command()
def check(
    dest: DestOpt,
    dataset_ids: IdsArg = None,
    version: Annotated[
        str | None, typer.Option("--version", help="This version (one dataset only).")
    ] = None,
    latest: Annotated[
        bool, typer.Option("--latest", help="The newest stored version of each dataset.")
    ] = False,
    checksums: Annotated[
        bool,
        typer.Option("--checksums", help="Also read every file again and compare its SHA-256."),
    ] = False,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
) -> None:
    """Check that stored files are all there with the size (and SHA-256) of the manifest.
    An incomplete version is checked against the files copied so far."""
    if (version is None) == (not latest):
        raise _fail("give either --version <version> or --latest")
    cfg, cfg_path = _config(config)
    datasets = _select_datasets(_catalog(catalog), dataset_ids, ids_file, signal, repository, all_)
    if version is not None and len(datasets) != 1:
        raise _fail("--version applies to one dataset; use --latest for several")
    store = _store(dest, cfg, cfg_path)
    bad = 0
    for d in datasets:
        versions = stored_versions(store, d.id, ROOT)
        chosen = version or (list(versions)[-1] if versions else None)
        if chosen is None:
            if not all_:
                console.print(f"[yellow]![/] {d.id}: not stored", highlight=False)
            continue
        problems = verify(store, d.id, chosen, ROOT, deep=checksums)
        note = "" if versions.get(chosen) else " (incomplete: files copied so far)"
        if problems:
            bad += 1
            console.print(f"[red]✗[/] {d.id} {chosen}: {len(problems)} problems{note}")
            for path, problem in sorted(problems.items()):
                console.print(f"    {path}: {problem}", highlight=False)
        else:
            what = "sizes and SHA-256" if checksums else "sizes"
            console.print(f"[green]✓[/] {d.id} {chosen}: {what} ok{note}", highlight=False)
    raise typer.Exit(1 if bad else 0)


def _warn_unfinished(store: Store, dest: str) -> None:
    left = store.unfinished(ROOT)
    if left:
        _warn(
            f"{plural(len(left), 'unfinished upload')} ({_size(sum(u.size for u in left))}) in "
            f"{dest}: left by a stopped run; they use storage but are not files. Remove them "
            f"with `biosignal-data cleanup --dest {dest}` when no run is copying there."
        )


@app.command()
def cleanup(
    dest: DestOpt,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List the unfinished uploads; remove nothing.")
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Remove unfinished uploads that stopped runs left in --dest (S3 multipart uploads,
    temporary .part files). They use storage but are not files. Run it only when no
    download to --dest is in progress: it would remove that run's uploads too."""
    cfg, cfg_path = _config(config)
    store = _store(dest, cfg, cfg_path)
    left = store.unfinished(ROOT)
    for item in left:
        started = item.started.isoformat(timespec="seconds") if item.started else "?"
        console.print(f"  {item.key}  {_size(item.size)}  started {started}", highlight=False)
    total = f"{plural(len(left), 'unfinished upload')} ({_size(sum(u.size for u in left))})"
    if dry_run or not left:
        console.print(f"{total} in {dest}", highlight=False)
        return
    for item in left:
        store.discard(item)
    console.print(f"[green]removed {total}[/] from {dest}", highlight=False)


# --- dev -----------------------------------------------------------------------------------


@dev.command()
def validate(catalog: CatalogOpt = None) -> None:
    """Validate every catalog file."""
    typer.echo(f"{len(_catalog(catalog))} catalog entries are valid")


@dev.command()
def docs(
    out: Annotated[Path, typer.Option(help="Output folder.")] = Path("docs/datasets"),
    check: Annotated[bool, typer.Option(help="Fail if the generated docs are stale.")] = False,
    catalog: CatalogOpt = None,
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

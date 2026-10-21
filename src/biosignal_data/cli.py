"""Command-line interface: `biosignal-data --help`.

    list       find datasets in the catalog
    info       one catalog entry: what it is and how to get access
    download   copy datasets from their sources to --dest (or show the plan: --dry-run)
    import     copy a dataset you downloaded by hand from a folder to --dest
    status     what is stored in --dest
    check      check stored files against their manifests
    standardize  convert raw versions in --dest to the standard layer (Zarr + Parquet)
    index      build the training index (subjects, splits, signal types) from the standard layer
    quality    measure each channel of the standard layer (flat, invalid, limits, values)
    cleanup    remove unfinished uploads that stopped runs left in --dest
    dev        maintainer commands (validation, generated docs)

The datasets are listed in catalog.yaml (what a download needs) and described in
docs/datasets.yaml (for people; read only by list, info, --signal and the docs).

Settings that do not change between runs come from biosignal-data.yaml (see config.py);
secrets from BIOSIGNAL_* environment variables. Nothing has a default.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from rich.markup import escape

from biosignal_data.batch import DONE, NEEDS_ACTION, Outcome, download_many
from biosignal_data.catalog import (
    AccessTier,
    Catalog,
    CatalogError,
    Dataset,
    DatasetInfo,
    Infos,
    check_pair,
    default_catalog_path,
    default_info_path,
)
from biosignal_data.config import (
    Config,
    ConfigError,
    DownloadConfig,
    StandardizeConfig,
    default_config_path,
    load_config,
    need,
)
from biosignal_data.console import (
    DownloadDisplay,
    ProgressBar,
    StandardizeBar,
    console,
    setup_logging,
)
from biosignal_data.fetchers import FetchError, FetchOptions
from biosignal_data.fetchers.base import MissingCredentialsError, require_env
from biosignal_data.index import AlreadyIndexed, IndexRules, build_index
from biosignal_data.index.rules import RulesError, default_rules_path
from biosignal_data.index.subjects import SubjectError
from biosignal_data.ingest import AlreadyComplete, Plan, planned, verify
from biosignal_data.read.raw import complete_versions, stored_versions
from biosignal_data.standard.convert import (
    AlreadyStandard,
    StandardPlan,
    convert,
    iter_not_converted,
    plan_standard,
)
from biosignal_data.standard.layout import StandardLocation
from biosignal_data.standard.manifest import StandardManifest
from biosignal_data.storage.layout import VersionLocation
from biosignal_data.storage.manifest import Manifest
from biosignal_data.storage.stores import S3Settings, S3Store, Store, StoreFactory, open_store
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
    typer.Option("--catalog", help=f"The dataset list. Default: {default_catalog_path()}."),
]
DescriptionsOpt = Annotated[
    Path | None,
    typer.Option(
        "--descriptions", help=f"The dataset descriptions. Default: {default_info_path()}."
    ),
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
    list[str],
    typer.Option(
        "--signal", "-s", help="Datasets with this signal (from the descriptions). Repeatable."
    ),
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
ReplaceOpt = Annotated[
    bool,
    typer.Option("--replace", help="Make complete versions again instead of skipping them."),
]
AddMissingOpt = Annotated[
    bool,
    typer.Option(
        "--add-missing",
        help="For complete versions: list the source again and copy only the files they lack.",
    ),
]


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


def _infos(path: Path | None) -> Infos:
    try:
        return Infos.load(path)
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


def _info(infos: Infos, dataset_id: str) -> DatasetInfo:
    try:
        return infos.get(dataset_id)
    except KeyError as exc:
        raise _fail(str(exc.args[0] if exc.args else exc)) from exc


def _select_datasets(
    cat: Catalog,
    ids: list[str] | None,
    ids_file: Path | None,
    signal: list[str],
    repository: list[str],
    all_: bool,
    descriptions: Path | None,
) -> list[Dataset]:
    """Exactly one way to select: ids (arguments and/or --ids-file), filters (--signal,
    --repository; all filters must match), or --all. Only --signal reads the descriptions."""
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
    chosen = cat.filter(repositories=repository)
    if signal:
        with_signal = {d.id for d in _infos(descriptions).filter(signals=signal)}
        chosen = [d for d in chosen if d.id in with_signal]
    return chosen


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


def _factory(dest: str, config: Config, config_path: Path) -> StoreFactory:
    """Opens --dest again in each worker process (keys from the environment)."""
    if not dest.startswith("s3://"):
        return StoreFactory(dest, None)
    s3 = _need(config.s3, "s3", config_path)
    return StoreFactory(dest, (s3.endpoint, s3.region, s3.addressing))


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
    as_json: Annotated[bool, typer.Option("--json", help="Full descriptions as JSON.")] = False,
    descriptions: DescriptionsOpt = None,
) -> None:
    """Find datasets, by their descriptions. Without filters: all of them."""
    infos = _infos(descriptions).filter(signals=signal, tiers=access, repositories=repository)
    if as_json:
        typer.echo(json.dumps([d.model_dump(mode="json") for d in infos], indent=2))
        return
    for d in infos:
        hours = f"{d.stats.hours:,.0f} h" if d.stats.hours is not None else "? h"
        typer.echo(f"{d.id:55} {d.access.tier.value:13} {','.join(d.signals):20} {hours:>12}")
    console.print(plural(len(infos), "dataset"), highlight=False)


@app.command()
def info(
    dataset_id: Annotated[str, typer.Argument(help="Dataset id, e.g. physionet/mitdb.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="The catalog entry and description as JSON.")
    ] = False,
    catalog: CatalogOpt = None,
    descriptions: DescriptionsOpt = None,
) -> None:
    """One dataset: what it is, how to get access, and where the tool downloads it from."""
    entry = _dataset(_catalog(catalog), dataset_id)
    d = _info(_infos(descriptions), dataset_id)
    if as_json:
        both = {
            "catalog": entry.model_dump(mode="json", exclude_defaults=True),
            "description": d.model_dump(mode="json", exclude_defaults=True),
        }
        typer.echo(json.dumps(both, indent=2))
        return
    typer.secho(f"{d.id}  {d.name}", bold=True)
    for label, value in _info_rows(entry, d):
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


def _info_rows(entry: Dataset, d: DatasetInfo) -> list[tuple[str, str]]:
    stats = d.stats
    approx = "~" if stats.hours_estimated else ""
    size_parts = [
        f"{stats.subjects:,} subjects" if stats.subjects is not None else "",
        f"{stats.recordings:,} recordings" if stats.recordings is not None else "",
        f"{approx}{stats.hours:,.0f} h" if stats.hours is not None else "",
        _size(stats.size_bytes) if stats.size_bytes is not None else "",
    ]
    pinned = entry.pinned_version
    version = f" · version {pinned}" if pinned else " · version not pinned"
    return [
        ("signals", ", ".join(d.signals)),
        ("size", " · ".join(p for p in size_parts if p) or "?"),
        ("access", f"{d.access.tier.value} · license {entry.license or '?'}"),
        ("source", entry.source.type + version),
        ("homepage", str(d.homepage) if d.homepage else ""),
    ]


# --- download / import ---------------------------------------------------------------------


@app.command()
def download(
    dataset_ids: IdsArg = None,
    dest: Annotated[str | None, typer.Option("--dest", "-d", help=_DEST_HELP)] = None,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Show the plan for --dest (version, files, size); copy nothing."
        ),
    ] = False,
    replace: ReplaceOpt = False,
    add_missing: AddMissingOpt = False,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
    descriptions: DescriptionsOpt = None,
) -> None:
    """Copy datasets from their sources, unchanged and complete, to --dest.

    Select datasets by ids, --ids-file, --signal/--repository, or --all. Many datasets copy
    at the same time. Datasets that need you to act first (credentials, download by hand)
    are skipped with a warning and listed at the end.

    Each dataset is planned first (version and file list, read once from the source),
    then exactly that plan is copied, its files in parallel. Files go to
    <dest>/raw/<id>/<version>/data/ with a manifest of checksums. If a run stops, run the
    same command again: it continues where it stopped.

    A complete version is skipped, unless --replace: then every file is copied again, in
    place over the old one, and the version is complete again when all have succeeded.
    With --add-missing, a complete version gets only the files the source lists and it
    lacks (e.g. files added to the source later); stored files are not touched.
    """
    setup_logging(verbose)
    if replace and add_missing:
        raise _fail("--replace and --add-missing exclude each other")
    cfg, cfg_path = _config(config)
    settings = _need(cfg.download, "download", cfg_path)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids, ids_file, signal, repository, all_, descriptions
    )
    staging = cfg.staging
    options = FetchOptions(
        staging_dir=staging.dir if staging else None,
        keep_staging=staging.keep if staging else None,
        connections=settings.workers,
        physionet_website_connections=settings.physionet_website_connections,
    )
    console.print(
        f"settings from {cfg_path}: workers {settings.workers}, parallel_datasets "
        f"{settings.parallel_datasets}, physionet_website_connections "
        f"{settings.physionet_website_connections}",
        highlight=False,
        soft_wrap=True,
    )
    if not dest:
        raise _fail("missing --dest (a local folder or s3://bucket[/prefix])")
    store = _store(dest, cfg, cfg_path, settings.workers)
    if dry_run:
        for dataset in datasets:
            _preview(dataset, store, options, replace, add_missing)
        return
    _run(datasets, store, options, settings, replace, add_missing)


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
    _run([dataset], store, options, settings, replace=False, add_missing=False)


def _run(
    datasets: list[Dataset],
    store: Store,
    options: FetchOptions,
    settings: DownloadConfig,
    replace: bool,
    add_missing: bool,
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
        outcomes = _copy(datasets, store, options, settings, replace, add_missing)
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
    replace: bool,
    add_missing: bool,
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
            replace=replace,
            add_missing=add_missing,
            workers=settings.workers,
            parallel_datasets=settings.parallel_datasets,
            progress_for=progress_for,
            on_done=on_done,
        )


_OUTCOME_STYLE = {
    "complete": ("✓", "green"),
    "already": ("•", "yellow"),
    "incomplete": ("✗", "red"),
    "credentials": ("!", "yellow"),
    "restricted": ("!", "yellow"),
    "no-recordings": ("✗", "red"),
    "manual": ("!", "yellow"),
    "unavailable": ("!", "yellow"),
    "failed": ("✗", "red"),
}


def _print_outcome(outcome: Outcome) -> None:
    mark, color = _OUTCOME_STYLE[outcome.status]
    d, r = outcome.dataset, outcome.report
    if r is not None:
        text = (
            f"{d.id} {r.version}: {outcome.status} · {plural(r.files, 'file')} · "
            f"{plural(r.copied, 'file')} ({_size(r.bytes_copied)}) copied"
        )
        if r.skipped:
            text += f" · {r.skipped:,} already there"
        if r.failed:
            text += f" · {plural(len(r.failed), 'file')} failed (run again to retry)"
        if r.differs:
            text += (
                f" · [yellow]{plural(len(r.differs), 'file')} served other than the source "
                "lists them, kept as served (in the manifest)[/]"
            )
        if r.missing:
            text += (
                f" · [yellow]{plural(len(r.missing), 'file')} listed by the source but "
                "missing there (in the manifest)[/]"
            )
        if outcome.message:
            text += f" · {outcome.message}"
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
    failed = sum(len(by_status.get(s, [])) for s in ("failed", "incomplete", "no-recordings"))
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
        ("Copied, but no recordings: no file can hold signals (metadata only)", "no-recordings"),
        ("Failed", "failed"),
        ("Incomplete, some files failed (run again to retry)", "incomplete"),
    ]
    for title, status in groups:
        items = by_status.get(status, [])
        if not items:
            continue
        color = "red" if status in ("failed", "incomplete", "no-recordings") else "yellow"
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
    dataset: Dataset, store: Store, options: FetchOptions, replace: bool, add_missing: bool
) -> None:
    """The plan of one dataset against --dest, without copying: version, files, size."""
    try:
        with planned(
            dataset, store, options=options, root=ROOT, replace=replace, add_missing=add_missing
        ) as plan:
            _print_plan(plan)
    except AlreadyComplete as exc:
        console.print(f"[yellow]•[/] {dataset.id}: {escape(str(exc))}", highlight=False)
    except FetchError as exc:  # needs action, or the source refused: say so, go on
        console.print(
            f"[yellow]![/] {dataset.id}: {escape(str(exc))}", highlight=False, soft_wrap=True
        )
    except (OSError, RuntimeError, ValueError) as exc:  # as in a real run: report, go on
        console.print(
            f"[red]✗[/] {dataset.id}: {type(exc).__name__}: {escape(str(exc))}",
            highlight=False,
            soft_wrap=True,
        )


def _print_plan(plan: Plan) -> None:
    unknown = f" ({plan.unknown_sizes} sizes unknown)" if plan.unknown_sizes else ""
    again = " (complete; --replace makes it again)" if plan.replace else ""
    typer.secho(f"{plan.dataset.id} {plan.version.value}{again}", bold=True)
    typer.echo(
        f"  {plural(len(plan.files), 'file')}, {_size(plan.total_bytes)}{unknown}; "
        f"{plural(plan.recording_files, 'file')} can hold recordings"
    )


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
    descriptions: DescriptionsOpt = None,
) -> None:
    """What is stored in --dest: versions, complete or not, files and size."""
    cfg, cfg_path = _config(config)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids, ids_file, signal, repository, all_, descriptions
    )
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
            state = "complete" if complete else "[yellow]incomplete (interrupted)[/]"
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
    descriptions: DescriptionsOpt = None,
) -> None:
    """Check that stored files are all there with the size (and SHA-256) of the manifest.
    An incomplete (interrupted) version is checked against the files copied so far."""
    if (version is None) == (not latest):
        raise _fail("give either --version <version> or --latest")
    cfg, cfg_path = _config(config)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids, ids_file, signal, repository, all_, descriptions
    )
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


@app.command()
def standardize(
    dataset_ids: IdsArg = None,
    dest: Annotated[str | None, typer.Option("--dest", "-d", help=_DEST_HELP)] = None,
    version: Annotated[
        str | None,
        typer.Option("--version", help="This raw version (one dataset only)."),
    ] = None,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the recordings found; convert nothing."),
    ] = False,
    replace: ReplaceOpt = False,
    from_zip: Annotated[
        bool,
        typer.Option(
            "--from-zip",
            help="PhysioNet only: stage the version's ZIP from PhysioNet's AWS copy on local "
            "disk (checked against its ETag and SHA256SUMS.txt) instead of reading <dest>/raw/.",
        ),
    ] = False,
    from_mirror: Annotated[
        bool,
        typer.Option(
            "--from-mirror",
            help="PhysioNet only: read the version straight from PhysioNet's AWS copy, each "
            "recording's files checked against SHA256SUMS.txt as they are converted (no raw "
            "version, no ZIP; icentia11k).",
        ),
    ] = False,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
    descriptions: DescriptionsOpt = None,
) -> None:
    """Convert raw versions to the standard layer: samples at their native sampling rates
    with the exact stored values, as compressed blocks with one block index; metadata as
    Parquet tables.

    The raw version is read from <dest>/raw/ (or, with --from-zip, staged from the source's
    ZIP), the result written to <dest>/standard-blocks/<id>/<raw version>/. Without
    --version, the newest complete raw version is used (with --from-zip: the catalog's
    pinned version). Every raw file is either used or listed in the manifest as not converted, with
    the reason. If a run stops, run the same command again: it continues where it stopped.
    A complete version is skipped, unless --replace: then it is removed and made again from
    the raw version. See docs/standard-layer.md.
    """
    setup_logging(verbose)
    cfg, cfg_path = _config(config)
    settings = _need(cfg.standardize, "standardize", cfg_path)
    staging = _need(cfg.staging, "staging", cfg_path)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids, ids_file, signal, repository, all_, descriptions
    )
    if version is not None and len(datasets) != 1:
        raise _fail("--version applies to one dataset")
    if from_zip and from_mirror:
        raise _fail("--from-zip and --from-mirror exclude each other")
    if not dest:
        raise _fail("missing --dest (a local folder or s3://bucket[/prefix])")
    store = _store(dest, cfg, cfg_path, settings.workers)
    console.print(
        f"settings from {cfg_path}: workers {settings.workers}; raw files are fetched to "
        f"{staging.dir.expanduser() / 'standardize'}",
        highlight=False,
        soft_wrap=True,
    )
    failed = 0
    for d in datasets:
        try:
            if from_zip:
                website = _need(cfg.download, "download", cfg_path).physionet_website_connections
                ok = _standardize_from_zip(
                    d, version, store, settings, staging.dir, replace, website
                )
                failed += not ok
                continue
            if from_mirror:
                ok = _standardize_from_mirror(d, version, store, settings, staging.dir, replace)
                failed += not ok
                continue
            ok = _standardize_one(d.id, version, store, settings, staging.dir, dry_run, replace)
        except KeyboardInterrupt:
            console.print(
                "[yellow]stopped.[/] Recordings finished so far are kept; run the same "
                "command again to continue.",
                highlight=False,
            )
            raise typer.Exit(130) from None
        failed += not ok
    raise typer.Exit(1 if failed else 0)


def _standardize_from_zip(
    dataset: Dataset,
    version: str | None,
    store: Store,
    settings: StandardizeConfig,
    staging_dir: Path,
    replace: bool,
    website_connections: int,
) -> bool:
    """Pipeline physionet-zip: stage the version's ZIP locally, convert, remove the stage.
    `website_connections`: requests to physionet.org at the same time, for a version that
    is not on PhysioNet's AWS copy."""
    from biosignal_data.fetchers import physionet_zip  # noqa: PLC0415

    version = version or dataset.pinned_version
    if version is None:
        console.print(f"[red]✗[/] {dataset.id}: --from-zip needs --version or a pinned version")
        return False
    work = staging_dir.expanduser() / "standardize"
    try:
        staged = physionet_zip.stage(
            dataset, version, work, connections=settings.workers,
            website_connections=website_connections,
        )  # fmt: skip
    except (physionet_zip.ZipCheckError, TypeError) as exc:
        console.print(f"[red]✗[/] {dataset.id} {version}: {exc}", highlight=False, soft_wrap=True)
        return False
    try:
        checked = sum(1 for f in staged.manifest.files if f.source_checksum)
        by_size = len(staged.manifest.files) - checked
        console.print(
            f"[dim]{dataset.id} {version}: staged {len(staged.manifest.files):,} files: "
            f"{checked:,} checked against SHA256SUMS.txt, {by_size:,} (not listed there) by "
            "size",
            highlight=False,
            soft_wrap=True,
        )
        return _standardize_one(
            dataset.id, version, store, settings, staging_dir, False, replace,
            raw_store=staged.store,
        )  # fmt: skip
    finally:
        shutil.rmtree(staged.folder, ignore_errors=True)


def _standardize_from_mirror(
    dataset: Dataset, version: str | None, store: Store, settings: StandardizeConfig,
    staging_dir: Path, replace: bool,
) -> bool:  # fmt: skip
    """Pipeline physionet-mirror: the version read from PhysioNet's AWS copy as it is
    converted (fetchers/physionet_mirror.py)."""
    from biosignal_data.fetchers import physionet_mirror  # noqa: PLC0415

    version = version or dataset.pinned_version
    if version is None:
        console.print(f"[red]✗[/] {dataset.id}: --from-mirror needs --version or a pinned version")
        return False
    try:
        raw = physionet_mirror.mirror(
            dataset, version, connections=settings.workers, keys=_physionet_aws_keys()
        )
    except (physionet_mirror.MirrorError, MissingCredentialsError, TypeError) as exc:
        console.print(f"[red]✗[/] {dataset.id} {version}: {exc}", highlight=False, soft_wrap=True)
        return False
    return _standardize_one(
        dataset.id, version, store, settings, staging_dir, False, replace, raw_store=raw
    )


PHYSIONET_AWS_KEYS = (
    "BIOSIGNAL_PHYSIONET_AWS_ACCESS_KEY_ID",
    "BIOSIGNAL_PHYSIONET_AWS_SECRET_ACCESS_KEY",
)


def _physionet_aws_keys() -> tuple[str, str] | None:
    """The keys of an AWS account linked to PhysioNet, for versions behind an access point
    (docs/access.md); None when neither variable is set."""
    if not any(os.environ.get(n) for n in PHYSIONET_AWS_KEYS):
        return None
    key_id, secret = require_env(*PHYSIONET_AWS_KEYS)
    return key_id, secret


def _standardize_one(
    dataset_id: str,
    version: str | None,
    store: Store,
    settings: StandardizeConfig,
    staging_dir: Path,
    dry_run: bool,
    replace: bool,
    raw_store: Store | None = None,
) -> bool:
    """Plan (and unless `dry_run`, convert) one dataset; print one result line."""
    if version is None:
        versions = complete_versions(store, dataset_id, ROOT)
        if not versions:
            console.print(
                f"[yellow]![/] {dataset_id}: no complete raw version",
                highlight=False,
                soft_wrap=True,
            )
            return True
        version = versions[-1]
        console.print(
            f"[dim]{dataset_id}: newest complete raw version {version} (no --version given)",
            highlight=False,
            soft_wrap=True,
        )
    try:
        plan = plan_standard(
            store, dataset_id, version, raw_root=ROOT, replace=replace, raw_store=raw_store
        )
    except AlreadyStandard:
        console.print(
            f"[green]✓[/] {dataset_id} {version}: already complete (--replace makes it again)",
            highlight=False,
            soft_wrap=True,
        )
        return True
    except FileNotFoundError as exc:
        console.print(f"[red]✗[/] {dataset_id}: {exc}", highlight=False, soft_wrap=True)
        return False
    if dry_run:
        _print_standard_plan(plan)
        return True
    with StandardizeBar(f"{dataset_id} [dim]{version}") as bar:
        report = convert(
            plan,
            store,
            workers=settings.workers,
            part_bytes=settings.part_bytes,
            work_dir=staging_dir.expanduser() / "standardize",
            processes=settings.processes,
            progress=bar,
        )
    if not report.complete:
        console.print(
            f"[red]✗[/] {dataset_id} {version}: {plural(len(report.failed), 'recording')} "
            f"failed of {report.units:,}; run again to retry them",
            highlight=False,
            soft_wrap=True,
        )
        for unit_id, error in sorted(report.failed.items()):
            console.print(f"    {unit_id}: {error}", highlight=False, soft_wrap=True)
        return False
    location = StandardLocation(dataset_id, version)
    manifest = StandardManifest.model_validate_json(store.get_bytes(location.manifest_key) or b"")
    console.print(
        f"[green]✓[/] {dataset_id} {version}: {plural(manifest.recordings, 'recording')}, "
        f"{plural(manifest.channels, 'channel')}, {plural(manifest.events, 'event')} -> "
        f"{report.location}",
        highlight=False,
        soft_wrap=True,
    )
    for reason, count in iter_not_converted(manifest):
        console.print(f"    [yellow]not converted[/]: {plural(count, 'file')}: {reason}",
                      highlight=False, soft_wrap=True)  # fmt: skip
    for problem in manifest.problems:
        console.print(f"    [yellow]problem[/]: {problem}", highlight=False, soft_wrap=True)
    return True


def _print_standard_plan(plan: StandardPlan) -> None:
    formats: dict[str, int] = {}
    for unit in plan.units:
        formats[unit.format] = formats.get(unit.format, 0) + 1
    annotations = sum(len(u.annotations) for u in plan.units)
    views = len(plan.discovery.views)
    found = ", ".join(f"{n:,} {fmt.upper()}" for fmt, n in sorted(formats.items()))
    again = " (complete; --replace makes it again)" if plan.replace else ""
    console.print(
        f"{plan.raw.location.dataset_id} {plan.raw.location.version}{again}: "
        f"{plural(len(plan.units), 'recording')} ({found or 'none'}), "
        f"{plural(annotations, 'annotation file')}; {len(plan.to_convert):,} to convert, "
        f"{len(plan.done):,} done by an earlier run"
        + (f"; {plural(views, 'view')} of other records, not stored twice" if views else ""),
        highlight=False,
        soft_wrap=True,
    )


@app.command("quality")
def quality(
    dest: DestOpt,
    dataset_ids: IdsArg = None,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    replace: ReplaceOpt = False,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
    descriptions: DescriptionsOpt = None,
) -> None:
    """Measure every channel of the newest complete standard version of each dataset:
    invalid and non-finite samples, flat runs, samples at the recorder's limits, and the
    physical values. Written to <dest>/quality/<id>/<version>/; nothing is deleted or
    changed. The settings come from the `quality` section of biosignal-data.yaml."""
    from biosignal_data.quality import AlreadyMeasured, measure  # noqa: PLC0415

    setup_logging(verbose)
    cfg, cfg_path = _config(config)
    settings = _need(cfg.quality, "quality", cfg_path)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids, ids_file, signal, repository, all_, descriptions
    )
    store = _store(dest, cfg, cfg_path, settings.workers)
    failed = 0
    for d in datasets:
        try:
            m = measure(
                store, _factory(dest, cfg, cfg_path), d.id, flat_s=settings.flat_s,
                percentile_sample=settings.percentile_sample,
                seed=settings.seed, workers=settings.workers, replace=replace,
            )  # fmt: skip
        except AlreadyMeasured as exc:
            console.print(f"[green]✓[/] {d.id}: {escape(str(exc))}", highlight=False)
            continue
        except FileNotFoundError as exc:
            console.print(f"[yellow]![/] {d.id}: {escape(str(exc))}", highlight=False)
            continue
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            failed += 1
            console.print(f"[red]✗[/] {d.id}: {type(exc).__name__}: {escape(str(exc))}")
            continue
        console.print(
            f"[green]✓[/] {d.id} {m.standard_version}: {plural(m.recordings, 'recording')}, "
            f"{plural(m.channels, 'channel')} measured",
            highlight=False,
        )
    raise typer.Exit(1 if failed else 0)


@app.command("index")
def index_(
    dest: DestOpt,
    dataset_ids: IdsArg = None,
    ids_file: IdsFileOpt = None,
    signal: SignalOpt = [],  # noqa: B006
    repository: RepositoryOpt = [],  # noqa: B006
    all_: AllOpt = False,
    version: Annotated[
        str | None,
        typer.Option("--version", help="The label of the index. Default: today's UTC date."),
    ] = None,
    replace: ReplaceOpt = False,
    rules: Annotated[
        Path | None,
        typer.Option(
            "--rules", help=f"Subject and channel rules. Default: {default_rules_path()}."
        ),
    ] = None,
    verbose: VerboseOpt = False,
    config: ConfigOpt = None,
    catalog: CatalogOpt = None,
    descriptions: DescriptionsOpt = None,
) -> None:
    """Build the training index from the complete standard versions in --dest: every
    recording with its subject and split (by subject, stable), every channel with its
    signal type. Written to <dest>/index/<version>/; the signals are not copied.
    See docs/training-layer.md.
    """
    setup_logging(verbose)
    cfg, cfg_path = _config(config)
    datasets = _select_datasets(
        _catalog(catalog), dataset_ids, ids_file, signal, repository, all_, descriptions
    )
    store = _store(dest, cfg, cfg_path)
    try:
        report = build_index(
            store,
            [d.id for d in datasets],
            IndexRules.load(rules),
            version=version,
            replace=replace,
        )
    except (AlreadyIndexed, RulesError, SubjectError) as exc:
        raise _fail(str(exc)) from exc
    m = report.manifest
    console.print(
        f"[green]✓[/] index {m.version}: {plural(len(m.datasets), 'dataset')}, "
        f"{plural(m.recordings, 'recording')}, {plural(m.subjects, 'subject')} "
        f"({m.subjects_assumed:,} assumed one per recording) -> {report.location}",
        highlight=False,
        soft_wrap=True,
    )
    console.print(f"    splits: {m.splits}", highlight=False, soft_wrap=True)
    console.print(f"    signal types: {m.signal_types}", highlight=False, soft_wrap=True)
    for dataset_id, why in m.left_out.items():
        console.print(f"    [yellow]left out[/]: {dataset_id}: {why}", highlight=False)
    for name, count in m.unknown_channels.items():
        console.print(
            f"    [yellow]unknown signal type[/]: {name} ({plural(count, 'channel')})",
            highlight=False,
            soft_wrap=True,
        )


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
def validate(catalog: CatalogOpt = None, descriptions: DescriptionsOpt = None) -> None:
    """Validate catalog.yaml and docs/datasets.yaml, and that they list the same datasets."""
    cat, infos = _catalog(catalog), _infos(descriptions)
    try:
        check_pair(cat, infos)
    except CatalogError as exc:
        raise _fail(str(exc), 2) from exc
    typer.echo(f"{len(cat)} datasets: catalog and descriptions are valid and match")


@dev.command()
def docs(
    out: Annotated[Path, typer.Option(help="Output folder.")] = Path("docs/datasets"),
    check: Annotated[bool, typer.Option(help="Fail if the generated docs are stale.")] = False,
    catalog: CatalogOpt = None,
    descriptions: DescriptionsOpt = None,
) -> None:
    """Generate the dataset pages in docs/datasets/ from the two dataset files."""
    from biosignal_data.docsgen import write_docs  # noqa: PLC0415

    cat, infos = _catalog(catalog), _infos(descriptions)
    try:
        check_pair(cat, infos)
    except CatalogError as exc:
        raise _fail(str(exc), 2) from exc
    stale = write_docs(cat, infos, out, check=check)
    if check and stale:
        raise _fail("stale docs: " + ", ".join(stale) + "; run `biosignal-data dev docs`")
    typer.echo(f"docs {'checked' if check else 'written'} in {out}")


def main() -> None:  # pragma: no cover
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(app())

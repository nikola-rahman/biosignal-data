"""Terminal output for the CLI: logging and the download progress display (rich).

⠴ physionet/mitdb 1.0.0 ━━━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━  3/5 recordings  2.1 MB  410 kB/s  0:00:04
  ↓ 103.dat (+2 more)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable

from rich.console import Console, RenderableType
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Column
from rich.text import Text

console = Console(stderr=True)


def setup_logging(verbose: bool) -> None:
    """Our logs at INFO (DEBUG with -v); third-party libraries only at WARNING."""
    handler = RichHandler(console=console, show_path=False, markup=False, log_time_format="%X")
    logging.basicConfig(level=logging.WARNING, format="%(message)s", handlers=[handler], force=True)
    logging.getLogger("biosignal_data").setLevel(logging.DEBUG if verbose else logging.INFO)


def _size(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} TB"


class _RecordingsColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        if task.total is None:
            return Text("")
        return Text(f"{task.completed:,.0f}/{task.total:,.0f} recordings")


class _BytesColumn(ProgressColumn):
    """Bytes downloaded (of the total, if known) and the average speed."""

    def render(self, task: Task) -> Text:
        done = task.fields.get("bytes", 0)
        total = task.fields.get("total_bytes")
        amount = _size(done) + (f" / {_size(total)}" if total else "")
        elapsed = task.elapsed or 0
        speed = f"  {_size(done / elapsed)}/s" if elapsed > 1 and done else ""
        return Text(amount + speed, style="progress.download")


class _DownloadProgress(Progress):
    """A `Progress` that prints the files being downloaded under each active bar."""

    def __init__(self) -> None:
        self.status: dict[TaskID, str] = {}  # set first: rendering can start in __init__
        super().__init__(
            SpinnerColumn(finished_text="[green]✓"),
            TextColumn("[bold]{task.description}", table_column=Column(no_wrap=True)),
            BarColumn(bar_width=None),  # fills the width that the other columns leave
            _RecordingsColumn(table_column=Column(no_wrap=True)),
            _BytesColumn(table_column=Column(no_wrap=True)),
            TimeRemainingColumn(table_column=Column(no_wrap=True)),
            console=console,
            expand=True,
            disable=not console.is_terminal,
        )

    def get_renderables(self) -> Iterable[RenderableType]:
        yield self.make_tasks_table(self.tasks)
        for line in list(self.status.values()):
            if line:
                yield Text(f"  ↓ {line}", style="dim", no_wrap=True, overflow="ellipsis")


def download_progress() -> _DownloadProgress:
    return _DownloadProgress()


class ProgressBar:
    """Implements `ingest.IngestProgress` with one task of the download display."""

    def __init__(self, progress: _DownloadProgress, dataset_id: str) -> None:
        self._progress = progress
        self._dataset_id = dataset_id
        self._lock = threading.Lock()
        self._active: list[str] = []  # files downloading now, oldest first
        self._bytes = 0
        self._failed = 0
        self._task = progress.add_task(f"{dataset_id} [dim]listing files...", total=None)
        progress.status[self._task] = ""

    def start(self, version: str, recordings: int, done: int, total_bytes: int | None) -> None:
        self._progress.update(
            self._task,
            description=f"{self._dataset_id} [dim]{version}",
            total=recordings,
            completed=done,
            bytes=0,
            total_bytes=total_bytes,
        )

    def file_started(self, path: str) -> None:
        with self._lock:
            self._active.append(path)
            self._show()

    def advance(self, n: int) -> None:
        with self._lock:
            self._bytes += n
            self._progress.update(self._task, bytes=self._bytes)

    def file_done(self, path: str, ok: bool) -> None:
        with self._lock:
            if path in self._active:
                self._active.remove(path)
            if not ok:
                self._failed += 1
            self._show()

    def recording_done(self) -> None:
        self._progress.advance(self._task, 1)

    def _show(self) -> None:
        failed = f"  [{self._failed} failed]" if self._failed else ""
        if not self._active:
            self._progress.status[self._task] = failed.strip()
            return
        more = f" (+{len(self._active) - 1} more)" if len(self._active) > 1 else ""
        self._progress.status[self._task] = f"{self._active[-1]}{more}{failed}"

    def finish(self, state: str) -> None:
        """Keep the bar for `complete` and `subset` runs; drop it otherwise (the CLI prints
        the error)."""
        self._progress.status.pop(self._task, None)
        if state not in ("complete", "subset"):
            self._progress.remove_task(self._task)
            return
        task = next(t for t in self._progress.tasks if t.id == self._task)
        style = "green" if state == "complete" else "yellow"
        self._progress.update(
            self._task,
            description=f"{task.description} [{style}]{state}[/]",
            total=task.total or task.completed,
            completed=task.total or task.completed,
        )

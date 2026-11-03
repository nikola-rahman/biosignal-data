from __future__ import annotations

import logging
import threading

import pytest

from biosignal_data import progress


def test_a_long_phase_logs_its_progress_with_rate_and_eta(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    monkeypatch.setattr("biosignal_data.progress.time.monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="biosignal_data.progress")
    ticker = progress.Ticker("x: reading headers", 1000)

    def count() -> None:
        for _ in range(100):
            ticker.add()

    threads = [threading.Thread(target=count) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not caplog.messages  # not yet EVERY seconds
    clock[0] = 10.0
    ticker.add(nbytes=2 * 10**9)
    assert caplog.messages == ["x: reading headers: 201 of 1,000 files, 2.0 GB (20/s, 1 min left)"]
    clock[0] = 12.0
    ticker.add()  # within EVERY of the last line: quiet
    ticker.finish()
    assert caplog.messages[-1] == (
        "x: reading headers: 202 of 1,000 files, 2.0 GB (17/s, 1 min left): done"
    )


def test_a_short_phase_or_an_unknown_total_stays_short(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = [0.0]
    monkeypatch.setattr("biosignal_data.progress.time.monotonic", lambda: clock[0])
    caplog.set_level(logging.INFO, logger="biosignal_data.progress")
    progress.Ticker("quick", 3).finish()
    assert not caplog.messages  # under EVERY seconds: no line at all
    ticker = progress.Ticker("listing", None)
    clock[0] = 6.0
    ticker.add(4000)
    assert caplog.messages == ["listing: 4,000 files (667/s)"]

from __future__ import annotations

import pytest
import responses

from biosignal_data.batch import download_many
from biosignal_data.storage.stores import LocalStore
from tests.conftest import make_dataset


def _http(dataset_id: str, name: str) -> dict[str, object]:
    return {
        "id": dataset_id,
        "source": {
            "type": "http",
            "version": "1",
            "files": [{"url": f"https://example.org/{dataset_id}/{name}"}],
        },
    }


@responses.activate
def test_many_datasets_share_one_pool_and_report_each(
    store: LocalStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NSRR_TOKEN", raising=False)
    for i in range(3):
        responses.get(f"https://example.org/test/ok{i}/rec{i}.edf", body=b"x" * (i + 1))
    responses.get("https://example.org/test/broken/rec.edf", status=404)
    datasets = [
        *(make_dataset(**_http(f"test/ok{i}", f"rec{i}.edf")) for i in range(3)),
        make_dataset(**_http("test/broken", "rec.edf")),
        make_dataset(id="test/byhand"),
        make_dataset(id="test/gone", access={"tier": "unavailable"}),
        make_dataset(
            id="test/sleep",
            source={"type": "nsrr", "slug": "x"},
            access={"tier": "application"},
        ),
    ]
    finished: list[str] = []
    outcomes = download_many(
        datasets,
        store,
        workers=4,
        parallel_datasets=3,
        on_done=lambda o: finished.append(o.dataset.id),
    )
    status = {o.dataset.id: o.status for o in outcomes}
    assert status == {
        "test/ok0": "complete",
        "test/ok1": "complete",
        "test/ok2": "complete",
        "test/broken": "incomplete",
        "test/byhand": "manual",
        "test/gone": "unavailable",
        "test/sleep": "credentials",
    }
    assert sorted(finished) == sorted(status)
    assert store.get_bytes("raw/test/ok2/1/data/rec2.edf") == b"xxx"

    again = {
        o.dataset.id: o.status
        for o in download_many(datasets[:3], store, workers=4, parallel_datasets=2)
    }
    assert set(again.values()) == {"already"}

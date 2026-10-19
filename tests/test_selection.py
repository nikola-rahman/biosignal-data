from __future__ import annotations

from biosignal_data.selection import select

BIDS = [
    ".datalad/config",
    "README",
    "dataset_description.json",
    "participants.tsv",
    "task-rest_eeg.json",
    "sub-01/sub-01_scans.tsv",
    "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
    "sub-01/eeg/sub-01_task-rest_eeg.vmrk",
    "sub-01/eeg/sub-01_task-rest_eeg.eeg",
    "sub-01/eeg/sub-01_task-rest_channels.tsv",
    "sub-01/eeg/sub-01_task-rest_run-1_eeg.edf",
    "sub-01/eeg/sub-01_task-rest_run-10_eeg.edf",
    "sub-02/sub-02_scans.tsv",
    "sub-02/eeg/sub-02_task-rest_eeg.edf",
    "sub-02/eeg/sub-02_task-rest_eeg.json",
    "derivatives/report.html",
]


def test_no_options_selects_everything() -> None:
    selection = select(BIDS)
    assert selection.paths == sorted(BIDS) and len(selection.recordings) == 4


def test_limit_counts_bids_recordings_with_metadata() -> None:
    selection = select(BIDS, limit=1)
    assert len(selection.recordings) == 1
    assert selection.paths == [
        "README",
        "dataset_description.json",
        "participants.tsv",
        "sub-01/eeg/sub-01_task-rest_channels.tsv",
        "sub-01/eeg/sub-01_task-rest_eeg.eeg",
        "sub-01/eeg/sub-01_task-rest_eeg.vhdr",
        "sub-01/eeg/sub-01_task-rest_eeg.vmrk",
        "sub-01/sub-01_scans.tsv",
        "task-rest_eeg.json",
    ]


def test_runs_are_separate_recordings() -> None:
    paths = select(BIDS, limit=3).paths
    assert "sub-01/eeg/sub-01_task-rest_run-10_eeg.edf" in paths
    assert "sub-01/eeg/sub-01_task-rest_run-1_eeg.edf" in paths
    assert not any(p.startswith("sub-02/") for p in paths)


def test_include_then_limit() -> None:
    paths = select(BIDS, include=["sub-02/**", "**/*.json", "**/*.tsv"], limit=1).paths
    assert "sub-02/eeg/sub-02_task-rest_eeg.edf" in paths
    assert "sub-02/eeg/sub-02_task-rest_eeg.json" in paths
    assert not any(p.startswith("sub-01/") for p in paths)


def test_wfdb_record_files_are_grouped() -> None:
    files = ["RECORDS", "SHA256SUMS.txt", "100.atr", "100.dat", "100.hea", "101.dat", "101.hea"]
    selection = select(files, limit=1)
    assert selection.paths == ["100.atr", "100.dat", "100.hea", "RECORDS", "SHA256SUMS.txt"]


def test_folder_format_is_one_recording() -> None:
    files = [
        "sub-1/meg/sub-1_task-x_run-1_meg.ds/a.meg4",
        "sub-1/meg/sub-1_task-x_run-1_meg.ds/a.res4",
        "sub-1/meg/sub-1_task-x_run-1_meg.json",
        "sub-1/meg/sub-1_task-x_run-2_meg.ds/a.meg4",
        "sub-1/meg/sub-1_task-x_run-2_meg.json",
    ]
    assert select(files, limit=1).paths == files[:3]


def test_unknown_formats_count_each_data_file() -> None:
    files = ["README.md", "part1.zip", "part2.zip", "docs/info.pdf"]
    selection = select(files, limit=1)
    assert selection.paths == ["README.md", "part1.zip"]
    assert selection.recordings == [["part1.zip"]]


def test_compressed_signal_files() -> None:
    files = ["a.edf.gz", "a.txt", "b.edf.gz"]
    assert select(files, limit=1).paths == ["a.edf.gz", "a.txt"]


def test_hidden_files_are_never_recordings() -> None:
    files = [".datalad/config", ".gitattributes", "README", "sub-1/eeg/sub-1_task-a_eeg.edf"]
    assert select(files, limit=1).recordings == [["sub-1/eeg/sub-1_task-a_eeg.edf"]]


def test_raw_recordings_come_before_derivatives() -> None:
    files = [
        "derivatives/db.mat",
        "sub-1/eeg/sub-1_task-a_eeg.set",
        "sub-1/eeg/sub-1_task-a_eeg.fdt",
    ]
    first = select(files, limit=1).recordings[0]
    assert first == ["sub-1/eeg/sub-1_task-a_eeg.fdt", "sub-1/eeg/sub-1_task-a_eeg.set"]


def test_compressed_tables_are_data() -> None:
    files = ["sub-1/sub-1_task-a_physio.tsv.gz", "sub-1/sub-1_task-a_physio.json", "README"]
    selection = select(files, limit=1)
    assert selection.recordings == [
        ["sub-1/sub-1_task-a_physio.json", "sub-1/sub-1_task-a_physio.tsv.gz"]
    ]


def test_metadata_only_has_no_recordings() -> None:
    files = [".datalad/config", "README", "sub-1/eeg/sub-1_task-a_channels.tsv"]
    assert select(files, limit=1).recordings == []

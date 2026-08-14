import json

import polars as pl

from models.dataset_models import DatasetStatus, dataset_directory
from models.detection_models import (
    ApprovedConfig,
    ColumnSpec,
    FileFormat,
    FixedWidthLayout,
)
from models.file_models import FileStatus
from services.mft.transport import LocalTransport
from services.orchestrator.orchestrator import (
    Pipeline,
    PipelineSettings,
    Workspace,
)


class FailingTransport(LocalTransport):
    """LocalTransport whose downloads always fail (network simulation)."""

    def get(self, remote_path: str, local_path: str) -> None:
        raise OSError("simulated network failure")


def make_pipeline(tmp_path, remote_files, *, layout=None, download_workers=2):
    mft = tmp_path / "mft"
    mft.mkdir(exist_ok=True)
    for name, content in remote_files.items():
        (mft / name).write_text(content, encoding="utf-8")

    workspace = Workspace.at(tmp_path / "workspace")
    settings = PipelineSettings(
        source_path="/",
        transport_factory=lambda: LocalTransport(mft),
        fixed_width_layout=layout,
        download_workers=download_workers,
        parser_workers=1,
    )
    return Pipeline(workspace, settings)


def status_of(pipeline, name):
    match = [f for f in pipeline.files.values() if f.name == name]
    assert match, f"No file named {name}"
    return match[0].status


def complete_datasets(pipeline):
    return {
        d.source_file_name: d
        for d in pipeline.datasets.values()
        if d.status is DatasetStatus.COMPLETE
    }


# ---------------------------------------------------------------------
# End-to-end happy path.
# ---------------------------------------------------------------------
def test_flat_csv_goes_to_raw_deleted(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        {"sales.csv": "STORE,UNITS,PRICE\n1001,45,500\n1002,60,700\n"},
    )
    pipeline.discover()
    summary = pipeline.process()

    assert status_of(pipeline, "sales.csv") is FileStatus.RAW_DELETED
    assert summary.parsed == 1
    assert summary.failed == 0

    datasets = complete_datasets(pipeline)
    assert "sales.csv" in datasets
    dataset = datasets["sales.csv"]
    assert dataset.row_count == 2
    assert dataset.schema["STORE"] == "String"

    dataset_dir = dataset_directory(tmp_path / "workspace", dataset.dataset_id)
    assert (dataset_dir / "sales.csv.parquet").exists()
    assert (dataset_dir / "metadata.json").exists()
    # Raw staging file is cleaned up.
    assert not (tmp_path / "workspace/raw/sales.csv").exists()


def test_multiple_files_process_in_batches(tmp_path):
    files = {
        f"sales_{i}.csv": "STORE,UNITS,PRICE\n1001,45,500\n1002,60,700\n"
        for i in range(6)
    }
    pipeline = make_pipeline(tmp_path, files)
    pipeline.discover()
    summary = pipeline.process()

    assert summary.parsed == 6
    assert len(complete_datasets(pipeline)) == 6
    for name in files:
        assert status_of(pipeline, name) is FileStatus.RAW_DELETED


def test_multiline_delimited_pipeline(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        {"multi.csv": 'STORE,UNITS,PRICE\n1001,"45,000",500\n1002,60,700\n'},
    )
    pipeline.discover()
    pipeline.process()
    dataset = complete_datasets(pipeline)["multi.csv"]
    assert dataset.row_count == 2
    # The quoted delimiter is informational; no blocking issues.
    assert dataset.extra["parse_report"]["counters"] == {"quoted_delimiter": 1}


# ---------------------------------------------------------------------
# Approval gating.
# ---------------------------------------------------------------------
def test_low_confidence_file_waits_for_approval(tmp_path):
    # No delimiter, no layout -> detection is uncertain -> approval gate.
    pipeline = make_pipeline(
        tmp_path,
        {"junk.txt": "a short line\na longer second line here\nfour\n"},
    )
    pipeline.discover()
    summary = pipeline.process()

    assert status_of(pipeline, "junk.txt") is FileStatus.AWAITING_APPROVAL
    assert summary.pending_approval == 1
    assert summary.parsed == 0
    assert not complete_datasets(pipeline)


def test_approve_resumes_and_completes(tmp_path):
    # Detection flags this multiline file for review (low confidence);
    # approving with a delimited config resumes parsing.
    pipeline = make_pipeline(
        tmp_path,
        {"data.csv": 'STORE,UNITS,PRICE\n123,"multi\nline",abc\n456,def,ghi\n'},
    )
    pipeline.discover()
    pipeline.process()
    file_id = next(
        fid
        for fid, f in pipeline.files.items()
        if f.name == "data.csv" and f.status is FileStatus.AWAITING_APPROVAL
    )

    source = pipeline.get_file(file_id)
    result = pipeline._detector.detect(source.local_path)
    assert result.needs_approval()  # detection was uncertain
    config = ApprovedConfig(
        source_format=FileFormat.DELIMITED,
        delimiter=",",
        encoding=result.encoding,
        header_present=True,
        columns=["STORE", "UNITS", "PRICE"],
    )

    pipeline.approve(file_id, config)
    assert status_of(pipeline, "data.csv") is FileStatus.RAW_DELETED
    assert complete_datasets(pipeline)["data.csv"].row_count == 2


def test_approve_rejects_invalid_config(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        {"data.csv": 'STORE,UNITS,PRICE\n123,"multi\nline",abc\n456,def,ghi\n'},
    )
    pipeline.discover()
    pipeline.process()
    file_id = next(
        fid
        for fid, f in pipeline.files.items()
        if f.status is FileStatus.AWAITING_APPROVAL
    )

    import pytest

    with pytest.raises(ValueError):
        pipeline.approve(
            file_id,
            ApprovedConfig(
                source_format=FileFormat.DELIMITED, delimiter=None
            ),
        )


def test_preamble_file_needs_approval_and_parses_with_skip(tmp_path):
    # A file with a cover paragraph is detected but routed to approval;
    # accepting it with the detected skip_rows yields clean rows.
    pipeline = make_pipeline(
        tmp_path,
        {
            "report.csv": (
                "Monthly Sales Report\n"
                "Generated by Data Operations\n"
                "STORE,UNITS,PRICE\n"
                "1001,45,500\n"
                "1002,60,700\n"
            )
        },
    )
    pipeline.discover()
    pipeline.process()

    assert status_of(pipeline, "report.csv") is FileStatus.AWAITING_APPROVAL
    file_id = next(
        fid
        for fid, f in pipeline.files.items()
        if f.name == "report.csv" and f.status is FileStatus.AWAITING_APPROVAL
    )
    source = pipeline.get_file(file_id)
    result = pipeline._detector.detect(source.local_path)
    assert result.preamble_lines == 2
    assert result.confidence < 0.8  # sign-off required

    pipeline.approve(
        file_id,
        ApprovedConfig(
            source_format=FileFormat.DELIMITED,
            delimiter=",",
            encoding=result.encoding,
            header_present=True,
            columns=result.layout,
            skip_rows=result.preamble_lines,
        ),
    )
    assert status_of(pipeline, "report.csv") is FileStatus.RAW_DELETED
    dataset = complete_datasets(pipeline)["report.csv"]
    assert dataset.row_count == 2
    dataset_dir = dataset_directory(
        tmp_path / "workspace", dataset.dataset_id
    )
    frame = pl.read_parquet(dataset_dir / "report.csv.parquet")
    assert frame.height == 2
    assert frame["STORE"].to_list() == ["1001", "1002"]


def test_reprocess_redetects(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        {"data.csv": 'STORE,UNITS,PRICE\n123,"multi\nline",abc\n456,def,ghi\n'},
    )
    pipeline.discover()
    pipeline.process()
    file_id = next(
        fid
        for fid, f in pipeline.files.items()
        if f.status is FileStatus.AWAITING_APPROVAL
    )
    pipeline.reprocess(file_id)
    # Reprocessing re-detects and lands back at the approval gate.
    assert status_of(pipeline, "data.csv") is FileStatus.AWAITING_APPROVAL


# ---------------------------------------------------------------------
# Failure handling.
# ---------------------------------------------------------------------
def test_download_failure_is_recorded(tmp_path):
    mft = tmp_path / "mft"
    mft.mkdir()
    (mft / "broken.csv").write_text("A,B\n1,2\n")
    workspace = Workspace.at(tmp_path / "workspace")
    pipeline = Pipeline(
        workspace,
        PipelineSettings(
            source_path="/",
            transport_factory=lambda: FailingTransport(mft),
        ),
    )
    pipeline.discover()
    summary = pipeline.process()

    assert status_of(pipeline, "broken.csv") is FileStatus.DOWNLOAD_FAILED
    assert summary.failed == 1
    assert not complete_datasets(pipeline)


def test_empty_file_is_flagged_for_review(tmp_path):
    pipeline = make_pipeline(tmp_path, {"empty.csv": ""})
    pipeline.discover()
    summary = pipeline.process()

    assert status_of(pipeline, "empty.csv") is FileStatus.AWAITING_APPROVAL
    assert summary.pending_approval == 1


def test_discover_does_not_requeue_present_files(tmp_path):
    pipeline = make_pipeline(
        tmp_path,
        {"next.csv": "A,B\n1,2\n"},
        download_workers=2,
    )
    pipeline.discover()
    pipeline.process()
    before = sum(pipeline.discover_count() for _ in range(1))

    pipeline.discover()  # second discovery of the same source
    assert pipeline.discover_count() == before  # nothing new to queue


def test_fixed_width_with_layout_auto_parses(tmp_path):
    layout = tmp_path / "layout.csv"
    layout.write_text("Field,From,Length\nSTORE,1,4\nUNITS,5,3\n", encoding="utf-8")

    pipeline = make_pipeline(
        tmp_path,
        {"fixed.txt": "1001 45\n1002 60\n1003 70\n"},
        layout=layout,
    )
    pipeline.discover()
    pipeline.process()

    assert status_of(pipeline, "fixed.txt") is FileStatus.RAW_DELETED
    dataset = complete_datasets(pipeline)["fixed.txt"]
    assert dataset.row_count == 3


# ---------------------------------------------------------------------
# Phase 9: registry persistence + restart recovery.
# (Recovery + retry specifics live in test_registry_recovery.py so this
# file stays under the 500-line project limit.)
# ---------------------------------------------------------------------
def test_registry_written_on_every_transition(tmp_path):
    pipeline = make_pipeline(tmp_path, {"sales.csv": "A,B\n1,2\n"})
    pipeline.discover()
    registry = pipeline.workspace.root / "registry.json"
    assert registry.exists()

    pipeline.process()
    payload = json.loads(registry.read_text(encoding="utf-8"))
    assert "sales.csv" in [f["name"] for f in payload["files"].values()]


def test_restart_recovery_restores_registry(tmp_path):
    pipeline = make_pipeline(
        tmp_path, {"sales.csv": "STORE,UNITS,PRICE\n1001,45,500\n"}
    )
    pipeline.discover()
    pipeline.process()
    assert status_of(pipeline, "sales.csv") is FileStatus.RAW_DELETED

    # A brand-new Pipeline over the same workspace rehydrates from disk.
    restarted = make_pipeline(tmp_path, {"sales.csv": "STORE,UNITS,PRICE\n999\n"})
    assert status_of(restarted, "sales.csv") is FileStatus.RAW_DELETED
    assert "sales.csv" in complete_datasets(restarted)

    # Discovery sees it as known and never re-queues a finished file.
    before = restarted.discover_count()
    restarted.discover()
    assert restarted.discover_count() == before
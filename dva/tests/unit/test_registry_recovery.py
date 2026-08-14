"""Registry persistence + restart-recovery + retry tests (Phase 9).

Kept separate from test_orchestrator.py so neither file exceeds the
500-line project limit. Covers: crash recovery of transient states, retry
re-queueing semantics (all failure states actually reprocess), stale `.part`
sweeps and a corrupt-registry startup.
"""

import json

import pytest

from models.file_models import FileStatus, SourceFile
from services.mft.transport import LocalTransport
from services.orchestrator.orchestrator import (
    Pipeline,
    PipelineSettings,
    Workspace,
)


def seed_registry(workspace: Workspace, sources: list[SourceFile]) -> None:
    """Write a registry.json keyed by file_id directly (crash simulation)."""
    payload = {
        "version": 1,
        "files": {source.file_id: source.to_json_dict() for source in sources},
        "approved": {},
    }
    path = workspace.root / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def seeded_source(name, status) -> SourceFile:
    return SourceFile(name=name, remote_path=f"/{name}", status=status)


def status_of(pipeline, name):
    match = [f for f in pipeline.files.values() if f.name == name]
    assert match, f"No file named {name}"
    return match[0].status


def make_recovery_pipeline(tmp_path, workspace) -> Pipeline:
    mft = tmp_path / "mft"
    mft.mkdir(exist_ok=True)
    return Pipeline(
        workspace,
        PipelineSettings(
            source_path="/",
            transport_factory=lambda: LocalTransport(mft),
        ),
    )


def test_recover_inflight_requeues_transient_states(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    seed_registry(
        workspace,
        [
            seeded_source("downloading.csv", FileStatus.DOWNLOADING),
            seeded_source("detecting.csv", FileStatus.DETECTING),
            seeded_source("parsing.csv", FileStatus.PARSING),
            seeded_source("waiting.csv", FileStatus.AWAITING_APPROVAL),
            seeded_source("failed.csv", FileStatus.DOWNLOAD_FAILED),
            seeded_source("finished.csv", FileStatus.RAW_DELETED),
        ],
    )
    pipeline = make_recovery_pipeline(tmp_path, workspace)
    assert status_of(pipeline, "downloading.csv") is FileStatus.DOWNLOAD_QUEUED
    assert status_of(pipeline, "detecting.csv") is FileStatus.DOWNLOAD_VERIFIED
    assert status_of(pipeline, "parsing.csv") is FileStatus.DOWNLOAD_VERIFIED
    # Stable states are left untouched.
    assert status_of(pipeline, "waiting.csv") is FileStatus.AWAITING_APPROVAL
    assert status_of(pipeline, "failed.csv") is FileStatus.DOWNLOAD_FAILED
    assert status_of(pipeline, "finished.csv") is FileStatus.RAW_DELETED


def test_process_finishes_recovered_raw_cleanup(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    (workspace.raw_dir / "leftover.csv").write_text("raw", encoding="utf-8")
    source = seeded_source("leftover.csv", FileStatus.RAW_CLEANUP)
    source.local_path = workspace.raw_dir / "leftover.csv"
    seed_registry(workspace, [source])

    pipeline = make_recovery_pipeline(tmp_path, workspace)
    assert status_of(pipeline, "leftover.csv") is FileStatus.RAW_CLEANUP
    pipeline.process()
    assert status_of(pipeline, "leftover.csv") is FileStatus.RAW_DELETED
    assert not (workspace.raw_dir / "leftover.csv").exists()


def test_retry_failed_requeues_all_failures(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    seed_registry(
        workspace,
        [
            seeded_source("a.csv", FileStatus.DOWNLOAD_FAILED),
            seeded_source("b.csv", FileStatus.PARSE_FAILED),
            seeded_source("c.csv", FileStatus.AWAITING_APPROVAL),
        ],
    )
    pipeline = make_recovery_pipeline(tmp_path, workspace)
    assert pipeline.retry_failed() == 2
    assert status_of(pipeline, "a.csv") is FileStatus.DOWNLOAD_QUEUED
    # Parse-stage failures re-enter the parse pool (matching recovery) rather
    # than being stranded at a mid-pipeline state, so a subsequent process()
    # call genuinely reprocesses them.
    assert status_of(pipeline, "b.csv") is FileStatus.DOWNLOAD_VERIFIED
    assert status_of(pipeline, "c.csv") is FileStatus.AWAITING_APPROVAL


def test_retry_then_process_reparses_retried_file(tmp_path):
    """Regressed bug: retried write/verify-stage files were re-queued to
    mid-pipeline states that process() never inspects, so retry was a no-op."""
    recorded_source = seeded_source("a.csv", FileStatus.WRITE_FAILED)
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    # Retry skips download; put the raw file back where the parse pool finds it
    # and point local_path at it, exactly as a real persisted registry would.
    workspace.raw_dir.mkdir(exist_ok=True)
    raw_path = workspace.raw_dir / "a.csv"
    raw_path.write_text("STORE,UNITS,PRICE\n1001,2,5\n", encoding="utf-8")
    recorded_source.local_path = raw_path
    seed_registry(workspace, [recorded_source])
    mft = tmp_path / "mft"
    mft.mkdir(exist_ok=True)
    (mft / "a.csv").write_text("STORE,UNITS,PRICE\n1001,2,5\n", encoding="utf-8")
    pipeline = Pipeline(
        workspace,
        PipelineSettings(
            source_path="/",
            transport_factory=lambda: LocalTransport(mft),
        ),
    )
    assert status_of(pipeline, "a.csv") is FileStatus.WRITE_FAILED
    pipeline.retry_failed()
    assert status_of(pipeline, "a.csv") is FileStatus.DOWNLOAD_VERIFIED
    pipeline.process()
    assert status_of(pipeline, "a.csv") is FileStatus.RAW_DELETED


def test_retry_file_requeues_one_and_persists(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    seed_registry(
        workspace,
        [seeded_source("a.csv", FileStatus.DOWNLOAD_FAILED)],
    )
    mft = tmp_path / "mft"
    mft.mkdir(exist_ok=True)
    pipeline = Pipeline(
        workspace,
        PipelineSettings(
            source_path="/",
            transport_factory=lambda: LocalTransport(mft),
        ),
    )
    file_id = next(iter(pipeline.files))
    pipeline.retry_file(file_id)
    assert status_of(pipeline, "a.csv") is FileStatus.DOWNLOAD_QUEUED

    # The re-queued status survives a restart from the persisted registry.
    restarted = Pipeline(
        workspace,
        PipelineSettings(
            source_path="/",
            transport_factory=lambda: LocalTransport(mft),
        ),
    )
    assert status_of(restarted, "a.csv") is FileStatus.DOWNLOAD_QUEUED


def test_retry_file_rejects_non_failed(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    seed_registry(
        workspace,
        [seeded_source("ok.csv", FileStatus.AWAITING_APPROVAL)],
    )
    pipeline = make_recovery_pipeline(tmp_path, workspace)
    with pytest.raises(ValueError, match="Only failed files"):
        pipeline.retry_file(next(iter(pipeline.files)))


def test_cleanup_stale_parts_sweeps_orphans(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    (workspace.raw_dir / "half.csv.part").write_text("x", encoding="utf-8")
    (workspace.raw_dir / "ready.csv").write_text("x", encoding="utf-8")
    dataset_dir = workspace.datasets_dir / "abc123"
    dataset_dir.mkdir()
    (dataset_dir / "sales.csv.parquet.part").write_text("x", encoding="utf-8")
    (dataset_dir / "sales.csv.parquet").write_text("x", encoding="utf-8")

    pipeline = make_recovery_pipeline(tmp_path, workspace)
    assert pipeline.cleanup_stale_parts() == 2
    assert not (workspace.raw_dir / "half.csv.part").exists()
    assert not (dataset_dir / "sales.csv.parquet.part").exists()
    assert (workspace.raw_dir / "ready.csv").exists()
    assert (dataset_dir / "sales.csv.parquet").exists()


def test_corrupt_registry_does_not_block_startup(tmp_path):
    workspace = Workspace.at(tmp_path / "workspace").ensure()
    (workspace.root / "registry.json").write_text("{not json", encoding="utf-8")
    pipeline = make_recovery_pipeline(tmp_path, workspace)
    assert pipeline.files == {}
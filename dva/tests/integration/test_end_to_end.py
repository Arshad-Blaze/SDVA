"""End-to-end integration: mock MFT -> staging -> detect -> parse -> parquet
-> validate -> report (reference testing requirements).

These tests drive the real Orchestrator Pipeline against a local "MFT" folder
and then hand the resulting COMPLETE datasets to the Validator and Reports,
proving the whole boundary chain works together.
"""

import gzip

import polars as pl
import pytest

from models.dataset_models import DatasetStatus, dataset_directory
from models.file_models import FileStatus
from models.report_models import ItemValidationConfig
from models.validation_models import (
    ColumnMapping,
    PriceType,
    UnitType,
    ValidationConfig,
)
from services.mft.transport import LocalTransport
from services.orchestrator.orchestrator import (
    Pipeline,
    PipelineSettings,
    Workspace,
)
from services.reporting.report_service import generate_reports
from services.validator.validator import validate

BAU_CSV = (
    "STORE,UNITS,PRICE,UPC,DESC\n"
    "1001,2,10,11111,aaa\n"
    "1001,3,10,11111,aaa\n"
    "1002,5,20,22222,bbb\n"
)
TEST_CSV = (
    "STORE,UNITS,PRICE,UPC,DESC\n"
    "1001,1,10,11111,aaa\n"
    "1003,7,5,33333,ccc\n"
)


def make_pipeline(tmp_path, files: dict[str, bytes | str]):
    """LocalTransport with the given ``files`` as the MFT source."""
    mft = tmp_path / "mft"
    mft.mkdir(exist_ok=True)
    for name, content in files.items():
        payload = content.encode("utf-8") if isinstance(content, str) else content
        (mft / name).write_bytes(payload)

    workspace = Workspace.at(tmp_path / "workspace")
    settings = PipelineSettings(
        source_path="/",
        transport_factory=lambda: LocalTransport(mft),
        download_workers=2,
        parser_workers=1,
    )
    return Pipeline(workspace, settings), mft


def process_to_done(pipeline):
    """Discover + process until a pass moves nothing (files either finished,
    await approval, or are terminal failures)."""
    pipeline.discover()
    summary = pipeline.process()
    while True:
        progress = pipeline.process()
        if (progress.downloaded, progress.parsed) == (0, 0):
            return summary
        summary.parsed += progress.parsed


def dataset_parquet(pipeline, file_name):
    match = [d for d in pipeline.datasets.values()
             if d.source_file_name == file_name and d.status is DatasetStatus.COMPLETE]
    assert match, f"no COMPLETE dataset for {file_name}"
    dataset = match[0]
    return dataset_directory(pipeline.workspace.root, dataset.dataset_id) / dataset.parquet_files[0]


def test_end_to_end_bau_test_validate_report(tmp_path):
    pipeline, _ = make_pipeline(
        tmp_path,
        {"bau.csv": BAU_CSV, "test.csv.gz": gzip.compress(TEST_CSV.encode("utf-8"))},
    )
    process_to_done(pipeline)

    # Both files (one compressed) reached the end of the lifecycle.
    assert {f.name: f.status for f in pipeline.files.values()} == {
        "bau.csv": FileStatus.RAW_DELETED,
        "test.csv.gz": FileStatus.RAW_DELETED,
    }
    assert len(pipeline.datasets) == 2

    # The gz pipeline produced a PARQUET dataset with the decompressed content.
    test_df = pl.read_parquet(dataset_parquet(pipeline, "test.csv.gz"))
    assert test_df.height == 2
    assert set(test_df["STORE"].to_list()) == {"1001", "1003"}

    # --- Validator over the real dataset paths (Boundary 9). ---
    mapping = ColumnMapping(
        store_col="STORE",
        units_col="UNITS",
        price_col="PRICE",
        upc_col="UPC",
        desc_col="DESC",
        units_type=UnitType.QTY,
    )
    config = ValidationConfig(
        store_analysis=True, sales_analysis=True, upc_analysis=True,
        bau=mapping, test=mapping, price_type_bau=PriceType.UNIT_PRICE,
    )
    result = validate(
        dataset_parquet(pipeline, "bau.csv"),
        dataset_parquet(pipeline, "test.csv.gz"),
        config,
    )
    sales_comparison = result.frames["sales"]["comparison"]
    rows = {r["STORE_NUMBER"]: r for r in sales_comparison.iter_rows(named=True)}
    assert rows["1001"]["UNITS DIFFERENCE"] == 4  # BAU 5 vs TEST 1
    assert dict(result.metrics["upc"])["UPCs missing in TEST"] == "1"

    # --- Reports over the same datasets (Boundary 10). ---
    item_cfg = ItemValidationConfig(
        bau=ColumnMapping(upc_col="UPC", desc_col="DESC", units_col="UNITS", price_col="PRICE"),
        test=ColumnMapping(upc_col="UPC", desc_col="DESC", units_col="UNITS", price_col="PRICE"),
    )
    report = generate_reports(
        dataset_parquet(pipeline, "bau.csv"),
        dataset_parquet(pipeline, "test.csv.gz"),
        item_cfg,
        tmp_path / "reports",
    )
    comparison = pl.read_parquet(report.artifacts["comparison"])
    assert comparison["UPC"].to_list().count("11111") == 1
    assert report.artifacts["excel"].exists()


def test_ten_file_batch_failure_is_isolated(tmp_path):
    """Concurrency requirement: 10-file batch, one decompression failure does
    not stop or reprocess the nine healthy files."""
    files = {"f%02d.csv" % i: f"STORE,UNITS,PRICE\n10{i},1,5\n" for i in range(9)}
    files["big.csv"] = "STORE,UNITS,PRICE\n" + "1001,1,5\n" * 200  # slower parse
    files["corrupt.csv.gz"] = b"not a gzip stream at all"

    pipeline, _ = make_pipeline(tmp_path, files)
    summary = process_to_done(pipeline)

    statuses = {f.name: f.status for f in pipeline.files.values()}
    assert statuses["corrupt.csv.gz"] is FileStatus.DECOMPRESSION_FAILED
    assert summary.failed == 1
    # Every other file completed independently (raw cleaned up).
    assert len([f for f in statuses.values() if f is FileStatus.RAW_DELETED]) == 10

    # Correct the local raw (the broken archive), then retry only that file.
    corrupt_id = next(
        f.file_id for f in pipeline.files.values() if f.name == "corrupt.csv.gz"
    )
    dirty = pipeline.get_file(corrupt_id).local_path
    dirty.write_bytes(gzip.compress(b"STORE,UNITS\n1001,2\n"))
    pipeline.retry_file(corrupt_id)
    assert pipeline.get_file(corrupt_id).status is FileStatus.DOWNLOAD_VERIFIED
    pipeline.process()
    assert pipeline.get_file(corrupt_id).status is FileStatus.RAW_DELETED


def test_stale_part_and_decompressed_sweep(tmp_path):
    raw = tmp_path / "workspace" / "raw"
    raw.mkdir(parents=True)
    (raw / "leftover.part").write_text("x")
    (raw / "leftover.csv.gz.decompressed").write_text("x")

    pipeline, _ = make_pipeline(tmp_path, {"a.csv": "a\n1\n"})
    removed = pipeline.cleanup_stale_parts()
    assert removed == 2
    assert not (raw / "leftover.part").exists()
    assert not (raw / "leftover.csv.gz.decompressed").exists()
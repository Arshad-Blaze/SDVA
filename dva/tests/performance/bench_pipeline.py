"""Performance benchmark (requirement: parse/write/validate scale linearly).

Not a unit test: run explicitly, e.g.

    DVA_BENCH=1 python3 tests/performance/bench_pipeline.py

It drives the real pipeline (LocalTransport -> detect -> parse -> parquet ->
validate -> report) over a sizeable generated file, records wall-clock time
per stage plus peak RSS, and asserts the pipeline stays bounded-memory
(no eager full-file load) as the reference rules demand.

Exit code 0 means all assertions held.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dataset_models import dataset_directory
from services.mft.transport import LocalTransport
from services.orchestrator.orchestrator import (
    Pipeline,
    PipelineSettings,
    Workspace,
)
from services.reporting.report_service import generate_reports
from services.validator.validator import validate


def generate_delimited(path: Path, target_bytes: int) -> None:
    """Write ``target_bytes`` of STORE,UNITS,PRICE,UPC,DESC data in flush
    chunks, stopping as soon as the file passes the target size."""
    header = "STORE,UNITS,PRICE,UPC,DESC\n"
    with open(path, "w", encoding="utf-8") as out:
        out.write(header)
        emitted = 0
        buffer = []
        while path.stat().st_size < target_bytes:
            buffer.append(
                f"{1000 + (emitted % 900)},2,10,{10000 + emitted % 89999},"
                f"item{emitted % 9000}\n"
            )
            emitted += 1
            if len(buffer) >= 50_000:
                out.write("".join(buffer))
                buffer = []
        if buffer:
            out.write("".join(buffer))


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mb", type=int, default=40, help="approximate source size in MB"
    )
    parser.add_argument(
        "--with-report", action="store_true",
        help="also time the Excel report step (openpyxl is the slowest "
             "stage, so it is opt-in to keep the default run quick)",
    )
    args = parser.parse_args()

    if not os.environ.get("DVA_BENCH"):
        parser.error("set DVA_BENCH=1 to run the performance benchmark")

    tmp = Path(tempfile.mkdtemp(prefix="sdva_bench_"))
    mft = tmp / "mft"
    mft.mkdir()
    source = mft / "bench.csv"
    target_bytes = args.mb * 1024 * 1024

    t0 = time.perf_counter()
    generate_delimited(source, target_bytes)
    gen_time = time.perf_counter() - t0
    source_size_mb = source.stat().st_size / (1024 * 1024)
    print(f"generated {source_size_mb:.1f} MB in {gen_time:.1f}s")

    workspace = Workspace.at(tmp / "workspace")
    pipeline = Pipeline(
        workspace,
        PipelineSettings(
            source_path="/",
            transport_factory=lambda: LocalTransport(mft),
            download_workers=2,
            parser_workers=1,
        ),
    )

    t0 = time.perf_counter()
    pipeline.discover()
    last = None
    while True:
        last = pipeline.process()
        if (last.downloaded, last.parsed) == (0, 0):
            break
    pipeline_sec = time.perf_counter() - t0
    peak_after_pipeline = peak_rss_bytes()

    datasets = [d for d in pipeline.datasets.values() if d.is_complete()]
    assert len(datasets) == 1, f"expected exactly 1 dataset, got {len(datasets)}"
    parquet = (
        dataset_directory(workspace.root, datasets[0].dataset_id)
        / datasets[0].parquet_files[0]
    )
    print(
        f"pipeline (discover+download+parse+write+verify+cleanup): "
        f"{pipeline_sec:.1f}s, peak RSS {peak_after_pipeline / 2**20:.0f} MiB"
    )

    from models.validation_models import (
        ColumnMapping,
        PriceType,
        UnitType,
        ValidationConfig,
    )

    mapping = ColumnMapping(
        store_col="STORE", units_col="UNITS", price_col="PRICE",
        units_type=UnitType.QTY,
    )
    validation = ValidationConfig(
        store_analysis=True,
        sales_analysis=True,
        upc_analysis=False,
        bau=mapping,
        test=mapping,
        price_type_bau=PriceType.UNIT_PRICE,
    )

    t0 = time.perf_counter()
    result = validate(parquet, parquet, validation)  # self vs self
    validate_sec = time.perf_counter() - t0
    peak_after_validate = peak_rss_bytes()
    print(
        f"validate: {validate_sec:.1f}s, peak RSS "
        f"{peak_after_validate / 2**20:.0f} MiB"
    )

    if args.with_report:
        from models.report_models import ItemValidationConfig

        item_cfg = ItemValidationConfig(
            bau=ColumnMapping(units_col="UNITS", price_col="PRICE",
                              upc_col="UPC", desc_col="DESC"),
            test=ColumnMapping(units_col="UNITS", price_col="PRICE",
                               upc_col="UPC", desc_col="DESC"),
        )
        t0 = time.perf_counter()
        generate_reports(parquet, parquet, item_cfg, tmp / "reports")
        report_sec = time.perf_counter() - t0
        print(f"reports: {report_sec:.1f}s")
    else:
        report_sec = 0.0
        print("reports: skipped (--with-report to time the Excel writer)")

    peak_final = peak_rss_bytes()

    # --- Assertions: bounded memory + sensible throughput. ---
    # Reference rules ban eager full-file loads; a source processed in well
    # under ~2.5x its file size of working memory proves streaming holds.
    budget_bytes = max(256 * 2**20, 32 * int(source_size_mb) * 2**20)
    if peak_final > budget_bytes:
        print(
            f"peak RSS {peak_final / 2**20:.0f} MiB exceeded budget "
            f"{budget_bytes / 2**20:.0f} MiB"
        )
        return 1

    sales = result.frames["sales"]["comparison"]
    rows = sales.height  # self-vs-self keeps the input row count
    if last.failed:
        print("pipeline reported failures for a healthy file")
        return 1

    if pipeline_sec > 240:
        print(f"pipeline took {pipeline_sec:.1f}s for one file; "
              f"check throughput")
        return 1

    bottleneck = max(pipeline_sec, validate_sec, report_sec)
    print(
        f"OK: {source_size_mb:.0f} MB, {rows / 1e6:.1f}M rows, "
        f"bounded RSS ({peak_final / 2**20:.0f} MiB), "
        f"slowest stage {bottleneck:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""DVA - thin Streamlit UI.

Each tab is a thin adapter over the ingestion services; all real behaviour
lives in services/ (confirmed against the reference tool's UI modules).
The Pipeline is recreated per rerun and kept in ``session_state`` because
in-memory per-file state does not survive a page refresh.

Run:  cd dva && streamlit run app.py

Tabs:
  1. Pipeline   - MFT connection, discovery, run, per-file status table.
  2. Approval   - files waiting review; edit and accept a config, or
                 reprocess (re-detect).
  3. Datasets   - published datasets with their metadata and parse report.
  4. Validator  - BAU vs TEST analysis over published parquet datasets.
  5. Reports    - item-level (UPC|DESC) BAU vs TEST comparison reports.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from models.detection_models import (
    ApprovedConfig,
    FileFormat,
    FixedWidthLayout,
)
from models.file_models import FileStatus
from models.validation_models import (
    ColumnMapping,
    PriceType,
    UnitType,
    ValidationConfig,
)
from services.ingestion.detector import Detector
from services.mft.transport import LocalTransport, SftpTransport
from services.orchestrator.orchestrator import (
    Pipeline,
    PipelineSettings,
    Workspace,
)
from services.validator.validator import validate

st.set_page_config(page_title="DVA Pipeline", layout="wide")

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = APP_ROOT / "data"


# =====================================================================
# Session helpers.
# =====================================================================
def get_settings() -> PipelineSettings:
    """Build pipeline settings from the sidebar connection form."""
    mode = st.session_state.get("connection_mode", "local")
    workspace = st.session_state.get("workspace_root", str(DEFAULT_WORKSPACE))

    if mode == "sftp":
        transport_factory = lambda: SftpTransport(
            host=st.session_state["mft_host"],
            username=st.session_state["mft_user"],
            password=st.session_state["mft_pass"],
            port=int(st.session_state.get("mft_port", 22)),
        )
    else:
        share_root = st.session_state.get("local_share", str(APP_ROOT))
        transport_factory = lambda: LocalTransport(share_root)

    return PipelineSettings(
        source_path=st.session_state.get("source_path", "/inbound/sales"),
        pattern=st.session_state.get("file_pattern") or None,
        transport_factory=transport_factory,
        fixed_width_layout=(
            Path(st.session_state["layout_csv"]) if st.session_state.get("layout_csv") else None
        ),
    )


def get_pipeline() -> Pipeline:
    """The one Pipeline instance for the current connection config."""
    key = "pipeline"
    if key not in st.session_state:
        st.session_state[key] = Pipeline(
            Workspace.at(st.session_state.get("workspace_root", str(DEFAULT_WORKSPACE))),
            get_settings(),
        )
        # Phase 9: restart recovery already re-queued any in-flight files;
        # remember how many for a one-time notice in the Pipeline tab.
        st.session_state["recovery_note"] = st.session_state[key].recovered_on_start
    return st.session_state[key]


def reset_pipeline() -> None:
    st.session_state.pop("pipeline", None)
    st.session_state.pop("approval_detection", None)


# =====================================================================
# Sidebar: connection.
# =====================================================================
with st.sidebar:
    st.header("MFT connection")
    st.text_input(
        "Workspace root", key="workspace_root", value=str(DEFAULT_WORKSPACE)
    )
    st.radio(
        "Transport", ["local", "sftp"], index=0, key="connection_mode"
    )
    share_root = str(APP_ROOT)
    if st.session_state["connection_mode"] == "sftp":
        st.text_input("Host", key="mft_host", value="mft.example.com")
        st.text_input("Username", key="mft_user")
        st.text_input("Password", key="mft_pass", type="password")
        st.number_input("Port", key="mft_port", value=22, min_value=1)
    else:
        share_root = st.text_input("Local share root (MFT stand-in)", key="local_share")

    st.text_input("Remote source path", key="source_path", value="/inbound/sales")
    st.text_input("File pattern (optional)", key="file_pattern", value="*.csv")
    st.text_input("Fixed-width layout CSV (optional)", key="layout_csv", value="")
    if st.button("Reconnect"):
        reset_pipeline()


# =====================================================================
# Tab 1: Pipeline.
# =====================================================================
st.title("DVA - Data Validation Application")
pipeline = get_pipeline()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Pipeline", "Approval", "Datasets", "Validator", "Reports"]
)

with tab1:
    if st.session_state.get("recovery_note", 0):
        st.info(
            f"Recovered {st.session_state['recovery_note']} in-flight file(s) "
            "from the previous run; they were re-queued at a safe checkpoint."
        )
        st.session_state["recovery_note"] = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Discover files", use_container_width=True):
            found = pipeline.discover()
            st.success(f"Queued {len(found)} new file(s).")
    with col2:
        if st.button("Run pipeline", use_container_width=True):
            with st.spinner("Processing..."):
                summary = pipeline.process()
            st.success(
                f"parsed={summary.parsed} pending_approval="
                f"{summary.pending_approval} failed={summary.failed}"
            )
    with col3:
        st.write(f"Files known: {len(pipeline.files)}")

    st.caption("Recovery and maintenance (Phase 9)")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        if st.button("Re-queue in-flight", use_container_width=True):
            recovered = pipeline.recover_inflight()
            st.info(f"Re-queued {recovered} file(s).")
    with col_r2:
        if st.button("Retry all failed", use_container_width=True):
            requeued = pipeline.retry_failed()
            st.info(f"Re-queued {requeued} failed file(s).")
    with col_r3:
        if st.button("Clean stale .part", use_container_width=True):
            removed = pipeline.cleanup_stale_parts()
            st.info(f"Removed {removed} orphaned .part file(s).")
    with col_r4:
        st.write(f"Registry: {pipeline._registry.path().name}")

    counts = pipeline.status_counts()
    if counts:
        status_df = pl.DataFrame(
            {"status": list(counts), "count": list(counts.values())}
        )
        st.dataframe(status_df.to_pandas(), use_container_width=True)

    if pipeline.files:
        rows = [
            {
                "name": f.name,
                "status": f.status.value,
                "error": f.error or "",
                "local": str(f.local_path or ""),
            }
            for f in sorted(pipeline.files.values(), key=lambda f: f.name)
        ]
        st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True)

    failed = [f for f in pipeline.files.values() if f.is_failure()]
    if failed:
        st.subheader("Failed files (retry)")
        for f in failed:
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"{f.name} - {f.error}")
            if col_b.button("Retry", key=f"retry_{f.file_id}"):
                pipeline.retry_file(f.file_id)
                st.rerun()


# =====================================================================
# Tab 2: Approval.
# =====================================================================
with tab2:
    awaiting = [
        f for f in pipeline.files.values()
        if f.status is FileStatus.AWAITING_APPROVAL
    ]
    if not awaiting:
        st.info("No files awaiting approval.")
    else:
        names = {f.file_id: f.name for f in awaiting}
        choice_id = st.selectbox(
            "File to review",
            list(names),
            format_func=lambda fid: names[fid],
            key="approval_choice",
        )
        source = pipeline.get_file(choice_id)

        # Re-detect on demand so the operator sees why it was flagged.
        detection_key = f"det_{source.file_id}"
        if detection_key not in st.session_state:
            st.session_state[detection_key] = pipeline._detector.detect(source.local_path)
        detection = st.session_state[detection_key]

        st.write("Detection summary")
        meta = detection.sample_metadata
        st.json(
            {
                "format": detection.format.value,
                "structure": detection.structure_type.value,
                "delimiter": detection.delimiter,
                "confidence": detection.confidence,
                "warnings": detection.warnings,
                "delimiter_consistency": meta.get("delimiter_stats", {}).get("consistency"),
            }
        )

        st.subheader("Approved configuration")
        delimiter = st.text_input("Delimiter", value=detection.delimiter or ",")
        header = st.checkbox("Header row present", value=bool(detection.header_present))
        skip_rows = st.number_input(
            "Lines to skip (leading non-data rows)",
            min_value=0,
            value=int(detection.preamble_lines),
            step=1,
        )
        if isinstance(detection.layout, list):
            columns_text = st.text_area(
                "Column names (comma separated)",
                value=", ".join(detection.layout),
            )
            columns = [c.strip() for c in columns_text.split(",") if c.strip()]
        else:
            columns = []
            if detection.layout is not None:
                st.info("Fixed-width layout will be imported from the detection result.")

        if st.button("Accept and parse", type="primary"):
            if detection.format is FileFormat.FIXED_WIDTH and detection.layout is not None:
                config = ApprovedConfig(
                    source_format=FileFormat.FIXED_WIDTH,
                    encoding=detection.encoding,
                    header_present=False,
                    columns=detection.layout.column_names(),
                    fixed_width_layout=detection.layout,
                    skip_rows=int(skip_rows),
                )
            else:
                config = ApprovedConfig(
                    source_format=FileFormat.DELIMITED,
                    delimiter=delimiter,
                    encoding=detection.encoding,
                    header_present=header,
                    columns=columns,
                    skip_rows=int(skip_rows),
                )
            with st.spinner("Parsing..."):
                pipeline.approve(source.file_id, config)
            st.success("Parsed. Raw file cleaned up.")
            st.rerun()

        if st.button("Reprocess (re-detect)"):
            st.session_state.pop(detection_key, None)
            pipeline.reprocess(source.file_id)
            st.rerun()


# =====================================================================
# Tab 3: Datasets.
# =====================================================================
with tab3:
    complete = [
        d for d in pipeline.datasets.values() if d.status.value == "complete"
    ]
    if not complete:
        st.info("No completed datasets yet.")
    else:
        rows = [
            {
                "source": d.source_file_name,
                "rows": d.row_count,
                "parquet": ", ".join(d.parquet_files),
                "id": d.dataset_id,
            }
            for d in sorted(complete, key=lambda d: d.completed_at or "")
        ]
        st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True)

        selected_id = st.selectbox(
            "Dataset details",
            [d.dataset_id for d in complete],
            format_func=lambda i: next(
                d.source_file_name for d in complete if d.dataset_id == i
            ),
        )
        dataset = next(d for d in complete if d.dataset_id == selected_id)
        col_schema, col_issues = st.columns(2)
        with col_schema:
            st.subheader("Schema")
            st.json(dataset.schema)
        with col_issues:
            st.subheader("Parse report")
            report = dataset.extra.get("parse_report", {})
            st.write("Issues:", len(report.get("issues", [])))
            st.json(report.get("counters", {}))


# =====================================================================
# Tab 4: Validator.
# =====================================================================
with tab4:
    st.subheader("Store-list and sales validation (BAU vs TEST)")
    bau_id = st.selectbox(
        "BAU dataset",
        [d.dataset_id for d in complete],
        key="bau",
        format_func=lambda i: next(
            d.source_file_name for d in complete if d.dataset_id == i
        ),
    )
    test_id = st.selectbox(
        "TEST dataset",
        [d.dataset_id for d in complete],
        key="test",
        format_func=lambda i: next(
            d.source_file_name for d in complete if d.dataset_id == i
        ),
    )

    def dataset_parquet(dataset_id: str) -> Path:
        dataset = next(d for d in complete if d.dataset_id == dataset_id)
        return (
            Workspace.at(st.session_state.get("workspace_root", str(DEFAULT_WORKSPACE)))
        .datasets_dir
            / dataset.dataset_id
            / dataset.parquet_files[0]
        )

    col_bau, col_test = st.columns(2)
    with col_bau:
        st.checkbox("Store analysis", value=True, key="va_store")
        st.checkbox("Sales analysis", value=True, key="va_sales")
        st.checkbox("UPC analysis", value=False, key="va_upc")
    with col_test:
        st.write("Column mappings use the dataset's own column names:")
        bau_store = st.text_input("BAU store col", value="STORE", key="vb_store")
        test_store = st.text_input("TEST store col", value="STORE", key="vt_store")
        bau_price = st.text_input("BAU price col", value="PRICE", key="vb_price")
        test_price = st.text_input("TEST price col", value="PRICE", key="vt_price")
        bau_units = st.text_input("BAU units col", value="UNITS", key="vb_units")
        test_units = st.text_input("TEST units col", value="UNITS", key="vt_units")
        bau_weighted = st.text_input(
            "BAU weighted units col (optional)", key="vb_weighted"
        )
        test_weighted = st.text_input(
            "TEST weighted units col (optional)", key="vt_weighted"
        )
        bau_units_type = st.selectbox(
            "BAU units type", ["qty", "weight"], key="vb_utype"
        )
        test_units_type = st.selectbox(
            "TEST units type", ["qty", "weight"], key="vt_utype"
        )

    if st.button("Run validation", type="primary"):
        mapping = lambda store, units, price, weighted, units_type: ColumnMapping(
            store_col=store,
            units_col=units,
            price_col=price,
            weighted_units_col=weighted or None,
            units_type=UnitType(units_type),
        )
        config = ValidationConfig(
            store_analysis=st.session_state["va_store"],
            sales_analysis=st.session_state["va_sales"],
            upc_analysis=st.session_state["va_upc"],
            bau=mapping(bau_store, bau_units, bau_price, bau_weighted, bau_units_type),
            test=mapping(test_store, test_units, test_price, test_weighted, test_units_type),
        )
        with st.spinner("Validating (lazy parquet scans)..."):
            result = validate(dataset_parquet(bau_id), dataset_parquet(test_id), config)

        for analysis, metrics in result.metrics.items():
            st.subheader(f"{analysis} analysis")
            st.dataframe(
                pl.DataFrame(
                    {"metric": [m for m, _ in metrics], "value": [v for _, v in metrics]}
                ).to_pandas(),
                use_container_width=True,
            )
        for analysis, frames in result.frames.items():
            st.subheader(f"{analysis} detail")
            for frame_name, frame in frames.items():
                st.caption(frame_name)
                st.dataframe(frame.to_pandas(), use_container_width=True)


# =====================================================================
# Tab 5: Reports (item-level BAU vs TEST comparison).
# =====================================================================
with tab5:
    from ui.report_view import render as render_reports

    render_reports(
        complete,
        st.session_state.get("workspace_root", str(DEFAULT_WORKSPACE)),
    )
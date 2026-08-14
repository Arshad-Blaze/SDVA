"""Reports tab adapter: item-level BAU vs TEST comparison reports.

Thin Streamlit view over the reporting service (Boundary 10, final step:
... -> Validate -> Report). The data always lives on disk as Parquet; only
the small UI frames are kept in session state.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from models.report_models import ItemValidationConfig
from models.validation_models import ColumnMapping, PriceType, UnitType
from services.reporting.report_service import generate_reports


def _dataset_parquet(
    complete: list, dataset_id: str, workspace_root: str
) -> str:
    """Full path of a COMPLETE dataset's first parquet file on disk."""
    dataset = next(d for d in complete if d.dataset_id == dataset_id)
    return str(
        Path(workspace_root)
        / "datasets"
        / dataset.dataset_id
        / dataset.parquet_files[0]
    )


def render(complete: list, workspace_root: str) -> None:
    """Render the item-level report tab.

    Args:
        complete: list of COMPLETE DatasetMetadata (selectors).
        workspace_root: workspace root; the surrogate per-dataset paths are
            resolved from the dataset metadata directory on disk.
    """
    st.subheader("Item-level report (UPC|DESC comparison)")

    col_r_bau, col_r_test = st.columns(2)
    with col_r_bau:
        rep_bau = st.selectbox(
            "BAU dataset",
            [d.dataset_id for d in complete],
            key="rep_bau",
            format_func=lambda i: next(
                d.source_file_name for d in complete if d.dataset_id == i
            ),
        )
        upc_bau = st.text_input("BAU UPC col", value="UPC", key="ru_bau")
        desc_bau = st.text_input("BAU description col", value="DESC", key="rd_bau")
        units_bau = st.text_input("BAU units col", value="UNITS", key="rn_bau")
        price_bau = st.text_input("BAU price col", value="PRICE", key="rp_bau")
        wtd_bau = st.text_input("BAU weighted units col (optional)", key="rw_bau")
        units_type_bau = st.selectbox(
            "BAU units type", ["qty", "weight"], key="rut_bau"
        )
        st.checkbox("Unit price", key="rpt_bau")
        st.checkbox("Implied dollars (x100)", key="rid_bau")
    with col_r_test:
        rep_test = st.selectbox(
            "TEST dataset",
            [d.dataset_id for d in complete],
            key="rep_test",
            format_func=lambda i: next(
                d.source_file_name for d in complete if d.dataset_id == i
            ),
        )
        upc_test = st.text_input("TEST UPC col", value="UPC", key="ru_test")
        desc_test = st.text_input("TEST description col", value="DESC", key="rd_test")
        units_test = st.text_input("TEST units col", value="UNITS", key="rn_test")
        price_test = st.text_input("TEST price col", value="PRICE", key="rp_test")
        wtd_test = st.text_input("TEST weighted units col (optional)", key="rw_test")
        units_type_test = st.selectbox(
            "TEST units type", ["qty", "weight"], key="rut_test"
        )
        st.checkbox("Unit price", key="rpt_test")
        st.checkbox("Implied dollars (x100)", key="rid_test")

    st.text_input(
        "Output directory",
        value=f"{workspace_root}/reports",
        key="report_dir",
    )

    if st.button("Generate report", type="primary"):
        cfg = ItemValidationConfig(
            bau=ColumnMapping(
                upc_col=upc_bau,
                desc_col=desc_bau,
                units_col=units_bau,
                price_col=price_bau,
                weighted_units_col=wtd_bau or None,
                units_type=UnitType(units_type_bau),
            ),
            test=ColumnMapping(
                upc_col=upc_test,
                desc_col=desc_test,
                units_col=units_test,
                price_col=price_test,
                weighted_units_col=wtd_test or None,
                units_type=UnitType(units_type_test),
            ),
            price_type_bau=(
                PriceType.UNIT_PRICE if st.session_state["rpt_bau"]
                else PriceType.TOTAL_PRICE
            ),
            price_type_test=(
                PriceType.UNIT_PRICE if st.session_state["rpt_test"]
                else PriceType.TOTAL_PRICE
            ),
            implied_dollars_bau=st.session_state["rid_bau"],
            implied_dollars_test=st.session_state["rid_test"],
        )
        bau_parquet = _dataset_parquet(complete, rep_bau, workspace_root)
        test_parquet = _dataset_parquet(complete, rep_test, workspace_root)
        with st.spinner("Generating report..."):
            report = generate_reports(
                bau_parquet, test_parquet, cfg, st.session_state["report_dir"]
            )
        st.session_state["report_result"] = report
        st.success(
            f"Report written in {report.elapsed_seconds}s to the output directory."
        )

    report = st.session_state.get("report_result")
    if report is not None:
        st.subheader("Item-level metrics")
        st.dataframe(report.metrics.to_pandas(), use_container_width=True)
        col_top, col_bot = st.columns(2)
        with col_top:
            st.caption("Top 5 by dollar difference")
            st.dataframe(report.top_5_sales.to_pandas(), use_container_width=True)
            st.caption("Top 5 by units difference")
            st.dataframe(report.top_5_units.to_pandas(), use_container_width=True)
        with col_bot:
            st.caption("Bottom 5 by dollar difference")
            st.dataframe(report.bottom_5_sales.to_pandas(), use_container_width=True)
            st.caption("Bottom 5 by units difference")
            st.dataframe(report.bottom_5_units.to_pandas(), use_container_width=True)
        st.subheader("Artifacts")
        for name, path in report.artifacts.items():
            col_a, col_b = st.columns([3, 1])
            col_a.write(f"{name}: `{path}`")
            col_b.download_button(
                "Download",
                path.read_bytes(),
                file_name=path.name,
                key=f"dl_{name}",
            )
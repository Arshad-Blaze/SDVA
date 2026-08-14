# DVA User Guide

**Data Validation Application** — an end-to-end tool that moves files from an
MFT/SFTP source into clean Parquet datasets, then validates those datasets
(store list, sales, UPC) and produces comparison reports.

- Web UI: **Streamlit**
- Processing engine: **Polars** (Parquet)
- Default workspace: `dva/data/` (`raw/`, `datasets/`, registry, `reports/`)

---

## 1. Install and run

```bash
cd dva
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open the printed URL (default `http://localhost:8501`).

Requirements: Python ≥ 3.12, Polars ≥ 1.0, Streamlit ≥ 1.30, paramiko
(SFTP), pyarrow.

### Quick start (local mode, no MFT server)

1. **MFT connection** (right panel): set **Transport = `local`**, point
   **Local share root** at a folder containing your test files (the first
   field defaults to the app folder; set `source_path` to the sub-folder,
   e.g. `/inbound/sales`) and leave `File pattern` at `*.csv`.
2. Switch to **Tab 1 · Pipeline** → click **Discover files** → **Run pipeline**.
3. Files that need a decision land in **Tab 2 · Approval**; approve them.
4. Finished datasets appear in **Tab 3 · Datasets**.
5. Pick BAU and TEST datasets in **Tab 4 · Validator**, then **Tab 5 · Reports**.

### SFTP mode

Choose **Transport = `sftp`** and fill in host, username, password and port.
Downloads are written to `<workspace>/raw/<file>.part` and only renamed to the
final name after size/checksum verification.

---

## 2. The five tabs

### MFT connection (left panel, always visible)

| Setting | Purpose |
|---|---|
| **Workspace root** | Where `raw/` and `datasets/` live (`dva/data` default) |
| **Transport** | `local` (a folder stands in for the MFT server) or `sftp` |
| **Remote source path** | Directory on the MFT/local share to scan |
| **File pattern** | e.g. `*.csv`, `*.txt`, `*.dat` |
| **Fixed-width layout CSV** | Optional two-column CSV describing fixed-width columns (see §4) |
| **Reconnect** | Rebuild the pipeline with the current settings |

### Tab 1 · Pipeline

- **Discover files** — lists new files on the source and queues them.
- **Run pipeline** — processes every queueable file through detect → parse →
  Parquet → verify → cleanup; per-file statuses are shown in the table.
  Gzip- and zip-compressed sources are expanded automatically before
  detection (the archive itself is left on the MFT).
- **Recovery / maintenance**
  - *Re-queue in-flight*: after a restart, files stuck mid-flight are re-queued.
  - *Retry all failed*: re-queue every failed file.
  - *Clean stale .part*: delete orphaned partial downloads.

Statuses you will see, per file:

```
DISCOVERED → DOWNLOAD_QUEUED → DOWNLOADING → DOWNLOAD_VERIFIED → DETECTING
→ AWAITING_APPROVAL → PARSING → PARQUET_WRITING → DATASET_VERIFYING
→ COMPLETE → RAW_DELETED
```

Failures: `DOWNLOAD_FAILED`, `DECOMPRESSION_FAILED` (corrupt/empty gzip or
zip), `DETECTION_FAILED`, `PARSE_FAILED`, `WRITE_FAILED`,
`VERIFICATION_FAILED`, `CLEANUP_FAILED`. Failed files can be retried one by
one or all at once; a retried file is genuinely reprocessed (the retry no-op
bug is fixed and regression-tested).

### Tab 2 · Approval

Shown for any file the Detector could not classify with high confidence
(e.g. leading prose lines, mixed layouts, unusual delimiters). You can:

- Verify the **detection summary** (format, structure, delimiter, confidence,
  warnings, delimiter consistency).
- **Modify** the approved configuration:
  - **Delimiter** (e.g. `,`, `|`, `;`, TAB)
  - **Header row present** (on/off)
  - **Lines to skip** — number of leading non-data rows (banner/title/prose)
  - **Schema overrides / fixed-width columns** when applicable
- The configuration is persisted to the registry *outside* the session, so it
  survives restarts.

Actions: **Approve** (parse with these settings), **Reprocess** (re-detect the
file), or **Reject** (mark failed).

### Tab 3 · Datasets

Lists only **COMPLETE** datasets, with their schema (JSON) and the parse
report (issue counts and counters). This is the only place the Validator and
Reports read from — incomplete datasets are never exposed.

### Tab 4 · Validator

Compare a **BAU** (baseline) dataset against a **TEST** dataset:

- Toggle analyses: **Store analysis** (missing stores), **Sales analysis**
  (per-store units/dollars + comparison), **UPC analysis** (missing UPCs).
- Column mappings (use the dataset's own column names):
  - Store, units, price columns for BAU and TEST
  - **Weighted units** column (optional) — per row, the weighted value is used
    when present, otherwise the raw units column.
  - **Units type: `qty` or `weight`** — a column in every result frame records
    what the units count.
- Results are shown as metrics tables + detail frames (per-store totals, the
  full comparison with unit/dollar differences and percentages, and top/bottom
  5 stores).

Money semantics: **Unit price** checkbox means the price column holds a
per-unit price (dollars = units × price); otherwise the price column already
holds a line total. **Implied** checkboxes divide the raw value by 100
(cents / hundredths).

### Tab 5 · Reports (item-level)

Per-`UPC|DESC` comparison between a BAU and TEST dataset:

1. Choose BAU and TEST datasets and their column mappings (UPC, description,
   units, price), plus optional weighted-units columns and units type.
2. **Generate report** — writes four Parquet artifacts **plus an Excel
   workbook** into the output directory (default `<workspace>/reports`):
   - `item_validation_bau_summary.parquet`
   - `item_validation_test_summary.parquet`
   - `item_validation_comparison.parquet`
   - `item_validation_summary.parquet`
   - `item_validation.xlsx` (sheets: BAU Summary, TEST Summary, Comparison,
     Summary)
3. The UI shows the metrics table and top/bottom 5 by dollar and unit
   difference, and lets you download each artifact.

Every artifact carries a **Unit Type** column (`qty`/`weight`).

---

## 3. Business rules recap

| Concept | Meaning |
|---|---|
| **BAU vs TEST** | Baseline dataset vs the dataset under test |
| **Weighted units** | Units where the source reports a second, weighted quantity that should override the raw count per row (used when non-null) |
| **Units type** | Whether the units column counts items (`qty`) or mass (`weight`) |
| **Implied decimals** | Raw values stored as cents/hundredths; divide by 100 before totalling |
| **Unit price** | Price column holds a per-unit price → dollars = units × price |

---

## 4. Fixed-width files

Fixed-width files need a layout. Provide it in the **Fixed-width layout CSV**,
two columns:

```csv
Column,Width
STORE,5
UNITS,4
PRICE,8
```

A detected layout is shown in the Approval tab when applicable, and can be
edited there. Leading banner lines can be skipped with **Lines to skip**.

---

## 5. Troubleshooting

| Symptom | Likely fix |
|---|---|
| File stuck in `AWAITING_APPROVAL` | Open Tab 2 and approve/reprocess it |
| `DOWNLOAD_FAILED` | Check transport settings, host/credentials, or local share path |
| `PARSE_FAILED` | Check delimiter / header / skip-lines in Tab 2, then Retry |
| Wrong column names in Validator | The mappings use the **dataset's** column names, not the source file's header names |
| `VERIFICATION_FAILED` for a row count | The parsed data did not round-trip; check for truncated or malformed records in the parse report |
| Nothing appears in Reports | Only **COMPLETE** datasets are selectable; make sure the file finished in Tab 1 |

## 6. Runtime layout

```
<workspace>/
├── raw/              # transient: <file>.part then <file> (deleted after COMPLETE)
│                     # compressed inputs are expanded to <name>.decompressed temps
├── datasets/<id>/    # part-*.parquet + schema metadata.json
├── registry.json     # file statuses + approved configs — survives restarts
└── reports/          # item-level report artifacts (Parquet + Excel)
```
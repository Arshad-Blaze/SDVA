# DVA Architecture & Workflow

## Final architecture decision

**MFT → File Mover → Temporary Local Raw → Detector → Parser → Local Parquet → Validator → Reports**

The important optimization is **per-file progression**. A batch of 10 files does not wait for all 10 downloads. Each file independently moves through the pipeline after its own download is verified.

## High-level workflow

```text
                           MFT SERVER
                              |
                       File Discovery
                              |
                 +------------+------------+
                 |            |            |
               File 1       File 2       File N
                 |            |            |
             Download     Download     Download
                 |            |            |
          size/integrity size/integrity size/integrity
              verified      verified      verified
                 |            |            |
              Ready          Ready        Ready
                 |            |            |
             Detector      Detector     Detector
                 |            |            |
              Config       Config       Config
                 |            |            |
              Parser       Parser       Parser
                 |            |            |
             Parquet      Parquet      Parquet
                 |            |            |
          verify dataset verify dataset verify dataset
                 |            |            |
            delete raw   delete raw   delete raw
                 |            |            |
                 +------------+------------+
                              |
                         Local Dataset
                              |
                              v
                         VALIDATOR
                              |
                +-------------+-------------+
                |             |             |
             Store         Sales          UPC
            Analysis      Analysis      Analysis
                |             |             |
                +-------------+-------------+
                              |
                           Reports
```

## Per-file lifecycle

```text
DISCOVERED
   ↓
DOWNLOAD_QUEUED
   ↓
DOWNLOADING
   ↓
DOWNLOAD_VERIFIED
   ↓
DETECTING
   ↓
CONFIG_READY
   ↓
USER_APPROVAL (if required)
   ↓
PARSING
   ↓
PARQUET_WRITING
   ↓
DATASET_VERIFYING
   ↓
COMPLETE
   ↓
RAW_DELETED
```

## Concurrency model

```text
                    MFT
                     |
              Download Pool
             /      |       \
          File1   File2    File3
             \      |       /
               Ready Queue
                    |
              Parser Pool
                /       \
             Parser1   Parser2
                \       /
               Parquet
```

Start conservatively:
- Download workers: 2–4
- Parser workers: 1–2
- Tune only after measuring CPU, RAM, disk I/O and MFT throughput.

## Temporary vs persistent storage

```text
workspace/
├── raw/
│   ├── file1.ready
│   └── file2.part
│
├── datasets/
│   └── dataset_id/
│       ├── part-000.parquet
│       ├── part-001.parquet
│       └── metadata.json
│
├── configs/
└── reports/
```

`*.part` is never parsed. Raw files are deleted after successful Parquet verification.

## Critical invariants

1. MFT is source of truth.
2. Local raw is transient.
3. Parquet is the persistent working artifact.
4. Parser never performs business validation.
5. Validator never parses raw retailer formats.
6. Streamlit never owns the core processing logic.
7. Only COMPLETE datasets are visible to Validator.
8. One failed file does not invalidate unrelated completed files.

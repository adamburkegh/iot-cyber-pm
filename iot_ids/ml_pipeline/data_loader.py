#Load IoT-23 .labeled.csv and .log.labeled into DataFrames
import os
from pathlib import Path

import pandas as pd

from .zeek_parser import zeek_to_df

LABEL_BENIGN = "Benign"
LABEL_MALICIOUS = "Malicious"
VALID_LABELS = (LABEL_BENIGN, LABEL_MALICIOUS)


def normalize_label(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    lower = s.lower()
    if lower == LABEL_BENIGN.lower():
        return LABEL_BENIGN
    if lower == LABEL_MALICIOUS.lower():
        return LABEL_MALICIOUS
    return None


def discover_files(data_dir: str, include_zeek_logs: bool = True) -> list[str]:
    data_path = Path(data_dir)
    if not data_path.is_dir():
        return []
    csv_paths = sorted(
        str(f.resolve())
        for f in data_path.iterdir()
        if f.is_file() and f.name.lower().endswith(".labeled.csv")
    )
    zeek_paths = []
    if include_zeek_logs:
        zeek_paths = sorted(
            str(f.resolve())
            for f in data_path.iterdir()
            if f.is_file()
            and f.name.lower().endswith(".log.labeled")
            and not f.name.lower().endswith(".csv")
        )
    csv_set = set(csv_paths)
    sources = list(csv_paths)
    for zp in zeek_paths:
        if (zp + ".csv") not in csv_set:
            sources.append(zp)
    return sorted(sources)


def load_zeek_file(filepath: str, filename: str) -> tuple[pd.DataFrame, dict]:
    df = zeek_to_df(filepath)
    df["source_file"] = filename
    if "label" in df.columns:
        df["parsed_label"] = df["label"].apply(normalize_label)
    else:
        df["parsed_label"] = None
    parsed_ok = df["parsed_label"].notna().sum()
    label_dist = (
        df["parsed_label"].value_counts().to_dict()
        if df["parsed_label"].notna().any()
        else {}
    )
    label_dist = {k: int(v) for k, v in label_dist.items() if k is not None}
    return df, {
        "filename": filename,
        "rows": len(df),
        "label_distribution": label_dist,
        "parsing_errors": len(df) - int(parsed_ok),
        "is_zeek": True,
    }


def load_file(filepath: str) -> tuple[pd.DataFrame, dict]:
    filename = os.path.basename(filepath)
    fp_lower = filepath.lower()
    if fp_lower.endswith(".log.labeled") and not fp_lower.endswith(".csv"):
        return load_zeek_file(filepath, filename)
    df = pd.read_csv(filepath, low_memory=False)
    df["source_file"] = filename
    if "label" not in df.columns:
        return df, {
            "filename": filename,
            "rows": len(df),
            "label_distribution": {},
            "parsing_errors": len(df),
            "is_zeek": False,
        }
    df["parsed_label"] = df["label"].apply(normalize_label)
    parsed_ok = df["parsed_label"].notna().sum()
    label_dist = {k: int(v) for k, v in df["parsed_label"].value_counts().to_dict().items() if k is not None}
    return df, {
        "filename": filename,
        "rows": len(df),
        "label_distribution": label_dist,
        "parsing_errors": len(df) - int(parsed_ok),
        "is_zeek": False,
    }


def load_zeek_folder(
    data_dir: str,
    required_cols: list[str] | None = None,
) -> pd.DataFrame:
    if required_cols is None:
        required_cols = ["uid", "label", "source_file"]
    files = discover_files(data_dir)
    if not files:
        return pd.DataFrame()
    frames = []
    for fp in files:
        df_file, _ = load_file(fp)
        frames.append(df_file)
    combined = pd.concat(frames, ignore_index=True)
    if "parsed_label" in combined.columns:
        combined = combined[combined["parsed_label"].notna()].copy()
        combined["label"] = combined["parsed_label"]
        combined = combined.drop(columns=["parsed_label"], errors="ignore")
    missing = [c for c in required_cols if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in data from {data_dir!r}: {missing}")
    return combined


def format_file_log_line(meta: dict) -> str:
    filename, rows = meta["filename"], meta["rows"]
    parsed_ok = rows - meta.get("parsing_errors", 0)
    if meta.get("is_zeek"):
        return f"  {filename}: {rows:,} rows (Zeek) -> {parsed_ok:,} labels parsed"
    if meta.get("parsing_errors", 0) == rows:
        return f"  {filename}: {rows} rows (no label column — cannot assign class)"
    return f"  {filename}: {rows:,} rows -> {parsed_ok:,} labels (from label column)"


def load_iot23_dataset(data_dir: str) -> tuple[pd.DataFrame, dict]:
    log: list[str] = []
    log.append("Discovering IoT-23 dataset files...")
    files = discover_files(data_dir)
    log.append(f"  Found {len(files)} files in {data_dir}/")
    if not files:
        out_meta = {
            "total_files": 0,
            "total_rows": 0,
            "files_processed": [],
            "label_distribution": {},
        }
        out_meta["processing_log"] = log
        return pd.DataFrame(), out_meta
    log.append("Loading and parsing datasets...")
    frames = []
    for fp in files:
        df_file, meta = load_file(fp)
        frames.append(df_file)
        log.append(format_file_log_line(meta))
    combined = pd.concat(frames, ignore_index=True)
    if "parsed_label" in combined.columns:
        combined = combined[combined["parsed_label"].notna()].copy()
        combined["label"] = combined["parsed_label"]
        combined = combined.drop(columns=["parsed_label"], errors="ignore")
    total_rows = len(combined)
    label_dist = combined["label"].value_counts().to_dict() if "label" in combined.columns else {}
    log.append("Combining datasets...")
    log.append(f"  Total rows: {total_rows:,}")
    log.append(f"  Label distribution: {label_dist}")
    log.append("  Added source_file column for provenance tracking")
    out_meta = {
        "total_files": len(files),
        "total_rows": total_rows,
        "files_processed": [os.path.basename(f) for f in files],
        "label_distribution": label_dist,
    }
    out_meta["processing_log"] = log
    return combined, out_meta

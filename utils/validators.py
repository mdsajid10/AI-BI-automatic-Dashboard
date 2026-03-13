"""
Data validation utilities for AI BI Dashboard Generator.
Validates uploaded files for format, size, and data quality.
"""

import pandas as pd
from typing import Tuple, Optional

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

MAX_FILE_SIZE_MB = 200
ALLOWED_EXTENSIONS = [".csv", ".xlsx", ".xls"]
MAX_ROWS = 1_000_000
MAX_COLUMNS = 500


# ──────────────────────────────────────────────
# File Validation
# ──────────────────────────────────────────────

def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """Validate that the uploaded file has an allowed extension."""
    if not filename:
        return False, "No file name provided."
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
    return True, "File extension is valid."


def validate_file_size(file_size_bytes: int) -> Tuple[bool, str]:
    """Validate that the file size is within limits."""
    size_mb = file_size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File size ({size_mb:.1f} MB) exceeds the maximum allowed size ({MAX_FILE_SIZE_MB} MB)."
    return True, f"File size ({size_mb:.1f} MB) is within limits."


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str, dict]:
    """
    Validate a loaded DataFrame for quality issues.
    Returns (is_valid, message, quality_report).
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "empty_columns": [],
        "high_missing_columns": [],
        "low_variance_columns": [],
        "warnings": [],
    }

    # Check row/column limits
    if len(df) == 0:
        return False, "The uploaded file contains no data rows.", report
    if len(df) > MAX_ROWS:
        return False, f"Dataset has {len(df):,} rows, exceeding the limit of {MAX_ROWS:,}.", report
    if len(df.columns) > MAX_COLUMNS:
        return False, f"Dataset has {len(df.columns)} columns, exceeding the limit of {MAX_COLUMNS}.", report

    # Check for completely empty columns
    for col in df.columns:
        if df[col].isna().all():
            report["empty_columns"].append(col)

    # Check for high missing value ratio
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        if missing_pct > 50:
            report["high_missing_columns"].append({"column": col, "missing_pct": round(missing_pct, 1)})

    # Build warnings
    if report["empty_columns"]:
        report["warnings"].append(
            f"{len(report['empty_columns'])} column(s) are completely empty: {', '.join(report['empty_columns'][:5])}"
        )
    if report["high_missing_columns"]:
        cols_info = ", ".join(
            [f"{c['column']} ({c['missing_pct']}%)" for c in report["high_missing_columns"][:5]]
        )
        report["warnings"].append(f"High missing values in: {cols_info}")
    if report["duplicate_rows"] > 0:
        dup_pct = report["duplicate_rows"] / len(df) * 100
        report["warnings"].append(
            f"{report['duplicate_rows']:,} duplicate rows detected ({dup_pct:.1f}% of data)."
        )

    is_valid = len(report["empty_columns"]) < len(df.columns)  # At least some columns have data
    msg = "Dataset validated successfully." if is_valid else "Dataset has critical quality issues."
    if report["warnings"]:
        msg += " Warnings: " + " | ".join(report["warnings"])

    return is_valid, msg, report


def validate_upload(filename: str, file_size_bytes: int) -> Tuple[bool, str]:
    """Run all pre-load validations on an uploaded file."""
    valid, msg = validate_file_extension(filename)
    if not valid:
        return False, msg

    valid, msg = validate_file_size(file_size_bytes)
    if not valid:
        return False, msg

    return True, "File passed pre-load validation."

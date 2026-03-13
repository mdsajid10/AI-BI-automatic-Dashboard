"""
Data Loader Module — AI BI Dashboard Generator.
Handles CSV / Excel file loading with validation and type inference.
"""

import pandas as pd
import streamlit as st
from typing import Tuple, Optional

from utils.validators import validate_upload, validate_dataframe
from utils.helpers import classify_columns, safe_convert_datetime


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str, Optional[dict]]:
    """
    Load an uploaded file into a validated DataFrame.

    Returns:
        (df, message, quality_report)
        - df is None when validation fails.
    """
    # Pre-load validation (extension + size)
    valid, msg = validate_upload(uploaded_file.name, uploaded_file.size)
    if not valid:
        return None, msg, None

    # Read the file
    try:
        df = _read_file(uploaded_file)
    except Exception as e:
        return None, f"Failed to read file: {e}", None

    # Post-load quality validation
    valid, msg, report = validate_dataframe(df)
    if not valid:
        return None, msg, report

    # Infer & convert datetime columns
    col_types = classify_columns(df)
    df = safe_convert_datetime(df, col_types["datetime"])

    return df, msg, report


def get_column_types(df: pd.DataFrame) -> dict:
    """Return classified column types for a DataFrame."""
    return classify_columns(df)


# ──────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────

def _read_file(uploaded_file) -> pd.DataFrame:
    """Read CSV or Excel into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {name}")

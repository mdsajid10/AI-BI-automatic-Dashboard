"""
Interactive Filters Module — AI BI Dashboard Generator.
Builds Streamlit sidebar filters (category, date range, numeric sliders)
and returns a filtered copy of the DataFrame.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional


def render_sidebar_filters(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    """
    Build sidebar filter widgets and return a filtered DataFrame.

    Supports:
    - Categorical multi-select filters
    - Date range pickers
    - Numeric range sliders
    """
    filtered = df.copy()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Filters")

    # ── Categorical Filters ─────────────────────
    cat_cols = col_types.get("categorical", [])
    for col in cat_cols[:5]:  # limit to 5 to avoid sidebar overflow
        unique_vals = sorted(df[col].dropna().unique().astype(str))
        if 1 < len(unique_vals) <= 50:
            selected = st.sidebar.multiselect(
                f"📋 {col}",
                options=unique_vals,
                default=[],
                key=f"filter_cat_{col}",
            )
            if selected:
                filtered = filtered[filtered[col].astype(str).isin(selected)]

    # ── Date Range Filters ──────────────────────
    dt_cols = col_types.get("datetime", [])
    for col in dt_cols[:2]:
        try:
            date_series = pd.to_datetime(filtered[col], errors="coerce").dropna()
            if len(date_series) == 0:
                continue
            min_date = date_series.min().date()
            max_date = date_series.max().date()

            if min_date < max_date:
                date_range = st.sidebar.date_input(
                    f"📅 {col}",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"filter_dt_{col}",
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start, end = date_range
                    mask = (
                        pd.to_datetime(filtered[col], errors="coerce").dt.date >= start
                    ) & (
                        pd.to_datetime(filtered[col], errors="coerce").dt.date <= end
                    )
                    filtered = filtered[mask]
        except Exception:
            pass

    # ── Numeric Range Filters ───────────────────
    num_cols = col_types.get("numerical", [])
    for col in num_cols[:4]:  # limit to 4
        series = df[col].dropna()
        if len(series) == 0:
            continue

        col_min = float(series.min())
        col_max = float(series.max())

        if col_min >= col_max:
            continue

        # Determine step
        value_range = col_max - col_min
        if value_range > 1000:
            step = round(value_range / 100, 0)
        elif value_range > 10:
            step = round(value_range / 100, 1)
        else:
            step = round(value_range / 100, 2)
        step = max(step, 0.01)

        selected_range = st.sidebar.slider(
            f"📊 {col}",
            min_value=col_min,
            max_value=col_max,
            value=(col_min, col_max),
            step=step,
            key=f"filter_num_{col}",
        )
        if selected_range != (col_min, col_max):
            filtered = filtered[
                (filtered[col] >= selected_range[0]) & (filtered[col] <= selected_range[1])
            ]

    # ── Filter Summary ──────────────────────────
    original_count = len(df)
    filtered_count = len(filtered)
    if filtered_count < original_count:
        reduction = (1 - filtered_count / original_count) * 100
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"**Showing** {filtered_count:,} / {original_count:,} records "
            f"({reduction:.1f}% filtered)"
        )

    return filtered

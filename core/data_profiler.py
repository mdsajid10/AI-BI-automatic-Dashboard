"""
Data Profiler Module — AI BI Dashboard Generator.
Performs automatic data profiling: column types, statistics, missing values,
cardinality, and generates a structured profile report.
"""

import pandas as pd
import numpy as np
from typing import Any

from utils.helpers import classify_columns, format_number, format_percentage


class DataProfiler:
    """Generates a comprehensive data profile from a DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.col_types = classify_columns(df)

    def generate_profile(self) -> dict:
        """Build the full profile report dictionary."""
        profile = {
            "overview": self._overview(),
            "columns": self._column_profiles(),
            "missing_summary": self._missing_summary(),
            "col_types": self.col_types,
        }
        return profile

    # ── Overview ────────────────────────────────

    def _overview(self) -> dict:
        df = self.df
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numerical_count": len(self.col_types["numerical"]),
            "categorical_count": len(self.col_types["categorical"]),
            "datetime_count": len(self.col_types["datetime"]),
            "total_missing": int(df.isna().sum().sum()),
            "missing_pct": round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        }

    # ── Per-column profiles ─────────────────────

    def _column_profiles(self) -> list:
        profiles = []
        for col in self.df.columns:
            series = self.df[col]
            p = {
                "name": col,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "missing_pct": round(series.isna().sum() / len(self.df) * 100, 1) if len(self.df) > 0 else 0,
                "unique": int(series.nunique()),
                "category": self._get_col_category(col),
            }

            if col in self.col_types["numerical"]:
                desc = series.describe()
                p.update({
                    "mean": round(float(desc["mean"]), 2) if pd.notna(desc["mean"]) else None,
                    "std": round(float(desc["std"]), 2) if pd.notna(desc["std"]) else None,
                    "min": float(desc["min"]) if pd.notna(desc["min"]) else None,
                    "max": float(desc["max"]) if pd.notna(desc["max"]) else None,
                    "median": round(float(series.median()), 2) if pd.notna(series.median()) else None,
                    "q1": round(float(desc["25%"]), 2) if pd.notna(desc["25%"]) else None,
                    "q3": round(float(desc["75%"]), 2) if pd.notna(desc["75%"]) else None,
                    "skew": round(float(series.skew()), 2) if pd.notna(series.skew()) else None,
                })

            elif col in self.col_types["categorical"]:
                top = series.value_counts().head(5)
                p["top_values"] = {str(k): int(v) for k, v in top.items()}

            elif col in self.col_types["datetime"]:
                valid = pd.to_datetime(series, errors="coerce").dropna()
                if len(valid) > 0:
                    p["date_min"] = str(valid.min().date())
                    p["date_max"] = str(valid.max().date())
                    p["date_range_days"] = (valid.max() - valid.min()).days

            profiles.append(p)
        return profiles

    def _get_col_category(self, col: str) -> str:
        if col in self.col_types["numerical"]:
            return "numerical"
        if col in self.col_types["categorical"]:
            return "categorical"
        if col in self.col_types["datetime"]:
            return "datetime"
        return "unknown"

    # ── Missing summary ─────────────────────────

    def _missing_summary(self) -> list:
        missing = self.df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        return [
            {
                "column": col,
                "count": int(count),
                "pct": round(count / len(self.df) * 100, 1),
            }
            for col, count in missing.items()
        ]

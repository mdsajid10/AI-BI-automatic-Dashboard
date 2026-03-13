"""
Utility helpers for AI BI Dashboard Generator.
Common formatting, color palettes, and layout helpers.
"""

import pandas as pd
import numpy as np
from typing import Any


# ──────────────────────────────────────────────
# Color Palettes
# ──────────────────────────────────────────────

PALETTE_MODERN = [
    "#6C5CE7", "#00CEC9", "#FD79A8", "#FDCB6E",
    "#0984E3", "#E17055", "#00B894", "#A29BFE",
    "#74B9FF", "#FF7675", "#55EFC4", "#DFE6E9",
]

PALETTE_DARK = [
    "#BB86FC", "#03DAC6", "#CF6679", "#FFB74D",
    "#64B5F6", "#81C784", "#F06292", "#BA68C8",
    "#4DD0E1", "#FFD54F", "#AED581", "#90A4AE",
]

KPI_COLORS = {
    "primary": "#6C5CE7",
    "success": "#00B894",
    "warning": "#FDCB6E",
    "danger": "#E17055",
    "info": "#0984E3",
}


# ──────────────────────────────────────────────
# Formatting Helpers
# ──────────────────────────────────────────────

def format_number(value: float, precision: int = 2) -> str:
    """Format a number with K/M/B suffixes for readability."""
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value / 1_000_000_000:.{precision}f}B"
    elif abs_val >= 1_000_000:
        return f"{value / 1_000_000:.{precision}f}M"
    elif abs_val >= 1_000:
        return f"{value / 1_000:.{precision}f}K"
    elif isinstance(value, float):
        return f"{value:,.{precision}f}"
    else:
        return f"{value:,}"


def format_percentage(value: float, precision: int = 1) -> str:
    """Format a number as a percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{precision}f}%"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# ──────────────────────────────────────────────
# Data Helpers
# ──────────────────────────────────────────────

def classify_columns(df: pd.DataFrame) -> dict:
    """Classify dataframe columns into numerical, categorical, and datetime."""
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()

    # Try to detect datetime columns stored as strings
    for col in categorical[:]:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() / len(df) > 0.7:
                datetime_cols.append(col)
                categorical.remove(col)
        except Exception:
            pass

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_cols,
    }


def safe_convert_datetime(df: pd.DataFrame, datetime_cols: list) -> pd.DataFrame:
    """Convert detected datetime columns to proper datetime type."""
    df = df.copy()
    for col in datetime_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except Exception:
                pass
    return df


def get_top_categories(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Get top N categories from a series, grouping the rest as 'Other'."""
    counts = series.value_counts()
    if len(counts) <= top_n:
        return counts
    top = counts.head(top_n)
    other_count = counts.iloc[top_n:].sum()
    top["Other"] = other_count
    return top


# ──────────────────────────────────────────────
# CSS Helpers for Streamlit
# ──────────────────────────────────────────────

def inject_custom_css(theme: str = "Dark"):
    """Return custom CSS for dashboard styling based on the selected theme."""
    is_light = theme == "Light"
    
    # Theme Variables
    bg_color = "#ffffff" if is_light else "#0f0f1a"
    card_bg = "linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%)" if is_light else "linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%)"
    text_color = "#1e1e2e" if is_light else "#e0e0e8"
    sub_text = "#636e72" if is_light else "#a0a0b8"
    border = "rgba(108, 92, 231, 0.1)" if is_light else "rgba(108, 92, 231, 0.2)"
    sidebar_bg = "linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%)" if is_light else "linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%)"
    insight_bg = "#f8f9fa" if is_light else "#1a1a2e"
    header_color = "#1e1e2e" if is_light else "#e0e0e8"
    input_bg = "#ffffff" if is_light else "#262730"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Styles & Variables */
        .stApp {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
        }}
        
        /* Force Main Background */
        [data-testid="stAppViewContainer"], .main, .stApp {{
            background-color: {bg_color} !important;
        }}
        
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}

        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 1rem;
            max-width: 95%;
        }}

        /* KPI Card Styles */
        .kpi-card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid {border};
            box-shadow: 0 4px 15px rgba(0, 0, 0, {0.03 if is_light else 0.15});
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(108, 92, 231, {0.1 if is_light else 0.25});
        }}
        .kpi-label {{
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            font-weight: 500;
            color: {sub_text};
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }}
        .kpi-value {{
            font-family: 'Inter', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6C5CE7, #a29bfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }}
        .kpi-subtitle {{
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            color: {sub_text};
        }}

        /* Section Headers */
        .section-header {{
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: {header_color};
            padding: 0.5rem 0;
            border-bottom: 2px solid rgba(108, 92, 231, 0.3);
            margin-bottom: 1rem;
            margin-top: 1.5rem;
        }}

        /* Insight Card */
        .insight-card {{
            background: {insight_bg};
            border-left: 4px solid #6C5CE7;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin: 0.5rem 0;
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: {text_color};
            border: 1px solid {border};
        }}
        
        /* Inputs & Widgets Aggressive Styling */
        /* Inputs & Widgets Aggressive Styling */
        div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="popover"] > div,
        div[data-baseweb="base-input"] {{
            background-color: {input_bg} !important;
            border-radius: 8px !important;
            color: {text_color} !important;
            border: 1px solid {border} !important;
        }}
        
        /* File Uploader Styling */
        [data-testid="stFileUploader"] {{
            background-color: {input_bg} !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            border: 1px dashed #6C5CE7 !important;
        }}
        
        [data-testid="stFileUploaderDropzone"] {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        [data-testid="stFileUploaderDropzone"] p {{
            color: {text_color} !important;
        }}

        /* File Uploader inner button styling */
        [data-testid="stFileUploader"] button {{
            background-color: { "#ffffff" if is_light else "#262730" } !important;
            color: { "#1e1e2e" if is_light else "#e0e0e8" } !important;
            border: 1px solid #6C5CE7 !important;
            border-radius: 8px !important;
        }}
        
        [data-testid="stFileUploader"] button:hover {{
            background-color: #6C5CE7 !important;
            color: white !important;
            border-color: #6C5CE7 !important;
        }}

        /* Button Styling */
        div.stButton > button, div.stDownloadButton > button {{
            background-color: { "#ffffff" if is_light else "#262730" } !important;
            color: { "#1e1e2e" if is_light else "#e0e0e8" } !important;
            border: 1px solid #6C5CE7 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.3s ease !important;
        }}
        
        div.stButton > button:hover, div.stDownloadButton > button:hover {{
            background-color: #6C5CE7 !important;
            color: white !important;
            box-shadow: 0 4px 12px rgba(108, 92, 231, 0.3) !important;
        }}

        /* DataFrame / Data Table Styling */
        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        .stDataFrame div[data-testid="stDataFrameResizable"] {{
            background-color: {input_bg} !important;
        }}

        /* Specific Input Text Contrast */
        input, textarea, select {{
            color: {text_color} !important;
        }}
        
        /* Dropdown Popover / Options List Styling */
        div[data-baseweb="popover"] > div,
        ul[role="listbox"],
        li[role="option"] {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        
        li[role="option"]:hover {{
            background-color: rgba(108, 92, 231, 0.1) !important;
        }}

        /* Fix for number inputs and other containers */
        div.stNumberInput, div.stTextInput, div.stSelectbox, div.stMultiSelect {{
            background-color: transparent !important;
        }}
        
        /* Metric Card Text Contrast */
        [data-testid="stMetricLabel"] {{
            color: {sub_text} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {text_color} !important;
        }}
        [data-testid="stMetricDelta"] {{
            color: {text_color} !important;
        }}
        
        /* Universal Text Contrast */
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: {text_color} !important;
        }}

        /* Data Table Styling */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            background-color: {input_bg} !important;
            border-radius: 8px !important;
            border: 1px solid {border} !important;
        }}
        
        /* Pagination and Search Styling */
        [data-testid="stDataFramePagination"] span {{
            color: {text_color} !important;
        }}
        
        .dataframe {{
            font-family: 'Inter', sans-serif !important;
            font-size: 0.85rem !important;
            color: {text_color} !important;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: {sidebar_bg};
            border-right: 1px solid {border};
        }}
        
        [data-testid="stSidebar"] .stMarkdown p {{
            color: {text_color} !important;
        }}

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background-color: transparent !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 8px 16px;
            color: {text_color} !important;
            background-color: transparent !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {border} !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: rgba(108, 92, 231, 0.1) !important;
            font-weight: 600 !important;
            color: #6C5CE7 !important;
            border-bottom: 2px solid #6C5CE7 !important;
        }}
    </style>
    """


def render_kpi_card(label: str, value: str, subtitle: str = "", color: str = "") -> str:
    """Render an HTML KPI card for Streamlit."""
    color_style = ""
    if color:
        color_style = f"background: linear-gradient(135deg, {color}, {color}88);"
        value_style = f"background: linear-gradient(135deg, {color}, {color}cc); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    else:
        value_style = ""

    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="{value_style}">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """


def render_insight_card(text: str) -> str:
    """Render an HTML insight card."""
    return f'<div class="insight-card">💡 {text}</div>'

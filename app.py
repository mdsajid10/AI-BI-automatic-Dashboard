"""
AI BI Dashboard Generator — Main Application.
A professional AI-powered analytics platform built with Streamlit.
Supports dataset upload, auto-dashboard, NL queries, and export.
"""

import os
import sys
import io
import streamlit as st
import pandas as pd
import numpy as np

# ── Ensure project root is on sys.path ──────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from utils.helpers import inject_custom_css, render_kpi_card, render_insight_card, format_number
from core.data_loader import load_file, get_column_types
from core.data_profiler import DataProfiler
from core.chart_generator import ChartGenerator
from core.nl_query_engine import NLQueryEngine
from core.insight_engine import InsightEngine
from dashboard.filters import render_sidebar_filters
from dashboard.dashboard_builder import DashboardBuilder

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI BI Dashboard Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS injection moved inside main to support dynamic themes


# ──────────────────────────────────────────────
# Sidebar — File Upload & Info
# ──────────────────────────────────────────────

def render_sidebar():
    """Render sidebar with upload widget and dataset info."""
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding:1rem 0;">
            <h2 style="font-family:'Inter',sans-serif; background:linear-gradient(135deg,#6C5CE7,#00CEC9);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
            📊 AI BI Dashboard</h2>
            <p style="color:#888; font-size:0.8rem; margin-top:0.25rem;">
            Professional Analytics Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file (max 200 MB)",
    )

    # Theme toggle
    st.sidebar.markdown("---")
    theme = st.sidebar.selectbox("🎨 Theme", ["Dark", "Light"], index=0, key="theme_select")

    return uploaded_file, theme


# ──────────────────────────────────────────────
# Tab 1: Dashboard
# ──────────────────────────────────────────────

def tab_dashboard(df: pd.DataFrame, col_types: dict, theme: str):
    """Auto-generated BI dashboard with filters."""
    # Apply sidebar filters
    filtered_df = render_sidebar_filters(df, col_types)

    # Build dashboard
    builder = DashboardBuilder(filtered_df, col_types, theme=theme)
    builder.render()


# ──────────────────────────────────────────────
# Tab 2: Data Explorer
# ──────────────────────────────────────────────

def tab_data_explorer(df: pd.DataFrame, col_types: dict):
    """Data exploration with table, stats, search, and sort."""
    st.markdown('<div class="section-header">🔍 Data Explorer</div>', unsafe_allow_html=True)

    # Search
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_term = st.text_input("🔎 Search across all columns", "", key="explorer_search")
    with col2:
        sort_col = st.selectbox("Sort by", ["(none)"] + list(df.columns), key="explorer_sort")
    with col3:
        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True, key="explorer_order")

    display_df = df.copy()

    # Apply search
    if search_term:
        mask = pd.Series([False] * len(display_df), index=display_df.index)
        for col in display_df.columns:
            mask |= display_df[col].astype(str).str.contains(search_term, case=False, na=False)
        display_df = display_df[mask]

    # Apply sort
    if sort_col != "(none)":
        ascending = sort_order == "Ascending"
        display_df = display_df.sort_values(sort_col, ascending=ascending)

    # Pagination
    page_size = st.select_slider("Rows per page", options=[10, 25, 50, 100, 250], value=25, key="page_size")
    total_pages = max(1, (len(display_df) - 1) // page_size + 1)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="page_num")
    start = (page - 1) * page_size
    end = start + page_size

    st.dataframe(display_df.iloc[start:end], use_container_width=True, height=500)
    st.caption(f"Showing rows {start + 1}–{min(end, len(display_df))} of {len(display_df):,}")

    # Column statistics
    st.markdown('<div class="section-header">📈 Column Statistics</div>', unsafe_allow_html=True)
    profiler = DataProfiler(df)
    profile = profiler.generate_profile()

    stat_col = st.selectbox("Select column for details", df.columns.tolist(), key="stat_col")
    col_profile = next((cp for cp in profile["columns"] if cp["name"] == stat_col), None)

    if col_profile:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Type", col_profile["category"].title())
            st.metric("Dtype", col_profile["dtype"])
        with c2:
            st.metric("Missing", f"{col_profile['missing']} ({col_profile['missing_pct']}%)")
            st.metric("Unique Values", col_profile["unique"])
        with c3:
            if col_profile["category"] == "numerical":
                st.metric("Mean", format_number(col_profile.get("mean", 0)))
                st.metric("Std Dev", format_number(col_profile.get("std", 0)))
            elif col_profile["category"] == "categorical" and col_profile.get("top_values"):
                top = list(col_profile["top_values"].items())[:3]
                for val, count in top:
                    st.metric(f"Top: {val}", f"{count:,}")
            elif col_profile["category"] == "datetime":
                st.metric("Min Date", col_profile.get("date_min", "N/A"))
                st.metric("Max Date", col_profile.get("date_max", "N/A"))


# ──────────────────────────────────────────────
# Tab 3: Ask AI
# ──────────────────────────────────────────────

def tab_ask_ai(df: pd.DataFrame, col_types: dict, theme: str):
    """Natural language query interface."""
    st.markdown('<div class="section-header">💬 Ask AI About Your Data</div>', unsafe_allow_html=True)

    engine = NLQueryEngine(df, col_types, theme=theme)

    if not engine.is_available():
        st.info(
            "💡 **No AI API key detected.** Set `GROQ_API_KEY` or `OPENAI_API_KEY` in your `.env` file "
            "for AI-powered analysis. Rule-based analysis is still available."
        )

    # Suggested questions
    st.markdown("**💡 Try asking:**")
    suggestions = _generate_suggestions(col_types)
    selected_question = None
    if suggestions:
        suggestion_cols = st.columns(min(len(suggestions), 4))
        for i, suggestion in enumerate(suggestions[:4]):
            with suggestion_cols[min(i, len(suggestion_cols)-1)]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    selected_question = suggestion
    else:
        st.write("No suggestions available for this dataset.")

    # Query input
    question = st.text_input(
        "Type your question:",
        value=selected_question or "",
        placeholder="e.g., Show revenue by region",
        key="nl_question",
    )

    if question:
        with st.spinner("🔄 Analyzing your question..."):
            result = engine.ask(question)

        if result.get("explanation"):
            st.markdown(
                render_insight_card(f"**Analysis:** {result['explanation']}"),
                unsafe_allow_html=True,
            )

        if result.get("error"):
            st.warning(f"Note: {result['error']}")

        if result.get("figure"):
            st.plotly_chart(result["figure"], use_container_width=True, key="nl_chart")

        if result.get("result_df") is not None and len(result["result_df"]) > 0:
            with st.expander("📋 View result data", expanded=False):
                if theme == "Light":
                    styled_res = result["result_df"].style.set_properties(**{
                        'background-color': '#ffffff',
                        'color': '#111827',
                        'border-color': '#e5e7eb'
                    }).set_table_styles([{
                        'selector': 'th',
                        'props': [('background-color', '#f3f4f6'), ('color', '#111827')]
                    }])
                    st.dataframe(styled_res, use_container_width=True)
                else:
                    st.dataframe(result["result_df"], use_container_width=True)


def _generate_suggestions(col_types: dict) -> list[str]:
    suggestions = []
    cats = col_types.get("categorical", [])
    nums = col_types.get("numerical", [])
    dts = col_types.get("datetime", [])

    if cats and nums:
        suggestions.append(f"Show {nums[0]} by {cats[0]}")
    
    if cats:
        suggestions.append(f"Proportion of {cats[0]}")
        
    if len(nums) >= 2:
        suggestions.append(f"Correlation between {nums[0]} and {nums[1]}")
    
    if dts and nums:
        suggestions.append(f"Trend of {nums[0]} over time")
    
    if nums:
        suggestions.append(f"Distribution of {nums[0]}")
    
    if len(cats) >= 2 and nums:
        suggestions.append(f"Compare {nums[0]} across {cats[0]}")

    return suggestions[:6]


# ──────────────────────────────────────────────
# Tab 4: Export
# ──────────────────────────────────────────────

def tab_export(df: pd.DataFrame, col_types: dict, profile: dict):
    """Export dashboard as PDF, images, or CSV summary."""
    st.markdown('<div class="section-header">📥 Export Dashboard</div>', unsafe_allow_html=True)

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # CSV Summary Export
    with exp_col1:
        st.markdown("#### 📄 CSV Summary")
        st.write("Download dataset summary statistics as CSV.")
        if st.button("Generate CSV Summary", key="export_csv"):
            summary = _build_csv_summary(df, col_types)
            csv_data = summary.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                file_name="dashboard_summary.csv",
                mime="text/csv",
                key="download_csv",
            )

    # Data Export
    with exp_col2:
        st.markdown("#### 📊 Filtered Data")
        st.write("Download the current (filtered) dataset.")
        csv_full = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Data",
            csv_full,
            file_name="filtered_data.csv",
            mime="text/csv",
            key="download_data",
        )

    # Profile Report Export
    with exp_col3:
        st.markdown("#### 📋 Profile Report")
        st.write("Download data profile report as CSV.")
        if st.button("Generate Profile Report", key="export_profile"):
            # Clean up the nested dictionaries before exporting to CSV
            clean_profiles = []
            for col_prof in profile["columns"]:
                clean_p = col_prof.copy()
                if "top_values" in clean_p and isinstance(clean_p["top_values"], dict):
                    clean_p["top_values"] = ", ".join([f"{k}: {v}" for k, v in clean_p["top_values"].items()])
                clean_profiles.append(clean_p)
                
            profile_df = pd.DataFrame(clean_profiles)
            csv_profile = profile_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Profile",
                csv_profile,
                file_name="data_profile.csv",
                mime="text/csv",
                key="download_profile",
            )

    # PDF Report
    st.markdown("---")
    st.markdown("#### 📕 PDF Report")
    st.write("Generate a comprehensive PDF report of the dataset analysis.")
    if st.button("📕 Generate PDF Report", key="export_pdf"):
        with st.spinner("Generating PDF..."):
            pdf_bytes = _generate_pdf_report(df, col_types, profile)
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF Report",
                    pdf_bytes,
                    file_name="dashboard_report.pdf",
                    mime="application/pdf",
                    key="download_pdf",
                )
            else:
                st.error("Failed to generate PDF. Please check that fpdf2 is installed.")


def _build_csv_summary(df: pd.DataFrame, col_types: dict) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        row = {"Column": col, "Type": str(df[col].dtype), "Missing": df[col].isna().sum()}
        if col in col_types["numerical"]:
            row["Mean"] = round(df[col].mean(), 2)
            row["Median"] = round(df[col].median(), 2)
            row["Std"] = round(df[col].std(), 2)
            row["Min"] = df[col].min()
            row["Max"] = df[col].max()
        elif col in col_types["categorical"]:
            row["Unique"] = df[col].nunique()
            row["Top Value"] = str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def _clean_text_for_pdf(text: str) -> str:
    """Replace common unicode characters not supported by basic FPDF fonts."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '–': '-',  # en dash (U+2013)
        '—': '-',  # em dash (U+2014)
        '“': '"',  # left double quote
        '”': '"',  # right double quote
        '‘': "'",  # left single quote
        '’': "'",  # right single quote
        '…': '...',
        '€': 'EUR ',
        '£': 'GBP ',
        '©': '(c)',
        '®': '(R)',
        '™': '(TM)'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Final fallback: encode to ascii and replace missing chars, then decode back
    return text.encode('ascii', 'replace').decode('ascii')


def _generate_pdf_report(df: pd.DataFrame, col_types: dict, profile: dict) -> bytes:
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "AI BI Dashboard Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated from dataset with {len(df):,} records", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)

    # Overview
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Dataset Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    overview = profile.get("overview", {})
    overview_lines = [
        f"Total Rows: {overview.get('total_rows', 'N/A'):,}",
        f"Total Columns: {overview.get('total_columns', 'N/A')}",
        f"Numerical Columns: {overview.get('numerical_count', 0)}",
        f"Categorical Columns: {overview.get('categorical_count', 0)}",
        f"Datetime Columns: {overview.get('datetime_count', 0)}",
        f"Missing Values: {overview.get('missing_pct', 0)}%",
        f"Duplicate Rows: {overview.get('duplicate_rows', 0):,}",
    ]
    for line in overview_lines:
        pdf.cell(0, 7, _clean_text_for_pdf(line), new_x="LMARGIN", new_y="NEXT")

    # Column Details
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Column Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    for cp in profile.get("columns", [])[:30]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _clean_text_for_pdf(f"{cp['name']} ({cp['category']})"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, _clean_text_for_pdf(f"  Type: {cp['dtype']}  |  Missing: {cp['missing']} ({cp['missing_pct']}%)  |  Unique: {cp['unique']}"), new_x="LMARGIN", new_y="NEXT")
        if cp["category"] == "numerical":
            pdf.cell(0, 6, _clean_text_for_pdf(f"  Mean: {cp.get('mean', 'N/A')}  |  Median: {cp.get('median', 'N/A')}  |  Std: {cp.get('std', 'N/A')}"), new_x="LMARGIN", new_y="NEXT")
        elif cp["category"] == "categorical" and cp.get("top_values"):
            top_str = ", ".join(f"{k}: {v}" for k, v in list(cp["top_values"].items())[:3])
            pdf.cell(0, 6, _clean_text_for_pdf(f"  Top values: {top_str}"), new_x="LMARGIN", new_y="NEXT")

    # Insights
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Auto-Generated Insights", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)

    insight_eng = InsightEngine(df, col_types)
    insights = insight_eng.generate_insights()
    for ins in insights:
        clean = ins.replace("**", "").replace("📈", "").replace("📊", "").replace("🔍", "").replace("⚠️", "")
        pdf.multi_cell(0, 7, _clean_text_for_pdf(f"- {clean}"))
        pdf.ln(1)

    # Cast to bytes to avoid Streamlit Invalid binary data format: <class 'bytearray'> error
    return bytes(pdf.output())


# ──────────────────────────────────────────────
# Data Profile Sidebar Panel
# ──────────────────────────────────────────────

def render_profile_sidebar(profile: dict):
    """Render compact profile in the sidebar."""
    overview = profile["overview"]
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Dataset Profile")
    st.sidebar.markdown(
        f"**{overview['total_rows']:,}** rows × **{overview['total_columns']}** columns"
    )
    st.sidebar.markdown(
        f"🔢 {overview['numerical_count']} numerical · "
        f"🏷️ {overview['categorical_count']} categorical · "
        f"📅 {overview['datetime_count']} datetime"
    )
    if overview["total_missing"] > 0:
        st.sidebar.markdown(f"⚠️ {overview['missing_pct']}% missing values")
    if overview["duplicate_rows"] > 0:
        st.sidebar.markdown(f"⚠️ {overview['duplicate_rows']:,} duplicates")
    st.sidebar.markdown(f"💾 {overview['memory_mb']:.1f} MB in memory")


# ──────────────────────────────────────────────
# Landing Page (no data loaded)
# ──────────────────────────────────────────────

def render_landing():
    """Welcome screen shown before a dataset is uploaded."""
    st.markdown(
        """
        <div style="text-align:center; padding:3rem 1rem;">
            <h1 style="font-family:'Inter',sans-serif; font-size:2.5rem;
                background:linear-gradient(135deg,#6C5CE7,#00CEC9);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                AI BI Dashboard Generator
            </h1>
            <p style="color:#a0a0b8; font-size:1.1rem; max-width:600px; margin:1rem auto;">
                Upload a CSV or Excel dataset to automatically generate an interactive,
                Power BI–style dashboard with AI-powered insights and natural language queries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    features = [
        ("📊", "Auto Dashboard", "KPI cards, charts, and layout generated instantly"),
        ("🎛️", "Interactive Filters", "Category, date, and numeric range filters"),
        ("💬", "Ask AI", "Natural language questions → visualizations"),
        ("💡", "Smart Insights", "Automated trend, outlier, and correlation detection"),
        ("🔍", "Data Explorer", "Browse, search, and sort your dataset"),
        ("📥", "Export", "Download reports as PDF, CSV, or images"),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="kpi-card" style="margin-bottom:1rem;">
                    <div style="font-size:2rem;">{icon}</div>
                    <div class="kpi-label" style="margin-top:0.5rem;">{title}</div>
                    <div class="kpi-subtitle">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Load sample
    st.markdown("---")
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_sales.csv")
    if os.path.exists(sample_path):
        if st.button("🚀 Load Sample Dataset", use_container_width=True, key="load_sample"):
            df = pd.read_csv(sample_path)
            st.session_state["sample_df"] = df
            st.session_state["sample_name"] = "sample_sales.csv"
            st.rerun()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    uploaded_file, theme = render_sidebar()
    
    # Inject custom CSS with selected theme
    st.markdown(inject_custom_css(theme), unsafe_allow_html=True)

    # Determine data source
    df = None
    source_name = None

    if uploaded_file is not None:
        with st.spinner("📂 Loading and validating dataset..."):
            df, msg, report = load_file(uploaded_file)
        if df is None:
            st.error(f"❌ {msg}")
            return
        source_name = uploaded_file.name
        if report and report.get("warnings"):
            for w in report["warnings"]:
                st.warning(f"⚠️ {w}")

    elif "sample_df" in st.session_state:
        df = st.session_state["sample_df"]
        source_name = st.session_state.get("sample_name", "sample")

    if df is None:
        render_landing()
        return

    # Classify columns & profile
    col_types = get_column_types(df)
    profiler = DataProfiler(df)
    profile = profiler.generate_profile()
    render_profile_sidebar(profile)

    # Tabs
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"📁 **{source_name}**")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "🔍 Data Explorer", "💬 Ask AI", "📥 Export"]
    )

    with tab1:
        tab_dashboard(df, col_types, theme)
    with tab2:
        tab_data_explorer(df, col_types)
    with tab3:
        tab_ask_ai(df, col_types, theme)
    with tab4:
        tab_export(df, col_types, profile)


if __name__ == "__main__":
    main()

"""
Dashboard Builder Module — AI BI Dashboard Generator.
Orchestrates the KPI cards, chart grid, and insight rendering
to compose a Power BI–style dashboard layout in Streamlit.
"""

import pandas as pd
import numpy as np
import streamlit as st

from utils.helpers import (
    format_number,
    format_percentage,
    render_kpi_card,
    render_insight_card,
    KPI_COLORS,
)
from core.chart_generator import ChartGenerator
from core.insight_engine import InsightEngine


class DashboardBuilder:
    """Assembles the full auto-generated dashboard."""

    def __init__(self, df: pd.DataFrame, col_types: dict, theme: str = "Dark"):
        self.df = df
        self.col_types = col_types
        self.theme = theme
        self.chart_gen = ChartGenerator(df, col_types, theme=theme)
        self.insight_eng = InsightEngine(df, col_types)

    def render(self):
        """Render the complete dashboard layout."""
        self.render_kpi_row()
        self.render_chart_grid()
        self.render_insights()

    # ── KPI Row ─────────────────────────────────

    def render_kpi_row(self):
        """Top row of KPI summary cards."""
        st.markdown('<div class="section-header">📌 Key Metrics</div>', unsafe_allow_html=True)

        kpis = self._compute_kpis()
        cols = st.columns(len(kpis))
        color_keys = list(KPI_COLORS.values())

        for i, (label, value, subtitle) in enumerate(kpis):
            color = color_keys[i % len(color_keys)]
            with cols[i]:
                st.markdown(
                    render_kpi_card(label, value, subtitle, color),
                    unsafe_allow_html=True,
                )

    def _compute_kpis(self) -> list:
        """Return list of (label, value_str, subtitle_str) tuples."""
        kpis = []

        # Total records
        kpis.append(("Total Records", format_number(len(self.df)), "rows in dataset"))

        # Mean of first numeric column
        if self.col_types["numerical"]:
            col = self.col_types["numerical"][0]
            mean_val = self.df[col].mean()
            kpis.append((f"Avg {col}", format_number(mean_val), f"mean of {col}"))

        # Sum of second numeric column (if exists)
        if len(self.col_types["numerical"]) >= 2:
            col = self.col_types["numerical"][1]
            total = self.df[col].sum()
            kpis.append((f"Total {col}", format_number(total), f"sum of {col}"))

        # Unique count of first categorical (if exists)
        if self.col_types["categorical"]:
            col = self.col_types["categorical"][0]
            n_unique = self.df[col].nunique()
            kpis.append((f"Unique {col}", format_number(n_unique), f"distinct values"))

        # If less than 3 KPIs, add more numeric stats
        if len(kpis) < 4 and len(self.col_types["numerical"]) >= 1:
            col = self.col_types["numerical"][0]
            max_val = self.df[col].max()
            kpis.append((f"Max {col}", format_number(max_val), f"maximum value"))

        return kpis[:5]

    # ── Chart Grid ──────────────────────────────

    def render_chart_grid(self):
        """Two-column chart grid with auto-selected charts."""
        st.markdown('<div class="section-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

        charts = self.chart_gen.auto_charts()

        if not charts:
            st.info("No suitable charts could be generated for this dataset.")
            return

        # Render in 2-column grid
        for i in range(0, len(charts), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(charts):
                    title, fig = charts[idx]
                    with col:
                        st.plotly_chart(fig, use_container_width=True, key=f"auto_chart_{idx}")

    # ── Insights ────────────────────────────────

    def render_insights(self):
        """Render textual insight cards."""
        st.markdown('<div class="section-header">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)

        insights = self.insight_eng.generate_insights()

        if not insights:
            st.info("No significant insights detected.")
            return

        # AI summary (if available)
        ai_summary = self.insight_eng.get_ai_summary(insights)
        if ai_summary:
            st.markdown(
                f'<div class="insight-card" style="border-left-color: #00CEC9;">'
                f'🤖 <strong>AI Summary:</strong> {ai_summary}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

        # Individual insight cards in 2 columns
        col1, col2 = st.columns(2)
        for i, insight in enumerate(insights):
            target = col1 if i % 2 == 0 else col2
            with target:
                st.markdown(render_insight_card(insight), unsafe_allow_html=True)

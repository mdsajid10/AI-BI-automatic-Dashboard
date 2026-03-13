"""
Chart Generator Module — AI BI Dashboard Generator.
Automatically selects and generates interactive Plotly charts
based on column types and data characteristics.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import (
    PALETTE_MODERN,
    PALETTE_DARK,
    KPI_COLORS,
    format_number,
    get_top_categories,
)


def get_chart_template(theme: str = "Dark"):
    """Return a Plotly template dict based on the theme."""
    is_light = theme == "Light"
    return dict(
        layout=dict(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#2d3436" if is_light else "#d0d0e0", size=12),
            title_font=dict(size=16, color="#1e1e2e" if is_light else "#e0e0f0"),
            margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=11, color="#636e72" if is_light else "#b0b0c8"),
            ),
            xaxis=dict(
                gridcolor="rgba(108,92,231,0.15)" if is_light else "rgba(108,92,231,0.08)",
                zerolinecolor="rgba(108,92,231,0.2)" if is_light else "rgba(108,92,231,0.15)",
                tickfont=dict(color="#636e72" if is_light else "#888"),
                title_font=dict(color="#1e1e2e" if is_light else "#e0e0f0"),
            ),
            yaxis=dict(
                gridcolor="rgba(108,92,231,0.15)" if is_light else "rgba(108,92,231,0.08)",
                zerolinecolor="rgba(108,92,231,0.2)" if is_light else "rgba(108,92,231,0.15)",
                tickfont=dict(color="#636e72" if is_light else "#888"),
                title_font=dict(color="#1e1e2e" if is_light else "#e0e0f0"),
            ),
            colorway=PALETTE_MODERN,
        )
    )


class ChartGenerator:
    """Generates interactive Plotly charts from DataFrames."""

    def __init__(self, df: pd.DataFrame, col_types: dict, theme: str = "Dark"):
        self.df = df
        self.col_types = col_types
        self.theme = theme
        self.template = get_chart_template(theme)
        self.palette = PALETTE_MODERN

    # ── Auto Dashboard Charts ───────────────────

    def auto_charts(self) -> list:
        """
        Generate a list of (title, fig) tuples for automatic dashboard.
        Picks the best charts based on available column types.
        """
        charts = []

        # Category distribution (first suitable categorical col)
        if self.col_types["categorical"]:
            cat_col = self.col_types["categorical"][0]
            charts.append(("Category Distribution", self.category_bar(cat_col)))

        # Time trend (if datetime + numeric exist)
        if self.col_types["datetime"] and self.col_types["numerical"]:
            dt_col = self.col_types["datetime"][0]
            num_col = self.col_types["numerical"][0]
            charts.append(("Trend Over Time", self.time_trend(dt_col, num_col)))

        # Distribution histogram (first numeric)
        if self.col_types["numerical"]:
            num_col = self.col_types["numerical"][0]
            charts.append(("Distribution", self.histogram(num_col)))

        # Pie chart (first categorical if few categories)
        if self.col_types["categorical"]:
            for cat_col in self.col_types["categorical"]:
                if self.df[cat_col].nunique() <= 10:
                    charts.append(("Composition", self.pie_chart(cat_col)))
                    break

        # Correlation heatmap (if ≥2 numeric columns)
        if len(self.col_types["numerical"]) >= 2:
            charts.append(("Correlation Heatmap", self.correlation_heatmap()))

        # Scatter (if ≥2 numeric, different from heatmap pair)
        if len(self.col_types["numerical"]) >= 2:
            c1, c2 = self.col_types["numerical"][0], self.col_types["numerical"][1]
            color_col = self.col_types["categorical"][0] if self.col_types["categorical"] else None
            charts.append(("Scatter Analysis", self.scatter(c1, c2, color_col)))

        return charts

    # ── Individual Chart Builders ───────────────

    def category_bar(self, col: str, top_n: int = 12) -> go.Figure:
        """Bar chart of category value counts."""
        counts = get_top_categories(self.df[col].dropna(), top_n)
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": col, "y": "Count"},
            color_discrete_sequence=self.palette,
        )
        fig.update_layout(**self.template["layout"], title=f"{col} Distribution")
        fig.update_traces(marker_line_width=0, opacity=0.9)
        return fig

    def time_trend(self, dt_col: str, num_col: str) -> go.Figure:
        """Line chart of a numeric column over time."""
        df_sorted = self.df.dropna(subset=[dt_col, num_col]).copy()
        df_sorted[dt_col] = pd.to_datetime(df_sorted[dt_col], errors="coerce")
        df_sorted = df_sorted.dropna(subset=[dt_col]).sort_values(dt_col)

        # Auto-aggregate if too many points
        if len(df_sorted) > 200:
            df_sorted = df_sorted.set_index(dt_col).resample("W")[num_col].mean().reset_index()

        fig = px.line(
            df_sorted, x=dt_col, y=num_col,
            labels={dt_col: "Date", num_col: num_col},
            color_discrete_sequence=[self.palette[0]],
        )
        fig.update_layout(**self.template["layout"], title=f"{num_col} Over Time")
        fig.update_traces(line_width=2.5)
        # Add area fill
        fig.update_traces(fill="tozeroy", fillcolor=f"rgba(108,92,231,0.10)")
        return fig

    def histogram(self, col: str, bins: int = 40) -> go.Figure:
        """Histogram for a numeric column."""
        fig = px.histogram(
            self.df, x=col, nbins=bins,
            color_discrete_sequence=[self.palette[1]],
        )
        fig.update_layout(**self.template["layout"], title=f"{col} Distribution")
        fig.update_traces(marker_line_width=0, opacity=0.85)
        return fig

    def pie_chart(self, col: str, top_n: int = 8) -> go.Figure:
        """Pie / donut chart for a categorical column."""
        counts = get_top_categories(self.df[col].dropna(), top_n)
        fig = px.pie(
            names=counts.index.astype(str),
            values=counts.values,
            color_discrete_sequence=self.palette,
            hole=0.45,
        )
        fig.update_layout(**self.template["layout"], title=f"{col} Composition")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    def correlation_heatmap(self) -> go.Figure:
        """Heatmap of numeric column correlations."""
        num_df = self.df[self.col_types["numerical"]].dropna()
        if num_df.empty or len(num_df.columns) < 2:
            return self._empty_fig("Not enough numeric data for correlation.")

        corr = num_df.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#6C5CE7", "#1e1e2e", "#00CEC9"],
            zmin=-1, zmax=1,
        )
        fig.update_layout(**self.template["layout"], title="Correlation Matrix")
        return fig

    def scatter(self, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
        """Scatter plot between two numeric columns."""
        kwargs = dict(x=x_col, y=y_col, color_discrete_sequence=self.palette, opacity=0.7)
        if color_col and self.df[color_col].nunique() <= 15:
            kwargs["color"] = color_col

        fig = px.scatter(self.df, **kwargs)
        fig.update_layout(**self.template["layout"], title=f"{y_col} vs {x_col}")
        return fig

    def grouped_bar(self, cat_col: str, num_col: str, agg: str = "sum") -> go.Figure:
        """Grouped / aggregated bar chart."""
        grouped = self.df.groupby(cat_col)[num_col].agg(agg).reset_index()
        grouped = grouped.sort_values(num_col, ascending=False).head(15)
        fig = px.bar(
            grouped, x=cat_col, y=num_col,
            color_discrete_sequence=self.palette,
        )
        fig.update_layout(**self.template["layout"], title=f"{num_col} by {cat_col}")
        fig.update_traces(marker_line_width=0, opacity=0.9)
        return fig

    # ── Utility ─────────────────────────────────

    def _empty_fig(self, msg: str = "No data available") -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#a0a0b8"))
        fig.update_layout(**self.template["layout"])
        return fig

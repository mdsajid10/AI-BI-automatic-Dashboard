"""
Natural Language Query Engine — AI BI Dashboard Generator.
Converts natural language questions into data operations and visualizations
using Groq (Llama3) or OpenAI GPT as the AI backend.
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple

from utils.helpers import PALETTE_MODERN, format_number
from core.chart_generator import get_chart_template

# Lazy-import AI clients to avoid forcing both dependencies
_groq_client = None
_openai_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# Chart template is now dynamic via get_chart_template


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a data analyst assistant. Given a user's question about a dataset, return a JSON object that describes the analysis needed.

The dataset has these columns:
{columns_info}

Return ONLY a valid JSON object with these keys:
- "analysis_type": one of "groupby", "correlation", "distribution", "trend", "filter", "comparison", "summary"
- "columns": list of column names to use (1-3 columns)
- "aggregation": aggregation function if needed: "sum", "mean", "count", "median", "min", "max" (or null)
- "filter_conditions": list of {{"column": str, "operator": str, "value": any}} or null
- "chart_type": one of "bar", "line", "scatter", "histogram", "pie", "heatmap" (ALWAYS respect the user's specific request if they mention a chart type)
- "title": descriptive chart title
- "explanation": one sentence explaining the analysis

CRITICAL: If the user explicitly asks for a specific chart type (e.g., "pie chart", "line graph", "histogram"), you MUST return that chart type even if another type seems more standard.

Do NOT include any text outside the JSON object.
"""


class NLQueryEngine:
    """Process natural language questions against a DataFrame."""

    def __init__(self, df: pd.DataFrame, col_types: dict, theme: str = "Dark"):
        self.df = df
        self.col_types = col_types
        self.theme = theme
        self.template = get_chart_template(theme)
        self.provider = self._detect_provider()

    @staticmethod
    def _detect_provider() -> Optional[str]:
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return None

    def is_available(self) -> bool:
        return self.provider is not None

    # ── Main Entry Point ────────────────────────

    def ask(self, question: str) -> dict:
        """
        Process a natural language question.

        Returns:
            {
                "success": bool,
                "figure": go.Figure or None,
                "result_df": pd.DataFrame or None,
                "explanation": str,
                "error": str or None,
            }
        """
        if not self.is_available():
            return self._fallback_analysis(question)

        # Build column info string
        columns_info = self._columns_description()

        # Call LLM
        try:
            parsed = self._call_llm(question, columns_info)
        except Exception as e:
            return self._fallback_analysis(question, error_context=str(e))

        # Execute the parsed query
        try:
            result_df, fig = self._execute(parsed, question)
            return {
                "success": True,
                "figure": fig,
                "result_df": result_df,
                "explanation": parsed.get("explanation", ""),
                "error": None,
            }
        except Exception as e:
            return self._fallback_analysis(question, error_context=str(e))

    # ── LLM Interaction ─────────────────────────

    def _call_llm(self, question: str, columns_info: str) -> dict:
        prompt = _SYSTEM_PROMPT.format(columns_info=columns_info)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]

        if self.provider == "groq":
            client = _get_groq_client()
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content
        else:
            client = _get_openai_client()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.1,
                max_tokens=500,
            )
            raw = resp.choices[0].message.content

        # Parse JSON from response (handle markdown code blocks)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        return json.loads(raw)

    def _columns_description(self) -> str:
        lines = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            nunique = self.df[col].nunique()
            cat = "numerical" if col in self.col_types["numerical"] else \
                  "categorical" if col in self.col_types["categorical"] else \
                  "datetime" if col in self.col_types["datetime"] else "other"

            sample_vals = ""
            if cat == "categorical" and nunique <= 20:
                vals = self.df[col].dropna().unique()[:8]
                sample_vals = f" | values: {list(vals)}"
            elif cat == "numerical":
                sample_vals = f" | range: [{self.df[col].min()}, {self.df[col].max()}]"

            lines.append(f"- {col} ({cat}, {dtype}, {nunique} unique{sample_vals})")
        return "\n".join(lines)

    # ── Query Execution ─────────────────────────

    def _execute(self, parsed: dict, question: str = "") -> Tuple[pd.DataFrame, go.Figure]:
        analysis = parsed.get("analysis_type", "summary")
        columns = parsed.get("columns", [])
        agg = parsed.get("aggregation", "sum")
        
        # Override chart type based on explicit question mentions
        chart_type = self._detect_chart_type(question) or parsed.get("chart_type", "bar")
        title = parsed.get("title", "Analysis Result")

        # Validate columns exist
        columns = [c for c in columns if c in self.df.columns]
        if not columns:
            columns = self.col_types["numerical"][:1] or self.col_types["categorical"][:1]

        df = self.df.copy()

        # Apply filters if any
        filters = parsed.get("filter_conditions")
        if filters:
            df = self._apply_filters(df, filters)

        if analysis == "groupby" and len(columns) >= 2:
            group_col, val_col = columns[0], columns[1]
            result_df = df.groupby(group_col)[val_col].agg(agg or "sum").reset_index()
            result_df = result_df.sort_values(val_col, ascending=False).head(20)
            fig = self._make_chart(result_df, group_col, val_col, chart_type, title)

        elif analysis == "correlation" and len(columns) >= 2:
            result_df = df[columns].dropna()
            # If they ask for heatmap/pie, we might need different data, 
            # but for now we try to fit the numeric pairs into requested chart.
            fig = self._make_chart(result_df, columns[0], columns[1], chart_type or "scatter", title)

        elif analysis == "distribution" and columns:
            col = columns[0]
            if chart_type in ("pie", "bar"):
                result_df = df[col].value_counts().reset_index()
                result_df.columns = [col, "Count"]
                result_df = result_df.head(20)
                fig = self._make_chart(result_df, col, "Count", chart_type, title)
            elif chart_type == "histogram":
                result_df = df[[col]].dropna()
                fig = self._make_chart(result_df, None, col, "histogram", title)
            else:
                # Fallback to pie or bar for categorical distribution if requested
                result_df = df[col].value_counts().reset_index()
                result_df.columns = [col, "Count"]
                fig = self._make_chart(result_df, col, "Count", chart_type or "bar", title)

        elif analysis == "trend" and len(columns) >= 2:
            dt_col, val_col = columns[0], columns[1]
            result_df = df[[dt_col, val_col]].dropna().sort_values(dt_col)
            fig = self._make_chart(result_df, dt_col, val_col, chart_type or "line", title)

        elif analysis == "comparison" and len(columns) >= 2:
            group_col, val_col = columns[0], columns[1]
            result_df = df.groupby(group_col)[val_col].agg(agg or "mean").reset_index()
            result_df = result_df.sort_values(val_col, ascending=False).head(15)
            fig = self._make_chart(result_df, group_col, val_col, chart_type, title)

        else:
            # Fallback: simple summary
            chart_type = self._detect_chart_type(title + " " + (parsed.get("explanation", ""))) or chart_type
            
            if columns and columns[0] in self.col_types["numerical"]:
                result_df = df[[columns[0]]].describe().reset_index()
                result_df.columns = ["Statistic", columns[0]]
                fig = self._make_chart(result_df, "Statistic", columns[0], chart_type, title)
            elif columns and columns[0] in self.col_types["categorical"]:
                vc = df[columns[0]].value_counts().head(15).reset_index()
                vc.columns = [columns[0], "Count"]
                result_df = vc
                fig = self._make_chart(result_df, columns[0], "Count", chart_type, title)
            else:
                result_df = df.head(10)
                fig = go.Figure()
                fig.add_annotation(text="Could not determine appropriate chart", showarrow=False)
                fig.update_layout(**self.template["layout"], title=title)

        return result_df, fig

    @staticmethod
    def _apply_filters(df: pd.DataFrame, filters: list) -> pd.DataFrame:
        for f in filters:
            col = f.get("column")
            op = f.get("operator", "==")
            val = f.get("value")
            if col not in df.columns:
                continue
            try:
                if op == "==":
                    df = df[df[col] == val]
                elif op == "!=":
                    df = df[df[col] != val]
                elif op == ">":
                    df = df[df[col] > float(val)]
                elif op == "<":
                    df = df[df[col] < float(val)]
                elif op == ">=":
                    df = df[df[col] >= float(val)]
                elif op == "<=":
                    df = df[df[col] <= float(val)]
                elif op in ("in", "contains"):
                    df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
            except Exception:
                pass
        return df

    def _make_chart(self, df, x_col, y_col, chart_type, title) -> go.Figure:
        """Centralized chart factory to support all requested types."""
        try:
            if chart_type == "pie":
                fig = px.pie(df, names=x_col, values=y_col, color_discrete_sequence=PALETTE_MODERN, hole=0.45)
            elif chart_type == "line":
                fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=[PALETTE_MODERN[0]])
                fig.update_traces(line_width=2.5, fill="tozeroy", fillcolor="rgba(108,92,231,0.10)")
            elif chart_type == "area":
                 fig = px.area(df, x=x_col, y=y_col, color_discrete_sequence=PALETTE_MODERN)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=PALETTE_MODERN, opacity=0.7)
            elif chart_type == "histogram":
                # For histograms, x_col might be null if we use y_col as the variable
                plot_x = x_col if x_col else y_col
                fig = px.histogram(df, x=plot_x, nbins=40, color_discrete_sequence=[PALETTE_MODERN[1]])
            elif chart_type == "heatmap":
                if x_col and y_col:
                    # For a simple heatmap between two cols, we might need a 2D histogram or correlation
                    fig = px.density_heatmap(df, x=x_col, y=y_col, color_continuous_scale="Purples")
                else:
                    fig = px.imshow(df.corr(), color_continuous_scale="Purples")
            elif chart_type == "box":
                fig = px.box(df, x=x_col, y=y_col, color_discrete_sequence=PALETTE_MODERN)
            elif chart_type == "violin":
                fig = px.violin(df, x=x_col, y=y_col, color_discrete_sequence=PALETTE_MODERN, box=True)
            else:
                # Default to bar
                fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=PALETTE_MODERN)
            
            fig.update_layout(**self.template["layout"], title=title)
            return fig
        except Exception:
            # Absolute fallback if data doesn't fit the requested chart
            fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=PALETTE_MODERN)
            fig.update_layout(**self.template["layout"], title=title)
            return fig

    # ── Fallback (rule-based) ───────────────────

    def _fallback_analysis(self, question: str, error_context: str = None) -> dict:
        """Simple keyword-based analysis when no AI provider is available."""
        q = question.lower()
        explanation = ""
        fig = None
        result_df = None

        if error_context:
            explanation = f"AI service unavailable ({error_context}). Using rule-based analysis. "

        # Keyword patterns
        if any(kw in q for kw in ["by", "per", "each", "group"]):
            # Try to find cat + num columns from question
            cat_col = self._find_column_in_question(q, self.col_types["categorical"])
            num_col = self._find_column_in_question(q, self.col_types["numerical"])

            if cat_col and num_col:
                result_df = self.df.groupby(cat_col)[num_col].sum().reset_index()
                result_df = result_df.sort_values(num_col, ascending=False).head(15)
                
                chart_type = self._detect_chart_type(q) or "bar"
                fig = self._make_chart(result_df, cat_col, num_col, chart_type, f"{num_col} by {cat_col}")
                explanation += f"Grouped {num_col} by {cat_col} (using {chart_type} chart)."

        elif any(kw in q for kw in ["correlation", "relationship", "vs", "versus"]):
            nums = self.col_types["numerical"][:2]
            if len(nums) >= 2:
                result_df = self.df[nums].dropna()
                chart_type = self._detect_chart_type(q) or "scatter"
                fig = self._make_chart(result_df, nums[0], nums[1], chart_type, f"{nums[0]} vs {nums[1]}")
                explanation += f"Analyzed relationship between {nums[0]} and {nums[1]} using {chart_type} chart."

        elif any(kw in q for kw in ["distribution", "histogram", "spread"]):
            num_col = self._find_column_in_question(q, self.col_types["numerical"])
            if num_col:
                result_df = self.df[[num_col]].dropna()
                chart_type = self._detect_chart_type(q) or "histogram"
                fig = self._make_chart(result_df, None, num_col, chart_type, f"Distribution of {num_col}")
                explanation += f"Analyzed distribution of {num_col} using {chart_type} chart."

        elif any(kw in q for kw in ["trend", "over time", "timeline"]):
            dt_col = self.col_types["datetime"][0] if self.col_types["datetime"] else None
            num_col = self._find_column_in_question(q, self.col_types["numerical"])
            if dt_col and num_col:
                result_df = self.df[[dt_col, num_col]].dropna().sort_values(dt_col)
                chart_type = self._detect_chart_type(q) or "line"
                fig = self._make_chart(result_df, dt_col, num_col, chart_type, f"{num_col} Over Time")
                explanation += f"Trend of {num_col} over {dt_col} (as {chart_type})."

        if fig is None:
            # Last resort: top categorical value counts
            if self.col_types["categorical"]:
                col = self.col_types["categorical"][0]
                vc = self.df[col].value_counts().head(10).reset_index()
                vc.columns = [col, "Count"]
                result_df = vc
                chart_type = self._detect_chart_type(q) or "bar"
                fig = self._make_chart(result_df, col, "Count", chart_type, f"Top {col}")
                explanation += f"Showing top values of {col} (using {chart_type} chart)."
            else:
                fig = go.Figure()
                fig.add_annotation(text="Could not interpret the question.", showarrow=False)
                fig.update_layout(**self.template["layout"])
                explanation = "Could not interpret the question. Try rephrasing."

        return {
            "success": fig is not None,
            "figure": fig,
            "result_df": result_df,
            "explanation": explanation,
            "error": error_context,
        }

    def _find_column_in_question(self, question: str, candidates: list) -> Optional[str]:
        """Find the first column name mentioned (or partially matched) in the question."""
        q = question.lower()
        for col in candidates:
            if col.lower() in q or col.lower().replace("_", " ") in q:
                return col
        return candidates[0] if candidates else None

    def _detect_chart_type(self, question: str) -> Optional[str]:
        """Detect if the user explicitly requested a specific chart type."""
        q = question.lower().replace(" ", "").replace("_", "")
        mapping = {
            "pie": "pie",
            "donut": "pie",
            "circle": "pie",
            "bar": "bar",
            "column": "bar",
            "line": "line",
            "trend": "line",
            "scatter": "scatter",
            "dot": "scatter",
            "point": "scatter",
            "histogram": "histogram",
            "distribution": "histogram",
            "hist": "histogram",
            "heatmap": "heatmap",
            "density": "heatmap",
            "matrix": "heatmap",
            "area": "area",
            "box": "box",
            "whisker": "box",
            "violin": "violin"
        }
        for kw, chart in mapping.items():
            if kw in q:
                return chart
        return None

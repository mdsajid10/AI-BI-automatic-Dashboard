"""
Insight Engine Module — AI BI Dashboard Generator.
Generates statistical insights from a DataFrame — trends, top categories,
outliers, correlations, and distribution observations.
Optionally enhances insights with AI-generated explanations.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import List, Optional

from utils.helpers import format_number, format_percentage


class InsightEngine:
    """Generate textual insights from a DataFrame."""

    def __init__(self, df: pd.DataFrame, col_types: dict):
        self.df = df
        self.col_types = col_types

    def generate_insights(self, max_insights: int = 12) -> List[str]:
        """
        Return a list of human-readable insight strings.
        Combines rule-based statistical insights with optional AI enhancement.
        """
        insights = []

        insights.extend(self._overview_insights())
        insights.extend(self._top_category_insights())
        insights.extend(self._numeric_insights())
        insights.extend(self._correlation_insights())
        insights.extend(self._outlier_insights())
        insights.extend(self._trend_insights())

        # Deduplicate and cap
        seen = set()
        unique = []
        for ins in insights:
            if ins not in seen:
                seen.add(ins)
                unique.append(ins)
        return unique[:max_insights]

    # ── Overview ────────────────────────────────

    def _overview_insights(self) -> List[str]:
        insights = []
        n = len(self.df)
        ncols = len(self.df.columns)
        insights.append(f"The dataset contains **{n:,}** records across **{ncols}** columns.")

        dup = int(self.df.duplicated().sum())
        if dup > 0:
            pct = dup / n * 100
            insights.append(f"⚠️ **{dup:,}** duplicate rows detected ({pct:.1f}% of data).")

        total_missing = int(self.df.isna().sum().sum())
        if total_missing > 0:
            overall_pct = total_missing / (n * ncols) * 100
            insights.append(f"Missing values account for **{overall_pct:.1f}%** of all data cells.")

        return insights

    # ── Top Categories ──────────────────────────

    def _top_category_insights(self) -> List[str]:
        insights = []
        for col in self.col_types["categorical"][:3]:
            vc = self.df[col].value_counts()
            if len(vc) == 0:
                continue
            top_val = vc.index[0]
            top_count = vc.iloc[0]
            pct = top_count / len(self.df) * 100
            insights.append(
                f"**{col}**: \"{top_val}\" is the most frequent value, "
                f"appearing **{top_count:,}** times ({pct:.1f}%)."
            )

            if len(vc) >= 2:
                bottom_val = vc.index[-1]
                bottom_count = vc.iloc[-1]
                ratio = top_count / bottom_count if bottom_count > 0 else float("inf")
                if ratio > 5:
                    insights.append(
                        f"**{col}**: Highly skewed — \"{top_val}\" is **{ratio:.0f}x** more common "
                        f"than \"{bottom_val}\"."
                    )
        return insights

    # ── Numeric Insights ────────────────────────

    def _numeric_insights(self) -> List[str]:
        insights = []
        for col in self.col_types["numerical"][:4]:
            series = self.df[col].dropna()
            if len(series) == 0:
                continue

            mean_val = series.mean()
            median_val = series.median()
            std_val = series.std()

            # Skewness check
            skew = series.skew()
            if abs(skew) > 1:
                direction = "right (positively)" if skew > 0 else "left (negatively)"
                insights.append(
                    f"**{col}** is significantly skewed to the {direction} "
                    f"(skewness: {skew:.2f}), indicating non-uniform distribution."
                )

            # Coefficient of variation
            if mean_val != 0:
                cv = (std_val / abs(mean_val)) * 100
                if cv > 80:
                    insights.append(
                        f"**{col}** shows high variability (CV: {cv:.0f}%), "
                        f"ranging from {format_number(series.min())} to {format_number(series.max())}."
                    )

        return insights

    # ── Correlation Insights ────────────────────

    def _correlation_insights(self) -> List[str]:
        insights = []
        nums = self.col_types["numerical"]
        if len(nums) < 2:
            return insights

        corr_matrix = self.df[nums].corr()

        # Find strong correlations
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                r = corr_matrix.iloc[i, j]
                if abs(r) >= 0.7:
                    direction = "strong positive" if r > 0 else "strong negative"
                    insights.append(
                        f"📈 **{nums[i]}** and **{nums[j]}** have a {direction} "
                        f"correlation (r = {r:.2f})."
                    )
                elif abs(r) >= 0.5:
                    direction = "moderate positive" if r > 0 else "moderate negative"
                    insights.append(
                        f"**{nums[i]}** and **{nums[j]}** show a {direction} "
                        f"correlation (r = {r:.2f})."
                    )
        return insights

    # ── Outlier Insights ────────────────────────

    def _outlier_insights(self) -> List[str]:
        insights = []
        for col in self.col_types["numerical"][:4]:
            series = self.df[col].dropna()
            if len(series) < 10:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]

            if len(outliers) > 0:
                pct = len(outliers) / len(series) * 100
                if pct >= 1:
                    insights.append(
                        f"🔍 **{col}**: {len(outliers):,} outliers detected ({pct:.1f}%), "
                        f"outside range [{format_number(lower)} – {format_number(upper)}]."
                    )
        return insights

    # ── Trend Insights ──────────────────────────

    def _trend_insights(self) -> List[str]:
        insights = []
        if not self.col_types["datetime"] or not self.col_types["numerical"]:
            return insights

        dt_col = self.col_types["datetime"][0]
        for num_col in self.col_types["numerical"][:2]:
            try:
                temp = self.df[[dt_col, num_col]].dropna().copy()
                temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce")
                temp = temp.dropna().sort_values(dt_col)

                if len(temp) < 5:
                    continue

                # Compare first and last quarter
                n = len(temp)
                first_q = temp.head(n // 4)[num_col].mean()
                last_q = temp.tail(n // 4)[num_col].mean()

                if first_q > 0:
                    change = ((last_q - first_q) / first_q) * 100
                    direction = "increased" if change > 0 else "decreased"
                    if abs(change) > 5:
                        insights.append(
                            f"📊 **{num_col}** has {direction} by **{abs(change):.1f}%** "
                            f"from the earliest to the latest period."
                        )
            except Exception:
                pass

        return insights

    # ── AI-Enhanced Insights (optional) ─────────

    def get_ai_summary(self, insights: List[str]) -> Optional[str]:
        """
        Optionally use AI to create a cohesive narrative from the insights.
        Returns None if no AI provider is available.
        """
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None

        prompt = (
            "You are a data analyst. Given these statistical insights about a dataset, "
            "write a concise executive summary paragraph (3-5 sentences) highlighting "
            "the most important findings:\n\n"
            + "\n".join(f"- {ins}" for ins in insights[:8])
        )

        try:
            if os.getenv("GROQ_API_KEY"):
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                )
            else:
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=300,
                )
            return resp.choices[0].message.content.strip()
        except Exception:
            return None

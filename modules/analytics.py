"""
analytics.py
------------
Builds the Plotly figures used on the dashboard, plus helper functions for
applying interactive filters to the enriched dataframe.

A consistent dark, glassmorphism-friendly color palette is used across all
charts so they match the app's premium SaaS aesthetic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Palette (kept in sync with style.css design tokens)
# ----------------------------------------------------------------------
COLOR_SEQUENCE = [
    "#7C5CFC", "#22D3EE", "#F472B6", "#34D399", "#FBBF24",
    "#60A5FA", "#F87171", "#A78BFA", "#2DD4BF", "#FB923C",
]
BG_TRANSPARENT = "rgba(0,0,0,0)"
FONT_COLOR = "#E5E7EB"
GRID_COLOR = "rgba(229,231,235,0.08)"


def _base_layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=FONT_COLOR), x=0.02),
        paper_bgcolor=BG_TRANSPARENT,
        plot_bgcolor=BG_TRANSPARENT,
        font=dict(color=FONT_COLOR, family="Inter, sans-serif", size=12),
        margin=dict(l=10, r=10, t=48, b=10),
        height=height,
        legend=dict(bgcolor=BG_TRANSPARENT, font=dict(color=FONT_COLOR)),
        hoverlabel=dict(bgcolor="#1F2430", font_color="#F9FAFB"),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig


def apply_filters(
    df: pd.DataFrame,
    platform: Optional[List[str]] = None,
    industry: Optional[List[str]] = None,
    role: Optional[List[str]] = None,
    seniority: Optional[List[str]] = None,
    location: Optional[List[str]] = None,
    quality_range: Optional[tuple] = None,
) -> pd.DataFrame:
    filtered = df.copy()
    if platform:
        filtered = filtered[filtered["Detected Platform"].isin(platform)]
    if industry:
        filtered = filtered[filtered["Industry"].isin(industry)]
    if role:
        filtered = filtered[filtered["Professional Role"].isin(role)]
    if seniority:
        filtered = filtered[filtered["Seniority"].isin(seniority)]
    if location:
        filtered = filtered[filtered["Location Group"].isin(location)]
    if quality_range:
        lo, hi = quality_range
        filtered = filtered[
            (filtered["Profile Quality Score"] >= lo) & (filtered["Profile Quality Score"] <= hi)
        ]
    return filtered


def _bar_from_counts(series: pd.Series, title: str, top_n: int = 12) -> go.Figure:
    counts = series.value_counts().head(top_n).sort_values(ascending=True)
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        color=counts.index, color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(showlegend=False, marker_line_width=0)
    return _base_layout(fig, title)


def chart_by_platform(df: pd.DataFrame) -> go.Figure:
    return _bar_from_counts(df["Detected Platform"], "Profiles by Platform")


def chart_by_industry(df: pd.DataFrame) -> go.Figure:
    return _bar_from_counts(df["Industry"], "Profiles by Industry")


def chart_by_role(df: pd.DataFrame) -> go.Figure:
    return _bar_from_counts(df["Professional Role"], "Professional Role Distribution")


def chart_top_skills(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    all_skills = (
        df["Extracted Skills"].fillna("").apply(lambda s: [x.strip() for x in s.split(",") if x.strip()])
    )
    flat = [skill for row in all_skills for skill in row]
    if not flat:
        fig = go.Figure()
        return _base_layout(fig, "Top Skills (none detected)")
    counts = pd.Series(flat).value_counts().head(top_n).sort_values(ascending=True)
    fig = px.bar(
        x=counts.values, y=counts.index, orientation="h",
        color=counts.index, color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(showlegend=False, marker_line_width=0)
    return _base_layout(fig, "Top Skills", height=420)


def chart_seniority(df: pd.DataFrame) -> go.Figure:
    counts = df["Seniority"].value_counts()
    fig = px.pie(
        names=counts.index, values=counts.values, hole=0.55,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textfont=dict(color=FONT_COLOR), textinfo="percent+label")
    return _base_layout(fig, "Seniority Distribution")


def chart_location(df: pd.DataFrame) -> go.Figure:
    return _bar_from_counts(df["Location Group"], "Location Distribution")


def chart_quality_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Profile Quality Score", nbins=20,
        color_discrete_sequence=[COLOR_SEQUENCE[0]],
    )
    fig.update_traces(marker_line_width=0)
    return _base_layout(fig, "Profile Quality Distribution")


def chart_missing_data(df: pd.DataFrame, column_map: Dict[str, str]) -> go.Figure:
    rows = []
    for field_name, col in column_map.items():
        if col not in df.columns:
            continue
        missing_pct = df[col].apply(
            lambda v: pd.isna(v) or str(v).strip() in ("", "nan")
        ).mean() * 100
        rows.append((field_name.replace("_", " ").title(), round(missing_pct, 1)))
    if not rows:
        fig = go.Figure()
        return _base_layout(fig, "Missing Data Overview")
    labels, values = zip(*sorted(rows, key=lambda x: x[1]))
    fig = px.bar(
        x=values, y=labels, orientation="h",
        color=values, color_continuous_scale=["#34D399", "#FBBF24", "#F87171"],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False)
    return _base_layout(fig, "Missing Data Overview (% missing per field)")


def kpi_summary(df: pd.DataFrame) -> Dict[str, float]:
    total = len(df)
    valid = int((df["Profile Quality Score"] >= 50).sum()) if "Profile Quality Score" in df else 0
    duplicates = int(df["_is_exact_duplicate"].sum()) if "_is_exact_duplicate" in df else 0
    enriched = int((df["Profile Completeness"] > 0).sum()) if "Profile Completeness" in df else total
    avg_quality = round(df["Profile Quality Score"].mean(), 1) if "Profile Quality Score" in df and total else 0.0
    return {
        "total_profiles": total,
        "valid_profiles": valid,
        "duplicate_profiles": duplicates,
        "enriched_profiles": enriched,
        "avg_quality_score": avg_quality,
    }

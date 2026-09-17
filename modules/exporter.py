"""
exporter.py
-----------
Handles exporting cleaned/enriched datasets to CSV, XLSX, and a PDF
quality report built with ReportLab.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

DISPLAY_COLUMNS_EXCLUDE = {"_is_exact_duplicate", "_fuzzy_duplicate_group", "_quality_breakdown"}


def _presentable(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in DISPLAY_COLUMNS_EXCLUDE]
    return df[cols]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return _presentable(df).to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Profiles") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _presentable(df).to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def _mini_table(data_rows, col_widths=None):
    tbl = Table(data_rows, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7C5CFC")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F4F6")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def build_pdf_report(
    df: pd.DataFrame,
    column_map: Dict[str, str],
    cleaning_summary_lines: list,
    kpis: Dict[str, float],
) -> bytes:
    """Generate the Profile Quality Report PDF and return raw bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], textColor=colors.HexColor("#4C1D95"),
    )
    heading_style = ParagraphStyle(
        "HeadingCustom", parent=styles["Heading2"], textColor=colors.HexColor("#5B21B6"),
        spaceBefore=14, spaceAfter=8,
    )
    body_style = styles["BodyText"]

    story = []
    story.append(Paragraph("ProfileBoost AI", title_style))
    story.append(Paragraph("Profile Quality Report", styles["Heading3"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
    story.append(Spacer(1, 12))

    # Dataset summary
    story.append(Paragraph("Dataset Summary", heading_style))
    summary_rows = [["Metric", "Value"]] + [
        ["Total Profiles", str(kpis.get("total_profiles", 0))],
        ["Valid Profiles (score >= 50)", str(kpis.get("valid_profiles", 0))],
        ["Duplicate Profiles", str(kpis.get("duplicate_profiles", 0))],
        ["Enriched Profiles", str(kpis.get("enriched_profiles", 0))],
        ["Average Quality Score", f"{kpis.get('avg_quality_score', 0)}/100"],
    ]
    story.append(_mini_table(summary_rows, col_widths=[8 * cm, 6 * cm]))
    story.append(Spacer(1, 10))

    # Cleaning summary
    story.append(Paragraph("Data Cleaning Summary", heading_style))
    for line in cleaning_summary_lines:
        story.append(Paragraph(f"• {line}", body_style))
    story.append(Spacer(1, 10))

    # Platform distribution
    if "Detected Platform" in df.columns:
        story.append(Paragraph("Platform Distribution", heading_style))
        counts = df["Detected Platform"].value_counts().head(10)
        rows = [["Platform", "Count"]] + [[k, str(v)] for k, v in counts.items()]
        story.append(_mini_table(rows, col_widths=[8 * cm, 6 * cm]))
        story.append(Spacer(1, 10))

    # Industry distribution
    if "Industry" in df.columns:
        story.append(Paragraph("Industry Distribution", heading_style))
        counts = df["Industry"].value_counts().head(10)
        rows = [["Industry", "Count"]] + [[k, str(v)] for k, v in counts.items()]
        story.append(_mini_table(rows, col_widths=[8 * cm, 6 * cm]))
        story.append(Spacer(1, 10))

    # Top skills
    if "Extracted Skills" in df.columns:
        story.append(PageBreak())
        story.append(Paragraph("Top Skills", heading_style))
        all_skills = df["Extracted Skills"].fillna("").apply(
            lambda s: [x.strip() for x in s.split(",") if x.strip()]
        )
        flat = [s for row in all_skills for s in row]
        if flat:
            counts = pd.Series(flat).value_counts().head(15)
            rows = [["Skill", "Count"]] + [[k, str(v)] for k, v in counts.items()]
            story.append(_mini_table(rows, col_widths=[8 * cm, 6 * cm]))
        else:
            story.append(Paragraph("No skills detected in this dataset.", body_style))
        story.append(Spacer(1, 10))

    # Recommendations (rule-based, generated from the data itself)
    story.append(Paragraph("Recommendations", heading_style))
    recs = []
    avg_q = kpis.get("avg_quality_score", 0)
    if avg_q < 50:
        recs.append("Average profile quality is low. Encourage collection of bios, locations, and job titles.")
    elif avg_q < 75:
        recs.append("Profile quality is moderate. Focus on filling missing website and job title fields.")
    else:
        recs.append("Profile quality is strong overall. Maintain current data collection standards.")
    if kpis.get("duplicate_profiles", 0) > 0:
        recs.append(f"{kpis['duplicate_profiles']} duplicate record(s) detected — review before further use.")
    missing_url_pct = 0
    if "Has Valid Profile URL" in df.columns and len(df):
        missing_url_pct = round((~df["Has Valid Profile URL"]).mean() * 100, 1)
    if missing_url_pct > 20:
        recs.append(f"{missing_url_pct}% of profiles lack a valid profile URL — consider validating source links.")
    for r in recs:
        story.append(Paragraph(f"• {r}", body_style))

    doc.build(story)
    return buf.getvalue()

"""
ProfileBoost AI — Social Profile Data Enrichment Platform
===========================================================
Streamlit entry point. Ties together data_loader, cleaner, enrichment,
classifier, scoring, analytics, and exporter modules.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from modules import analytics, cleaner, data_loader, enrichment, exporter, scoring

APP_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = APP_DIR / "data" / "sample_profiles.csv"
CSS_PATH = APP_DIR / "style.css"

st.set_page_config(
    page_title="ProfileBoost AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
def load_css(path: Path) -> None:
    if path.exists():
        st.markdown(f"<style>{path.read_text()}</style>", unsafe_allow_html=True)


load_css(CSS_PATH)


def section_header(text: str) -> None:
    st.markdown(
        f'<div class="pb-section-header"><span class="pb-dot"></span>{text}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str = "purple") -> str:
    return f'<span class="pb-badge pb-badge-{kind}">{text}</span>'


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
for key, default in {
    "load_result": None,
    "cleaned_df": None,
    "cleaning_summary": None,
    "enriched_df": None,
    "scored_df": None,
    "column_map": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def run_pipeline(load_result: data_loader.LoadResult) -> None:
    """Run cleaning -> enrichment -> scoring and store results in session state."""
    with st.spinner("Cleaning data..."):
        cleaned_df, cleaning_summary = cleaner.clean_dataframe(
            load_result.dataframe, load_result.detected_columns
        )
    with st.spinner("Enriching profiles..."):
        enriched_df = enrichment.enrich_dataframe(cleaned_df, load_result.detected_columns)
    with st.spinner("Scoring profile quality..."):
        scored_df = scoring.score_dataframe(enriched_df, load_result.detected_columns)

    st.session_state.load_result = load_result
    st.session_state.cleaned_df = cleaned_df
    st.session_state.cleaning_summary = cleaning_summary
    st.session_state.enriched_df = enriched_df
    st.session_state.scored_df = scored_df
    st.session_state.column_map = load_result.detected_columns


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✨ ProfileBoost AI")
    st.caption("Social Profile Data Enrichment Platform")
    st.markdown("---")

    st.markdown("#### 1. Upload data")
    uploaded_file = st.file_uploader(
        "CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed"
    )

    use_sample = st.button("Or load sample dataset", use_container_width=True)

    st.markdown("---")
    st.markdown("#### About")
    st.caption(
        "Processes only user-provided or exported profile data. "
        "No scraping, no private data collection, no external AI API calls."
    )

    if st.session_state.scored_df is not None:
        st.markdown("---")
        if st.button("Reset session", use_container_width=True):
            for key in ("load_result", "cleaned_df", "cleaning_summary", "enriched_df", "scored_df", "column_map"):
                st.session_state[key] = None if key != "column_map" else {}
            st.rerun()


# ----------------------------------------------------------------------
# Handle uploads
# ----------------------------------------------------------------------
if uploaded_file is not None and (
    st.session_state.load_result is None
    or st.session_state.load_result.file_name != uploaded_file.name
):
    result = data_loader.load_and_validate(uploaded_file)
    if result.success:
        run_pipeline(result)
    else:
        st.session_state.load_result = result  # keep to show errors below

if use_sample and (
    st.session_state.load_result is None
    or st.session_state.load_result.file_name != "sample_profiles.csv"
):
    result = data_loader.load_sample_data(str(SAMPLE_DATA_PATH))
    if result.success:
        run_pipeline(result)
    else:
        st.session_state.load_result = result


# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown('<div class="pb-hero-title">ProfileBoost AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="pb-hero-subtitle">Clean • Enrich • Analyze • Export</div>',
    unsafe_allow_html=True,
)

load_result: data_loader.LoadResult | None = st.session_state.load_result

# ----------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------
if load_result is None:
    st.markdown(
        """
        <div class="pb-card">
        <b>Welcome!</b> Upload a CSV or Excel file containing publicly provided or
        exported social profile data using the sidebar, or click
        <i>"Or load sample dataset"</i> to explore the platform with 32 sample records.
        <br><br>
        ProfileBoost AI runs entirely locally — no external AI API keys, no scraping,
        no collection of private information.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------------
# Errors / warnings from load step
# ----------------------------------------------------------------------
if not load_result.success:
    for err in load_result.errors:
        st.error(err)
    st.stop()

for warn in load_result.warnings:
    st.warning(warn)

scored_df = st.session_state.scored_df
column_map = st.session_state.column_map
cleaning_summary = st.session_state.cleaning_summary

# ----------------------------------------------------------------------
# File summary
# ----------------------------------------------------------------------
section_header("File Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("File Name", load_result.file_name)
c2.metric("File Size", f"{load_result.file_size_kb:,.1f} KB")
c3.metric("Rows", f"{load_result.n_rows:,}")
c4.metric("Columns", f"{load_result.n_cols}")

with st.expander("Detected & missing fields", expanded=False):
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Detected fields**")
        if column_map:
            st.markdown(
                "".join(
                    badge(f"{k.replace('_', ' ').title()} → {v}", "green") + " "
                    for k, v in column_map.items()
                ),
                unsafe_allow_html=True,
            )
        else:
            st.write("None detected.")
    with dc2:
        st.markdown("**Missing / not detected**")
        missing = load_result.missing_fields
        if missing:
            st.markdown(
                "".join(badge(m.replace("_", " ").title(), "gray") + " " for m in missing),
                unsafe_allow_html=True,
            )
        else:
            st.write("All expected fields detected.")

with st.expander("Preview of uploaded data", expanded=False):
    st.dataframe(load_result.dataframe.head(10), use_container_width=True)


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_dashboard, tab_table, tab_compare, tab_export = st.tabs(
    ["📊 Dashboard", "🗂️ Data Table", "🔁 Before vs After", "⬇️ Export"]
)

# ========================================================================
# DASHBOARD TAB
# ========================================================================
with tab_dashboard:
    section_header("Filters")
    fc1, fc2, fc3 = st.columns(3)
    fc4, fc5, fc6 = st.columns(3)

    def _opts(col):
        return sorted([v for v in scored_df[col].dropna().unique() if str(v).strip()])

    f_platform = fc1.multiselect("Platform", _opts("Detected Platform"))
    f_industry = fc2.multiselect("Industry", _opts("Industry"))
    f_role = fc3.multiselect("Professional Role", _opts("Professional Role"))
    f_seniority = fc4.multiselect("Seniority", _opts("Seniority"))
    f_location = fc5.multiselect("Location", _opts("Location Group"))
    f_quality = fc6.slider("Quality Score range", 0, 100, (0, 100))

    filtered_df = analytics.apply_filters(
        scored_df,
        platform=f_platform or None,
        industry=f_industry or None,
        role=f_role or None,
        seniority=f_seniority or None,
        location=f_location or None,
        quality_range=f_quality,
    )

    if filtered_df.empty:
        st.warning("No profiles match the current filters.")
    else:
        kpis = analytics.kpi_summary(filtered_df)

        section_header("Key Metrics")
        st.markdown('<div class="pb-kpi-grid">', unsafe_allow_html=True)
        kpi_cols = st.columns(5)
        kpi_defs = [
            ("Total Profiles", kpis["total_profiles"], "pb-accent-primary"),
            ("Valid Profiles", kpis["valid_profiles"], "pb-accent-cyan"),
            ("Duplicate Profiles", kpis["duplicate_profiles"], "pb-accent-pink"),
            ("Enriched Profiles", kpis["enriched_profiles"], "pb-accent-green"),
            ("Avg Quality Score", f"{kpis['avg_quality_score']}/100", "pb-accent-yellow"),
        ]
        for col, (label, value, accent_class) in zip(kpi_cols, kpi_defs):
            with col:
                st.markdown(
                    f"""
                    <div class="pb-kpi-card">
                        <div class="pb-kpi-label">{label}</div>
                        <div class="pb-kpi-value {accent_class}">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

        section_header("Analytics")
        row1c1, row1c2 = st.columns(2)
        row1c1.plotly_chart(analytics.chart_by_platform(filtered_df), use_container_width=True)
        row1c2.plotly_chart(analytics.chart_by_industry(filtered_df), use_container_width=True)

        row2c1, row2c2 = st.columns(2)
        row2c1.plotly_chart(analytics.chart_by_role(filtered_df), use_container_width=True)
        row2c2.plotly_chart(analytics.chart_seniority(filtered_df), use_container_width=True)

        row3c1, row3c2 = st.columns(2)
        row3c1.plotly_chart(analytics.chart_location(filtered_df), use_container_width=True)
        row3c2.plotly_chart(analytics.chart_quality_distribution(filtered_df), use_container_width=True)

        st.plotly_chart(analytics.chart_top_skills(filtered_df), use_container_width=True)
        st.plotly_chart(analytics.chart_missing_data(filtered_df, column_map), use_container_width=True)


# ========================================================================
# DATA TABLE TAB
# ========================================================================
with tab_table:
    section_header("Enriched Records")

    original_cols = [c for c in column_map.values() if c in scored_df.columns]
    enriched_cols = [
        c for c in [
            "Extracted First Name", "Extracted Last Name", "Detected Platform",
            "Profile Type", "Professional Role", "Industry", "Seniority",
            "Extracted Skills", "Technology Keywords", "Location Group",
            "Bio Length", "Has Website", "Has Company", "Has Job Title",
            "Has Valid Profile URL", "Profile Completeness", "Profile Quality Score",
        ] if c in scored_df.columns
    ]

    search_term = st.text_input("Search (name, username, bio, skills...)", "")

    display_df = scored_df.copy()
    if search_term.strip():
        term = search_term.strip().lower()
        text_cols = [c for c in display_df.columns if display_df[c].dtype == object]
        mask = display_df[text_cols].apply(
            lambda col: col.astype(str).str.lower().str.contains(term, na=False)
        ).any(axis=1)
        display_df = display_df[mask]

    view_choice = st.radio(
        "Columns to show", ["Original + Enriched", "Original only", "Enriched only"],
        horizontal=True,
    )
    if view_choice == "Original only":
        show_cols = original_cols
    elif view_choice == "Enriched only":
        show_cols = enriched_cols
    else:
        show_cols = original_cols + [c for c in enriched_cols if c not in original_cols]

    st.markdown(
        f"Showing {len(display_df)} of {len(scored_df)} records. "
        f"{badge('Original field', 'gray')} {badge('Enriched / generated field', 'purple')}",
        unsafe_allow_html=True,
    )
    st.dataframe(display_df[show_cols], use_container_width=True, height=480)


# ========================================================================
# BEFORE vs AFTER TAB
# ========================================================================
with tab_compare:
    section_header("Before vs After")

    raw_df = load_result.dataframe
    detected_cols = list(column_map.values())

    def _missing_count(df: pd.DataFrame, cols: list) -> int:
        if not cols:
            return 0
        sub = df[cols]
        return int(sub.apply(lambda s: s.isna() | (s.astype(str).str.strip() == "")).sum().sum())

    before_missing = _missing_count(raw_df, detected_cols)
    after_missing = _missing_count(scored_df, detected_cols)
    exact_dupes = int(scored_df["_is_exact_duplicate"].sum()) if "_is_exact_duplicate" in scored_df else 0
    avg_quality_after = round(scored_df["Profile Quality Score"].mean(), 1) if len(scored_df) else 0

    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown(
            f"""
            <div class="pb-compare-col pb-compare-before">
            <h4>BEFORE</h4>
            <ul>
              <li><b>{before_missing}</b> missing/blank field values</li>
              <li><b>{exact_dupes}</b> duplicate record(s) present</li>
              <li>Inconsistent capitalization, spacing, and URL formats</li>
              <li>{raw_df.shape[1]} raw columns, no enrichment</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bc2:
        st.markdown(
            f"""
            <div class="pb-compare-col pb-compare-after">
            <h4>AFTER</h4>
            <ul>
              <li><b>{after_missing}</b> missing/blank field values (standardized, none deleted)</li>
              <li>Duplicates flagged, not silently removed</li>
              <li>Cleaned & standardized text, URLs, usernames, platforms</li>
              <li>{scored_df.shape[1]} total columns including enrichment</li>
              <li>Average Profile Quality: <b>{avg_quality_after}/100</b></li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_header("Cleaning Operations Performed")
    if cleaning_summary is not None:
        for line in cleaning_summary.operations:
            st.markdown(f"- {line}")

    section_header("Quality Score Improvement")
    bucket_counts = scored_df["Profile Quality Score"].apply(scoring.quality_bucket).value_counts()
    st.bar_chart(bucket_counts)


# ========================================================================
# EXPORT TAB
# ========================================================================
with tab_export:
    section_header("Download Enriched Data")

    exp_c1, exp_c2, exp_c3, exp_c4 = st.columns(4)

    with exp_c1:
        st.download_button(
            "⬇️ cleaned_profiles.csv",
            data=exporter.to_csv_bytes(st.session_state.cleaned_df),
            file_name="cleaned_profiles.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_c2:
        st.download_button(
            "⬇️ enriched_profiles.csv",
            data=exporter.to_csv_bytes(scored_df),
            file_name="enriched_profiles.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with exp_c3:
        st.download_button(
            "⬇️ enriched_profiles.xlsx",
            data=exporter.to_xlsx_bytes(scored_df),
            file_name="enriched_profiles.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with exp_c4:
        kpis_full = analytics.kpi_summary(scored_df)
        pdf_bytes = exporter.build_pdf_report(
            scored_df, column_map, cleaning_summary.operations if cleaning_summary else [], kpis_full
        )
        st.download_button(
            "⬇️ profile_quality_report.pdf",
            data=pdf_bytes,
            file_name="profile_quality_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    section_header("Profile Quality Breakdown (sample)")
    st.caption("Example scoring breakdown for the first record in the current dataset.")
    if len(scored_df):
        first_breakdown = scored_df.iloc[0]["_quality_breakdown"]
        bd_df = pd.DataFrame(first_breakdown)
        bd_df["earned"] = bd_df["earned"].astype(str) + " / " + bd_df["weight"].astype(str)
        st.dataframe(bd_df[["factor", "earned", "passed"]], use_container_width=True)

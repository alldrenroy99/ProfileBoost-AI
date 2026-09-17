"""
enrichment.py
-------------
Builds the derived/enriched columns for each profile record by combining
the cleaner's output with the classifier's rule-based logic. Also computes
a simple TF-IDF based "technology keyword" extraction as a lightweight
local-ML complement to the fixed skills dictionary.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from . import classifier
from .cleaner import is_valid_url

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


ENRICHED_COLUMN_LABELS = {
    "Extracted First Name": "enriched",
    "Extracted Last Name": "enriched",
    "Detected Platform": "enriched",
    "Profile Type": "enriched",
    "Professional Role": "enriched",
    "Industry": "enriched",
    "Seniority": "enriched",
    "Extracted Skills": "enriched",
    "Technology Keywords": "enriched",
    "Location Group": "enriched",
    "Bio Length": "enriched",
    "Has Website": "enriched",
    "Has Company": "enriched",
    "Has Job Title": "enriched",
    "Profile Completeness": "enriched",
}


def _split_name(full_name: str) -> tuple[str, str]:
    if not full_name or not str(full_name).strip():
        return "", ""
    parts = str(full_name).strip().split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _safe_str(val) -> str:
    """Convert a cell value to a clean string, treating NaN/None as empty."""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _location_group(location: str) -> str:
    """Coarse grouping: take the last comma-separated segment as a proxy for country/region."""
    if not location or not str(location).strip():
        return "Unknown"
    parts = [p.strip() for p in str(location).split(",") if p.strip()]
    return parts[-1] if parts else "Unknown"


def _tech_keywords_tfidf(bios: pd.Series, top_n: int = 5) -> List[List[str]]:
    """
    Lightweight local TF-IDF pass over all bios to surface distinctive terms
    per record. This complements (not replaces) the fixed skills dictionary.
    """
    texts = bios.fillna("").astype(str).tolist()
    if not _HAS_SKLEARN or not any(t.strip() for t in texts):
        return [[] for _ in texts]
    try:
        vectorizer = TfidfVectorizer(
            max_features=500, stop_words="english", ngram_range=(1, 2), min_df=1
        )
        matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        results = []
        for row in matrix:
            row_arr = row.toarray().flatten()
            if row_arr.sum() == 0:
                results.append([])
                continue
            top_idx = row_arr.argsort()[::-1][:top_n]
            top_terms = [feature_names[i] for i in top_idx if row_arr[i] > 0]
            results.append(top_terms)
        return results
    except ValueError:
        return [[] for _ in texts]


def enrich_dataframe(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """Return a new dataframe with all enriched columns appended."""
    enriched = df.copy()
    n = len(enriched)

    def col(field_name: str) -> pd.Series:
        c = column_map.get(field_name)
        return enriched[c] if c and c in enriched.columns else pd.Series([""] * n)

    full_names = col("full_name")
    first_names_in = col("first_name")
    last_names_in = col("last_name")
    bios = col("bio")
    job_titles = col("job_title")
    companies = col("company")
    locations = col("location")
    websites = col("website")
    skills_field = col("skills")
    platforms = col("platform")
    profile_urls = col("profile_url")

    ex_first, ex_last, industries, seniorities, profile_types = [], [], [], [], []
    roles, extracted_skills_list, location_groups = [], [], []
    bio_lengths, has_website, has_company, has_job_title = [], [], [], []

    for i in range(n):
        fn_in = _safe_str(first_names_in.iloc[i] if i < len(first_names_in) else "").strip()
        ln_in = _safe_str(last_names_in.iloc[i] if i < len(last_names_in) else "").strip()
        if fn_in or ln_in:
            fn, ln = fn_in, ln_in
        else:
            fn, ln = _split_name(_safe_str(full_names.iloc[i] if i < len(full_names) else ""))
        ex_first.append(fn)
        ex_last.append(ln)

        bio_val = _safe_str(bios.iloc[i] if i < len(bios) else "")
        title_val = _safe_str(job_titles.iloc[i] if i < len(job_titles) else "")
        company_val = _safe_str(companies.iloc[i] if i < len(companies) else "")
        loc_val = _safe_str(locations.iloc[i] if i < len(locations) else "")
        skills_raw = _safe_str(skills_field.iloc[i] if i < len(skills_field) else "")
        website_val = _safe_str(websites.iloc[i] if i < len(websites) else "")

        extracted = classifier.extract_skills(bio_val, title_val, skills_raw)
        extracted_skills_list.append(", ".join(extracted))

        industry, seniority, profile_type, role = classifier.classify_row(
            bio_val, title_val, company_val, extracted
        )
        industries.append(industry)
        seniorities.append(seniority)
        profile_types.append(profile_type)
        roles.append(role)

        location_groups.append(_location_group(loc_val))
        bio_lengths.append(len(bio_val))
        has_website.append(bool(website_val.strip()))
        has_company.append(bool(company_val.strip()))
        has_job_title.append(bool(title_val.strip()))

    enriched["Extracted First Name"] = ex_first
    enriched["Extracted Last Name"] = ex_last
    if len(platforms) == n:
        enriched["Detected Platform"] = platforms.apply(
            lambda v: v if _safe_str(v).strip() else "Unknown"
        )
    else:
        enriched["Detected Platform"] = "Unknown"
    enriched["Profile Type"] = profile_types
    enriched["Professional Role"] = roles
    enriched["Industry"] = industries
    enriched["Seniority"] = seniorities
    enriched["Extracted Skills"] = extracted_skills_list
    enriched["Technology Keywords"] = [
        ", ".join(kws) for kws in _tech_keywords_tfidf(bios if len(bios) == n else pd.Series([""] * n))
    ]
    enriched["Location Group"] = location_groups
    enriched["Bio Length"] = bio_lengths
    enriched["Has Website"] = has_website
    enriched["Has Company"] = has_company
    enriched["Has Job Title"] = has_job_title

    # Has valid profile URL flag (used by scoring too)
    if len(profile_urls) == n:
        enriched["Has Valid Profile URL"] = profile_urls.apply(is_valid_url)
    else:
        enriched["Has Valid Profile URL"] = False

    # Profile completeness (% of core fields populated)
    core_fields_cols = [
        column_map.get(f)
        for f in ("full_name", "username", "platform", "profile_url", "bio",
                   "location", "website", "job_title")
        if column_map.get(f) in enriched.columns
    ]
    if core_fields_cols:
        def _completeness(row):
            filled = sum(1 for c in core_fields_cols if _safe_str(row[c]).strip() != "")
            return round(filled / len(core_fields_cols) * 100, 1)
        enriched["Profile Completeness"] = enriched.apply(_completeness, axis=1)
    else:
        enriched["Profile Completeness"] = 0.0

    return enriched

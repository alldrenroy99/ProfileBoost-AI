"""
cleaner.py
----------
Data cleaning, normalization, and duplicate detection (exact + fuzzy).

Design principle: NEVER silently delete user data. Every cleaning pass
returns a human-readable summary of what was changed, and duplicate
records are flagged (not dropped) unless the caller explicitly requests
their removal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - graceful local fallback
    _HAS_RAPIDFUZZ = False

EMPTY_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "unknown", "#n/a"}

URL_PATTERN = re.compile(
    r"^(https?://)?(www\.)?[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+(/[^\s]*)?$"
)

PLATFORM_ALIASES = {
    "x": "X (Twitter)", "twitter": "X (Twitter)",
    "linkedin": "LinkedIn", "li": "LinkedIn",
    "instagram": "Instagram", "ig": "Instagram",
    "facebook": "Facebook", "fb": "Facebook",
    "github": "GitHub", "gh": "GitHub",
    "tiktok": "TikTok",
    "youtube": "YouTube", "yt": "YouTube",
    "threads": "Threads",
    "medium": "Medium",
    "behance": "Behance",
    "dribbble": "Dribbble",
    "pinterest": "Pinterest",
    "reddit": "Reddit",
}


@dataclass
class CleaningSummary:
    operations: List[str] = field(default_factory=list)
    whitespace_trimmed: int = 0
    text_normalized: int = 0
    urls_normalized: int = 0
    urls_malformed: int = 0
    usernames_normalized: int = 0
    platforms_standardized: int = 0
    empty_values_standardized: int = 0
    exact_duplicates: int = 0
    fuzzy_duplicate_groups: int = 0
    fuzzy_duplicate_records: int = 0
    type_corrections: int = 0
    missing_value_report: Dict[str, float] = field(default_factory=dict)

    def as_list(self) -> List[str]:
        lines = [
            f"Trimmed whitespace on {self.whitespace_trimmed} cell(s)",
            f"Normalized text casing/spacing on {self.text_normalized} cell(s)",
            f"Normalized {self.urls_normalized} profile URL(s); flagged {self.urls_malformed} malformed URL(s)",
            f"Normalized {self.usernames_normalized} username(s)",
            f"Standardized {self.platforms_standardized} platform name(s)",
            f"Standardized {self.empty_values_standardized} empty/placeholder value(s) to blank",
            f"Corrected data types on {self.type_corrections} column(s)",
            f"Found {self.exact_duplicates} exact duplicate row(s)",
            f"Found {self.fuzzy_duplicate_groups} fuzzy-duplicate group(s) "
            f"covering {self.fuzzy_duplicate_records} record(s)"
            + ("" if _HAS_RAPIDFUZZ else " [rapidfuzz not installed - skipped]"),
        ]
        return lines


def _is_empty_token(val) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip().lower() in EMPTY_TOKENS


def _clean_text_cell(val, summary: CleaningSummary) -> str:
    if pd.isna(val):
        return val
    original = str(val)
    cleaned = original.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned != original:
        summary.whitespace_trimmed += 1
    if _is_empty_token(cleaned):
        summary.empty_values_standardized += 1
        return ""
    if cleaned != original:
        summary.text_normalized += 1
    return cleaned


def _normalize_url(val, summary: CleaningSummary) -> str:
    if pd.isna(val) or str(val).strip() == "":
        return val
    original = str(val).strip()
    if original == "":
        return original
    candidate = original
    if not candidate.startswith(("http://", "https://")):
        candidate = "https://" + candidate.lstrip("/")
    if URL_PATTERN.match(candidate):
        summary.urls_normalized += 1
        return candidate
    summary.urls_malformed += 1
    return original  # keep original, just flagged as malformed elsewhere


def _normalize_username(val, summary: CleaningSummary) -> str:
    if pd.isna(val) or str(val).strip() == "":
        return val
    original = str(val).strip()
    cleaned = original.lstrip("@").strip().lower()
    cleaned = re.sub(r"\s+", "", cleaned)
    if cleaned != original:
        summary.usernames_normalized += 1
    return cleaned


def _normalize_platform(val, summary: CleaningSummary) -> str:
    if pd.isna(val) or str(val).strip() == "":
        return val
    key = str(val).strip().lower()
    standardized = PLATFORM_ALIASES.get(key, str(val).strip().title())
    if standardized != str(val).strip():
        summary.platforms_standardized += 1
    return standardized


def is_valid_url(val) -> bool:
    if pd.isna(val) or str(val).strip() == "":
        return False
    candidate = str(val).strip()
    if not candidate.startswith(("http://", "https://")):
        candidate = "https://" + candidate.lstrip("/")
    return bool(URL_PATTERN.match(candidate))


def clean_dataframe(
    df: pd.DataFrame, column_map: Dict[str, str]
) -> Tuple[pd.DataFrame, CleaningSummary]:
    """
    Clean a dataframe in place (on a copy) using the detected column map.
    Returns (cleaned_df, summary). No rows or columns are dropped here.
    """
    summary = CleaningSummary()
    cleaned = df.copy()

    # Generic text cleanup for all detected text-like fields
    text_fields = [
        "full_name", "first_name", "last_name", "bio", "location",
        "company", "job_title", "skills", "email",
    ]
    for field_name in text_fields:
        col = column_map.get(field_name)
        if col and col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(lambda v: _clean_text_cell(v, summary))
            summary.type_corrections += 1

    # URL normalization
    for field_name in ("profile_url", "website"):
        col = column_map.get(field_name)
        if col and col in cleaned.columns:
            cleaned[col] = cleaned[col].apply(lambda v: _normalize_url(v, summary))

    # Username normalization
    col = column_map.get("username")
    if col and col in cleaned.columns:
        cleaned[col] = cleaned[col].apply(lambda v: _normalize_username(v, summary))

    # Platform standardization
    col = column_map.get("platform")
    if col and col in cleaned.columns:
        cleaned[col] = cleaned[col].apply(lambda v: _normalize_platform(v, summary))

    # Missing-value analysis (based on original detected columns)
    detected_cols = [c for c in column_map.values() if c in cleaned.columns]
    if detected_cols:
        missing_pct = {
            c: round(cleaned[c].apply(_is_empty_token).mean() * 100, 1) for c in detected_cols
        }
        summary.missing_value_report = missing_pct

    # Exact duplicate detection (on the set of detected columns, case-insensitive)
    if detected_cols:
        dedup_key = cleaned[detected_cols].apply(
            lambda col_series: col_series.astype(str).str.strip().str.lower()
        )
        exact_dupe_mask = dedup_key.duplicated(keep=False)
        cleaned["_is_exact_duplicate"] = exact_dupe_mask
        summary.exact_duplicates = int(exact_dupe_mask.sum())
    else:
        cleaned["_is_exact_duplicate"] = False

    # Fuzzy duplicate detection using RapidFuzz on name/username/bio composite string
    cleaned["_fuzzy_duplicate_group"] = -1
    if _HAS_RAPIDFUZZ:
        key_cols = [
            column_map.get("full_name"),
            column_map.get("username"),
            column_map.get("bio"),
        ]
        key_cols = [c for c in key_cols if c and c in cleaned.columns]
        if key_cols:
            composite = cleaned[key_cols].astype(str).agg(" ".join, axis=1).str.lower().str.strip()
            n = len(composite)
            group_ids = [-1] * n
            next_group = 0
            # O(n^2) fuzzy comparison - fine for the app's row safety limits;
            # for very large datasets this section is skipped gracefully.
            if n <= 5000:
                values = composite.tolist()
                for i in range(n):
                    if group_ids[i] != -1 or values[i] == "":
                        continue
                    for j in range(i + 1, n):
                        if group_ids[j] != -1 or values[j] == "":
                            continue
                        score = fuzz.token_sort_ratio(values[i], values[j])
                        if score >= 90:
                            if group_ids[i] == -1:
                                group_ids[i] = next_group
                            group_ids[j] = group_ids[i]
                    if group_ids[i] != -1 and group_ids[i] == next_group:
                        next_group += 1
                cleaned["_fuzzy_duplicate_group"] = group_ids
                groups = pd.Series(group_ids)
                valid_groups = groups[groups != -1]
                summary.fuzzy_duplicate_groups = valid_groups.nunique()
                summary.fuzzy_duplicate_records = int((groups != -1).sum())
            else:
                summary.operations.append(
                    "Fuzzy duplicate scan skipped for performance (dataset exceeds 5,000 rows)."
                )
    else:
        summary.operations.append("RapidFuzz not installed - fuzzy duplicate detection skipped.")

    summary.operations = summary.as_list() + summary.operations
    return cleaned, summary

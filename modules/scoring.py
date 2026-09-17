"""
scoring.py
----------
Computes a transparent, explainable 0-100 Profile Quality Score based on
data completeness and validity. Each contributing factor is weighted and
the breakdown is returned alongside the score so the UI can show users
exactly why a profile scored the way it did.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

# Factor -> (column_map field OR enriched column, weight)
# Weights sum to 100.
SCORE_FACTORS: List[Tuple[str, str, int]] = [
    ("Name available", "full_name", 15),
    ("Username available", "username", 10),
    ("Platform available", "platform", 10),
    ("Valid profile URL", "__valid_url__", 15),
    ("Bio available", "bio", 15),
    ("Location available", "location", 10),
    ("Website available", "website", 5),
    ("Job title available", "job_title", 10),
    ("Skills detected", "__skills__", 10),
]


def _field_present(row: pd.Series, column_map: Dict[str, str], field_name: str) -> bool:
    col = column_map.get(field_name)
    if not col or col not in row.index:
        return False
    val = row[col]
    return not pd.isna(val) and str(val).strip() not in ("", "nan")


def score_row(row: pd.Series, column_map: Dict[str, str]) -> Tuple[int, List[Dict]]:
    """Return (score_0_to_100, breakdown_list)."""
    breakdown = []
    total = 0
    for label, key, weight in SCORE_FACTORS:
        if key == "__valid_url__":
            passed = bool(row.get("Has Valid Profile URL", False))
        elif key == "__skills__":
            skills_val = row.get("Extracted Skills", "")
            passed = bool(str(skills_val).strip())
        else:
            passed = _field_present(row, column_map, key)
        earned = weight if passed else 0
        total += earned
        breakdown.append({
            "factor": label,
            "weight": weight,
            "earned": earned,
            "passed": passed,
        })
    return int(round(total)), breakdown


def score_dataframe(df: pd.DataFrame, column_map: Dict[str, str]) -> pd.DataFrame:
    """Append 'Profile Quality Score' and a serialized breakdown column."""
    scored = df.copy()
    scores = []
    breakdowns = []
    for _, row in scored.iterrows():
        score, breakdown = score_row(row, column_map)
        scores.append(score)
        breakdowns.append(breakdown)
    scored["Profile Quality Score"] = scores
    scored["_quality_breakdown"] = breakdowns
    return scored


def quality_bucket(score: int) -> str:
    if score >= 85:
        return "Excellent (85-100)"
    if score >= 70:
        return "Good (70-84)"
    if score >= 50:
        return "Fair (50-69)"
    return "Poor (0-49)"

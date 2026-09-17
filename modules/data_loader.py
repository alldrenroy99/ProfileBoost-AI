"""
data_loader.py
---------------
Handles file ingestion (CSV / XLSX / XLS), basic validation, and automatic
column detection using flexible/fuzzy name matching against a canonical
schema of social-profile fields.

No external AI APIs are used anywhere in this module.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Safety limits (configurable). Raise/lower as needed for your environment.
MAX_FILE_SIZE_MB = 50
MAX_ROWS = 200_000

SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Canonical field -> list of likely header aliases (lower-cased, no punctuation).
CANONICAL_FIELDS: Dict[str, List[str]] = {
    "full_name": ["name", "full name", "fullname", "full_name", "profile name"],
    "first_name": ["first name", "firstname", "first_name", "given name"],
    "last_name": ["last name", "lastname", "last_name", "surname", "family name"],
    "username": ["username", "user name", "handle", "user_id", "screen name", "screenname"],
    "platform": ["platform", "social platform", "network", "source", "site"],
    "profile_url": ["profile url", "profile_url", "url", "link", "profile link", "page url"],
    "bio": ["bio", "about", "description", "summary", "about me", "headline"],
    "location": ["location", "city", "region", "country", "geo", "address"],
    "website": ["website", "web site", "personal website", "site url", "portfolio"],
    "company": ["company", "employer", "organization", "organisation", "workplace"],
    "job_title": ["job title", "title", "role", "position", "occupation", "job_title"],
    "skills": ["skills", "skill set", "skillset", "expertise", "tech stack"],
    "email": ["email", "email address", "e-mail", "contact email"],
}

REQUIRED_MINIMUM_FIELDS = 1  # at least one recognizable field must be present


@dataclass
class LoadResult:
    dataframe: Optional[pd.DataFrame]
    file_name: str = ""
    file_size_kb: float = 0.0
    n_rows: int = 0
    n_cols: int = 0
    detected_columns: Dict[str, str] = field(default_factory=dict)   # canonical -> original col
    missing_fields: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.dataframe is not None and not self.errors


def _normalize_header(col: str) -> str:
    return str(col).strip().lower().replace("_", " ").replace("-", " ")


def detect_columns(columns: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Match dataframe columns against the canonical schema using normalized
    exact/substring matching (no ML dependency required for this step).

    Returns (detected_map, missing_fields) where detected_map is
    {canonical_field: original_column_name}.
    """
    normalized = {col: _normalize_header(col) for col in columns}
    detected: Dict[str, str] = {}

    for canonical, aliases in CANONICAL_FIELDS.items():
        match_found = None
        # 1) exact normalized match first
        for col, norm in normalized.items():
            if norm in aliases:
                match_found = col
                break
        # 2) substring match as a fallback (e.g. "user bio text")
        if match_found is None:
            for col, norm in normalized.items():
                if any(alias in norm for alias in aliases):
                    match_found = col
                    break
        if match_found is not None:
            detected[canonical] = match_found

    missing = [f for f in CANONICAL_FIELDS if f not in detected]
    return detected, missing


def _read_any(uploaded_file) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "uploaded_file")
    lower = name.lower()

    if lower.endswith(".csv"):
        # Try a couple of common encodings before giving up.
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Unable to decode CSV file with common encodings (utf-8, latin-1).")

    if lower.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(uploaded_file)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unable to read Excel file: {exc}") from exc

    raise ValueError(f"Unsupported file type: {name}")


def load_and_validate(uploaded_file) -> LoadResult:
    """
    Main entry point: reads the uploaded file, validates it, and detects
    canonical columns. Never raises -- all failure modes are captured in
    LoadResult.errors so the UI can show friendly messages.
    """
    result = LoadResult(dataframe=None, file_name=getattr(uploaded_file, "name", "file"))

    name = result.file_name.lower()
    if not name.endswith(SUPPORTED_EXTENSIONS):
        result.errors.append(
            f"Unsupported file type '{name}'. Please upload a .csv, .xlsx, or .xls file."
        )
        return result

    # File size check
    try:
        uploaded_file.seek(0, io.SEEK_END)
        size_bytes = uploaded_file.tell()
        uploaded_file.seek(0)
        result.file_size_kb = round(size_bytes / 1024, 1)
        if size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024:
            result.errors.append(
                f"File is too large ({result.file_size_kb / 1024:.1f} MB). "
                f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
            )
            return result
        if size_bytes == 0:
            result.errors.append("The uploaded file is empty.")
            return result
    except Exception:  # noqa: BLE001
        pass  # some file-like objects don't support seek/tell reliably; continue

    # Read
    try:
        df = _read_any(uploaded_file)
    except ValueError as exc:
        result.errors.append(str(exc))
        return result
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"Could not read file: {exc}")
        return result

    if df is None or df.empty:
        result.errors.append("The file contains no data rows.")
        return result

    if len(df.columns) == 0:
        result.errors.append("No columns could be detected in the file.")
        return result

    # Row-count safety limit
    if len(df) > MAX_ROWS:
        result.warnings.append(
            f"Dataset has {len(df):,} rows, which exceeds the safety limit of "
            f"{MAX_ROWS:,}. Only the first {MAX_ROWS:,} rows will be processed."
        )
        df = df.head(MAX_ROWS).copy()

    # Drop fully-empty rows/columns which commonly appear in exported CSVs
    df = df.dropna(how="all").reset_index(drop=True)
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed", na=False)]

    detected, missing = detect_columns(list(df.columns))

    if len(detected) < REQUIRED_MINIMUM_FIELDS:
        result.errors.append(
            "None of the expected profile fields (name, username, platform, bio, etc.) "
            "could be detected in this file. Please check the column headers."
        )
        return result

    result.dataframe = df
    result.n_rows = len(df)
    result.n_cols = len(df.columns)
    result.detected_columns = detected
    result.missing_fields = missing
    return result


def load_sample_data(path: str) -> LoadResult:
    """Load the bundled sample dataset through the same validation path."""
    class _FakeUpload(io.BytesIO):
        def __init__(self, path: str):
            with open(path, "rb") as f:
                super().__init__(f.read())
            self.name = path.split("/")[-1]

    return load_and_validate(_FakeUpload(path))

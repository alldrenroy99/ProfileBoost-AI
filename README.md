# ProfileBoost AI — Social Profile Data Enrichment Platform

A local-first data enrichment and analytics platform for social profile
datasets. Upload a CSV or Excel export, and ProfileBoost AI cleans,
deduplicates, enriches, classifies, and scores each record — then gives
you a full analytics dashboard and downloadable reports.

**No OpenAI API, no paid APIs, no API keys required.** Everything runs
locally using pandas, scikit-learn (TF-IDF), RapidFuzz, and transparent
rule-based logic.

> ⚠️ This tool is designed to process **user-provided or already-exported**
> profile data only. It does not scrape private profiles, bypass
> authentication, or collect private information.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Click **"Or load sample dataset"** in the sidebar to explore the app
immediately with 32 sample profile records, or upload your own CSV/XLSX file.

## What it does

1. **Upload & validate** — CSV/XLSX/XLS with size and row safety limits.
2. **Automatic column detection** — flexible matching against a canonical
   schema (name, username, platform, bio, location, etc.).
3. **Data cleaning** — whitespace trimming, text normalization, URL/username/
   platform standardization, missing-value analysis. Nothing is silently
   deleted.
4. **Duplicate detection** — exact matches plus fuzzy matching via RapidFuzz.
5. **Enrichment** — derived fields like extracted skills, industry, seniority,
   profile type, location group, and TF-IDF technology keywords.
6. **Profile Quality Score (0–100)** — transparent, weighted, with a visible
   factor-by-factor breakdown.
7. **Analytics dashboard** — KPI cards, Plotly charts, interactive filters.
8. **Before vs. After** — see exactly what changed and improved.
9. **Export** — cleaned CSV, enriched CSV/XLSX, and a PDF quality report.

## Project structure

```
ProfileBoostAI/
├── app.py                  # Streamlit entry point
├── style.css                # Design system (dark, glassmorphism)
├── requirements.txt
├── modules/
│   ├── data_loader.py       # File ingestion + column detection
│   ├── cleaner.py           # Cleaning + duplicate detection
│   ├── enrichment.py        # Derived/enriched columns
│   ├── classifier.py        # Skills, industry, seniority, profile type
│   ├── scoring.py           # Profile Quality Score
│   ├── analytics.py         # Dashboard charts + filters
│   └── exporter.py          # CSV / XLSX / PDF export
├── data/
│   └── sample_profiles.csv  # 32 fictional sample records
└── reports/                 # (scratch space; exports are generated in-memory)
```

## Extending the skills dictionary

Add new skills to `SKILLS_DICTIONARY` in `modules/classifier.py` — the
extraction logic automatically picks up new entries.

## Notes

- If `rapidfuzz` isn't installed, fuzzy duplicate detection is skipped
  gracefully (exact duplicate detection still works).
- The fuzzy duplicate pass is O(n²) and automatically skips itself on
  datasets larger than 5,000 rows to keep the UI responsive.

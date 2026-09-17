"""
classifier.py
-------------
Rule-based, fully local classification of profiles:
 - skill extraction from a configurable dictionary
 - professional/industry category
 - approximate seniority
 - profile type (Professional / Personal / Business / Unknown)

No external AI API is used. All logic is transparent keyword/regex matching,
which keeps results explainable and easy to extend.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------
# 1. Configurable skills dictionary - easy to expand.
# ----------------------------------------------------------------------
SKILLS_DICTIONARY: List[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "SQL", "HTML", "CSS", "React", "Vue", "Angular", "Node.js", "Django",
    "Flask", "FastAPI", "Streamlit", "Pandas", "NumPy", "Scikit-learn",
    "TensorFlow", "PyTorch", "Keras", "NLP", "Machine Learning",
    "Deep Learning", "Computer Vision", "Data Science", "Data Analytics",
    "Data Engineering", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "Git", "GitHub", "CI/CD", "DevOps", "Excel", "Tableau", "Power BI",
    "Figma", "Photoshop", "Illustrator", "UX Design", "UI Design",
    "SEO", "Content Marketing", "Digital Marketing", "Copywriting",
    "Salesforce", "HubSpot", "Product Management", "Project Management",
    "Agile", "Scrum", "Recruiting", "HR Management",
]
# Normalized lookup for matching (lowercase -> canonical display form)
_SKILLS_LOOKUP = {s.lower(): s for s in SKILLS_DICTIONARY}


def extract_skills(*texts: str) -> List[str]:
    """Extract known skills from one or more free-text fields (bio, job title, skills list)."""
    combined = " ".join(t for t in texts if isinstance(t, str)).lower()
    if not combined.strip():
        return []
    found = []
    for skill_lower, skill_display in _SKILLS_LOOKUP.items():
        # word-boundary-ish match; tolerate "c++"/"c#" and "node.js" specially
        pattern = re.escape(skill_lower)
        if re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", combined):
            found.append(skill_display)
    return sorted(set(found))


# ----------------------------------------------------------------------
# 2. Professional / industry classification
# ----------------------------------------------------------------------
INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "AI / ML": ["ai", "artificial intelligence", "machine learning", "deep learning",
                "nlp", "computer vision", "ml engineer", "data scientist"],
    "Software Development": ["software engineer", "developer", "programmer", "full stack",
                              "backend", "frontend", "web developer", "mobile developer",
                              "software development"],
    "Data Science": ["data scientist", "data science", "statistician", "research scientist"],
    "Data Analytics": ["data analyst", "business intelligence", "bi analyst", "analytics"],
    "Marketing": ["marketing", "seo", "content strategist", "growth", "brand manager",
                  "social media manager"],
    "Sales": ["sales", "account executive", "business development", "bdr", "sdr"],
    "Finance": ["finance", "accountant", "financial analyst", "investment", "banking"],
    "HR / Recruitment": ["hr", "human resources", "recruiter", "talent acquisition",
                          "people operations"],
    "Design": ["designer", "ux", "ui", "graphic design", "product design", "illustrator"],
    "Education": ["teacher", "professor", "educator", "instructor", "tutor", "lecturer"],
    "Healthcare": ["nurse", "doctor", "physician", "healthcare", "medical", "clinician"],
    "Business": ["founder", "ceo", "entrepreneur", "consultant", "operations manager",
                 "business owner"],
    "Student": ["student", "undergraduate", "graduate student", "intern"],
}

SENIORITY_KEYWORDS: Dict[str, List[str]] = {
    "Intern": ["intern", "internship", "trainee"],
    "Junior": ["junior", "jr.", "jr ", "entry level", "entry-level", "associate"],
    "Mid-Level": ["mid level", "mid-level", "engineer ii", "analyst ii"],
    "Senior": ["senior", "sr.", "sr ", "principal"],
    "Lead": ["lead ", "team lead", "tech lead", "staff engineer"],
    "Manager": ["manager", "head of", "director"],
    "Executive": ["chief", "ceo", "cto", "cfo", "coo", "founder", "vp ", "vice president",
                  "executive"],
}

PROFILE_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "Business": ["official", "company", "brand", "store", "shop", "inc.", "llc", "team"],
    "Professional": ["engineer", "manager", "developer", "consultant", "analyst",
                      "designer", "founder", "scientist", "specialist"],
}


def _match_category(text: str, keyword_map: Dict[str, List[str]]) -> str:
    if not text or not text.strip():
        return "Unknown"
    lowered = text.lower()
    best_category, best_hits = "Unknown", 0
    for category, keywords in keyword_map.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best_hits:
            best_category, best_hits = category, hits
    return best_category if best_hits > 0 else "Unknown"


def classify_industry(bio: str, job_title: str, skills: List[str]) -> str:
    combined = f"{job_title or ''} {bio or ''} {' '.join(skills)}"
    industry = _match_category(combined, INDUSTRY_KEYWORDS)
    if industry == "Unknown" and any(
        s in ("Python", "SQL", "Machine Learning", "TensorFlow", "PyTorch") for s in skills
    ):
        return "AI / ML" if "Machine Learning" in skills or "TensorFlow" in skills else "Software Development"
    return industry


def classify_seniority(job_title: str, bio: str) -> str:
    combined = f"{job_title or ''} {bio or ''}".lower()
    return _match_category(combined, SENIORITY_KEYWORDS) or "Unknown"


def classify_profile_type(bio: str, job_title: str, company: str) -> str:
    combined = f"{bio or ''} {job_title or ''} {company or ''}"
    profile_type = _match_category(combined, PROFILE_TYPE_KEYWORDS)
    if profile_type == "Unknown":
        return "Personal" if (bio or "").strip() else "Unknown"
    return profile_type


def professional_role(job_title: str) -> str:
    """Return a cleaned-up display version of the job title, or 'Unknown'."""
    if not job_title or not str(job_title).strip():
        return "Unknown"
    return str(job_title).strip().title()


def classify_row(bio: str, job_title: str, company: str, skills: List[str]) -> Tuple[str, str, str, str]:
    """Convenience wrapper returning (industry, seniority, profile_type, role)."""
    industry = classify_industry(bio, job_title, skills)
    seniority = classify_seniority(job_title, bio)
    profile_type = classify_profile_type(bio, job_title, company)
    role = professional_role(job_title)
    return industry, seniority, profile_type, role

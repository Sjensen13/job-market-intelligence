"""
Streamlit job board: load from SQLite, filter, and link out to original postings.
"""

from __future__ import annotations

import ast
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_db() -> Path:
    return _project_root() / "data" / "jobs.db"


def _parse_tags(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s or s == "[]":
        return []
    try:
        val = json.loads(s)
    except json.JSONDecodeError:
        try:
            val = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    return []


def _effective_salary(row: pd.Series) -> float | None:
    for key in ("salary_avg", "salary_min", "salary_max"):
        v = row.get(key)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            try:
                f = float(v)
                if f > 0:
                    return f
            except (TypeError, ValueError):
                continue
    return None


def _normalize_remote(val: object) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "unknown"
    s = str(val).strip().lower()
    return s if s else "unknown"


def _remote_category(norm: str) -> str:
    if "remote" in norm and "non-remote" not in norm:
        return "remote"
    if norm == "hybrid":
        return "hybrid"
    if norm in ("on-site", "onsite", "on site", "office", "in-office", "in office"):
        return "on-site"
    return "other"


@st.cache_data(ttl=60)
def load_jobs(db_path: str) -> pd.DataFrame:
    p = Path(db_path)
    if not p.is_file():
        return pd.DataFrame()
    conn = sqlite3.connect(p)
    try:
        df = pd.read_sql_query(
            """
            SELECT
                title,
                company,
                location,
                remote_status,
                salary_min,
                salary_max,
                salary_avg,
                tags,
                posted_date,
                job_url
            FROM jobs
            ORDER BY posted_date IS NULL, posted_date DESC, id DESC
            """,
            conn,
        )
    finally:
        conn.close()
    return df


def _apply_filters(
    df: pd.DataFrame,
    *,
    title_kw: str,
    salary_min: float | None,
    remote_choice: str,
    skill_kw: str,
    locations: list[str],
) -> pd.DataFrame:
    out = df.copy()
    if title_kw.strip():
        mask = out["title"].fillna("").str.contains(title_kw.strip(), case=False, regex=False)
        out = out.loc[mask]
    if salary_min is not None and salary_min > 0:
        eff = out.apply(_effective_salary, axis=1)
        out = out.loc[eff.notna() & (eff >= float(salary_min))]
    if remote_choice != "All":
        norm = out["remote_status"].map(_normalize_remote)
        cat = norm.map(_remote_category)
        if remote_choice == "Remote":
            out = out.loc[cat == "remote"]
        elif remote_choice == "Hybrid":
            out = out.loc[cat == "hybrid"]
        elif remote_choice == "On-site":
            out = out.loc[cat == "on-site"]
        elif remote_choice == "Other / unspecified":
            out = out.loc[cat == "other"]
    if skill_kw.strip():
        kw = skill_kw.strip().lower()

        def row_matches_skill(row: pd.Series) -> bool:
            tags_l = " ".join(x.lower() for x in _parse_tags(row.get("tags")))
            if kw in tags_l:
                return True
            title_l = str(row.get("title") or "").lower()
            return kw in title_l

        out = out.loc[out.apply(row_matches_skill, axis=1)]
    if locations:
        out = out.loc[out["location"].fillna("").isin(locations)]
    return out


def _safe_http_url_href(val: object) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    if s.startswith("https://") or s.startswith("http://"):
        return html.escape(s, quote=True)
    return None


def _render_jobs_html_table(df: pd.DataFrame) -> None:
    """
    Render jobs without ``st.dataframe`` so a broken conda NumPy / PyArrow stack
    (``numpy.core.multiarray failed to import``) does not block the UI.
    """
    if df.empty:
        st.info("No rows match the current filters.")
        return

    th = (
        "<thead><tr>"
        "<th>Title</th><th>Company</th><th>Location</th><th>Work mode</th>"
        "<th>Salary</th><th>Tags</th><th>Posted</th><th>Apply</th>"
        "</tr></thead>"
    )
    rows_html: list[str] = []
    for _, row in df.iterrows():
        href = _safe_http_url_href(row.get("job_url"))
        if href:
            apply_cell = (
                f'<a href="{href}" target="_blank" rel="noopener noreferrer">Open posting</a>'
            )
        else:
            apply_cell = "—"
        sal = row.get("salary_display")
        if sal is None or (isinstance(sal, float) and pd.isna(sal)):
            sal_s = "—"
        else:
            sal_s = html.escape(str(sal))
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('title') or ''))}</td>"
            f"<td>{html.escape(str(row.get('company') or ''))}</td>"
            f"<td>{html.escape(str(row.get('location') or ''))}</td>"
            f"<td>{html.escape(str(row.get('remote_status') or ''))}</td>"
            f"<td>{sal_s}</td>"
            f"<td>{html.escape(str(row.get('tags') or ''))}</td>"
            f"<td>{html.escape(str(row.get('posted_date') or ''))}</td>"
            f"<td>{apply_cell}</td>"
            "</tr>"
        )

    table = (
        "<style>"
        ".jm-jobs { border-collapse: collapse; width: 100%; font-size: 0.9rem; }"
        ".jm-jobs th, .jm-jobs td { border: 1px solid rgba(128,128,128,0.35); padding: 6px 8px; vertical-align: top; }"
        ".jm-jobs th { background: rgba(128,128,128,0.12); text-align: left; }"
        ".jm-jobs tr:nth-child(even) { background: rgba(128,128,128,0.06); }"
        "</style>"
        f'<table class="jm-jobs">{th}<tbody>{"".join(rows_html)}</tbody></table>'
    )
    st.markdown(table, unsafe_allow_html=True)


def _top_skills(df: pd.DataFrame, n: int = 8) -> list[tuple[str, int]]:
    c: Counter[str] = Counter()
    for raw in df["tags"].dropna():
        for t in _parse_tags(raw):
            c[t.lower()] += 1
    top = c.most_common(n)
    return top


def main() -> None:
    st.set_page_config(page_title="Job market intelligence", layout="wide")
    st.title("Job listings")

    default_db = _default_db()
    db_input = st.sidebar.text_input("Database path", value=str(default_db))
    df_all = load_jobs(db_input)

    if df_all.empty:
        st.warning(
            f"No rows loaded. Check that the database exists and contains a `jobs` table: `{default_db}`"
        )
        return

    st.sidebar.header("Filters")
    title_kw = st.sidebar.text_input("Job title keyword", placeholder="e.g. scientist")
    salary_floor = st.sidebar.number_input(
        "Minimum salary (USD)",
        min_value=0,
        value=0,
        step=5000,
        help="Uses salary_avg, then salary_min, then salary_max when present.",
    )
    remote_choice = st.sidebar.selectbox(
        "Remote / on-site",
        ("All", "Remote", "Hybrid", "On-site", "Other / unspecified"),
    )
    loc_options = sorted(df_all["location"].dropna().astype(str).unique().tolist(), key=str.lower)
    locations = st.sidebar.multiselect("Location", options=loc_options, default=[])
    skill_kw = st.sidebar.text_input(
        "Skill tag (substring)",
        placeholder="e.g. python — also searches title",
        help="Matches parsed tag list, or any word in the job title when tags are empty.",
    )

    salary_min_val = float(salary_floor) if salary_floor and salary_floor > 0 else None
    filtered = _apply_filters(
        df_all,
        title_kw=title_kw,
        salary_min=salary_min_val,
        remote_choice=remote_choice,
        skill_kw=skill_kw,
        locations=locations,
    )

    eff_salaries = filtered.apply(_effective_salary, axis=1).dropna()
    avg_sal = float(eff_salaries.mean()) if not eff_salaries.empty else None
    norm_remote = filtered["remote_status"].map(_normalize_remote)
    remote_cat = norm_remote.map(_remote_category)
    n_remote_strict = int((remote_cat == "remote").sum())
    n_hybrid = int((remote_cat == "hybrid").sum())

    top_skills = _top_skills(filtered)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jobs (filtered)", f"{len(filtered):,}")
    m2.metric(
        "Avg. salary (filtered)",
        f"${avg_sal:,.0f}" if avg_sal is not None else "—",
    )
    m3.metric("Remote jobs", f"{n_remote_strict:,}")
    if top_skills:
        k0, v0 = top_skills[0]
        m4.metric("Most common skill (tags)", k0, delta=f"{v0} jobs")
    else:
        m4.metric("Most common skill (tags)", "—")

    sublines: list[str] = []
    if n_hybrid:
        sublines.append(f"{n_hybrid:,} hybrid role(s) in this slice (not counted as remote).")
    if len(top_skills) > 1:
        rest = ", ".join(f"{k} ({v})" for k, v in top_skills[1:])
        sublines.append(f"Other common tags: {rest}")
    if sublines:
        st.caption(" ".join(sublines))

    display = filtered.assign(
        salary_display=filtered.apply(
            lambda r: (
                f"${_effective_salary(r):,.0f}"
                if _effective_salary(r) is not None
                else "—"
            ),
            axis=1,
        )
    )
    show_cols = [
        "title",
        "company",
        "location",
        "remote_status",
        "salary_display",
        "tags",
        "posted_date",
        "job_url",
    ]
    display = display[[c for c in show_cols if c in display.columns]]

    st.subheader("Results")
    _render_jobs_html_table(display)


if __name__ == "__main__":
    main()
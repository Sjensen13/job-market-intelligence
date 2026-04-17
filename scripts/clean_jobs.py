#!/usr/bin/env python3
"""
Load raw job CSV, deduplicate, normalize salaries and text, infer work mode,
and add keyword-based tags from descriptions.
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _title_column(df: pd.DataFrame) -> str:
    if "job_title" in df.columns:
        return "job_title"
    if "title" in df.columns:
        return "title"
    raise ValueError("Expected a title column named 'job_title' or 'title'.")


def drop_duplicates_smart(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer job_id when present; otherwise title + company + location."""
    out = df.copy()
    title_col = _title_column(out)
    for col in ("company", "location", title_col):
        if col not in out.columns:
            out[col] = np.nan

    jid = out["job_id"] if "job_id" in out.columns else pd.Series(np.nan, index=out.index)
    jid_str = jid.astype("string")
    has_id = jid_str.notna() & (jid_str.str.strip() != "")

    comp = (
        out[title_col].astype("string").fillna("").str.strip()
        + "\x00"
        + out["company"].astype("string").fillna("").str.strip()
        + "\x00"
        + out["location"].astype("string").fillna("").str.strip()
    )
    dedup_key = np.where(has_id.to_numpy(), "id:" + jid_str.fillna("").astype(str), "cmp:" + comp)
    out["_dedup_key"] = dedup_key
    out = out.drop_duplicates(subset=["_dedup_key"], keep="first").drop(columns=["_dedup_key"])
    return out.reset_index(drop=True)


_TAG_RULES: list[tuple[str, list[re.Pattern[str]]]] = [
    ("Python", [re.compile(r"\bpython\b", re.I)]),
    ("SQL", [re.compile(r"\bsql\b", re.I)]),
    ("Excel", [re.compile(r"\bexcel\b", re.I)]),
    (
        "Deep Learning",
        [
            re.compile(r"\bpytorch\b", re.I),
            re.compile(r"\btensorflow\b", re.I),
            re.compile(r"\bkeras\b", re.I),
        ],
    ),
    (
        "BI",
        [
            re.compile(r"\btableau\b", re.I),
            re.compile(r"\bpower\s*bi\b", re.I),
            re.compile(r"\bpowerbi\b", re.I),
            re.compile(r"\blooker\b", re.I),
        ],
    ),
    (
        "Cloud",
        [
            re.compile(r"\baws\b", re.I),
            re.compile(r"\bazure\b", re.I),
            re.compile(r"\bgcp\b", re.I),
            re.compile(r"\bgoogle\s+cloud\b", re.I),
        ],
    ),
]

_BOOL_COL = {
    "Python": "has_python",
    "SQL": "has_sql",
    "Excel": "has_excel",
    "Deep Learning": "has_deep_learning",
    "BI": "has_bi",
    "Cloud": "has_cloud",
}


def _strip_html_and_whitespace(text: str) -> str:
    text = html_lib.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\xa0\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def clean_text_field(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .fillna("")
        .map(lambda s: re.sub(r"\s+", " ", str(s).strip()) if s is not None else "")
        .replace("", pd.NA)
    )


def clean_description(series: pd.Series) -> pd.Series:
    def _one(val: object) -> object:
        if pd.isna(val):
            return pd.NA
        s = str(val)
        if not s.strip():
            return pd.NA
        return _strip_html_and_whitespace(s)

    return series.map(_one)


def standardize_salaries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("salary_min", "salary_max"):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    mn = out["salary_min"]
    mx = out["salary_max"]
    out["salary_avg"] = np.where(
        mn.notna() & mx.notna(),
        (mn + mx) / 2.0,
        np.where(mn.notna(), mn, np.where(mx.notna(), mx, np.nan)),
    )
    return out


def infer_work_mode(row_title: str, row_location: str, row_description: str) -> str:
    blob = f"{row_title} {row_location} {row_description}".lower()

    if re.search(r"\bhybrid\b", blob):
        return "hybrid"
    if re.search(r"\bpartial(?:ly)?\s+remote\b", blob):
        return "hybrid"
    if re.search(r"\b\d+\s*[-–]\s*\d+\s*days?\b", blob) and re.search(
        r"\b(remote|office|on[- ]?site)\b",
        blob,
    ):
        return "hybrid"

    if re.search(
        r"\b(remote|work\s+from\s+home|wfh|fully\s+remote|100%\s+remote|distributed\s+team)\b",
        blob,
    ):
        return "remote"

    if re.search(r"\b(on[- ]?site|in[- ]?office|office[- ]?based)\b", blob):
        return "on-site"

    if re.search(r"\bremote\b", blob):
        return "remote"

    return "unknown"


def add_tags(df: pd.DataFrame, desc_series: pd.Series) -> pd.DataFrame:
    out = df.copy()
    desc_filled = desc_series.fillna("").astype(str)

    tags_list: list[list[str]] = []
    bool_rows: dict[str, list[bool]] = {c: [] for c in _BOOL_COL.values()}
    for text in desc_filled:
        lower = text.lower()
        row_tags: list[str] = []
        for label, patterns in _TAG_RULES:
            hit = any(p.search(lower) for p in patterns)
            bool_rows[_BOOL_COL[label]].append(hit)
            if hit:
                row_tags.append(label)
        tags_list.append(row_tags)

    for _label, col in _BOOL_COL.items():
        out[col] = bool_rows[col]

    out["tags"] = [json.dumps(t) for t in tags_list]
    return out


def add_work_mode_column(df: pd.DataFrame, title_col: str) -> pd.DataFrame:
    out = df.copy()
    titles = out[title_col].fillna("").astype(str)
    locs = out["location"].fillna("").astype(str) if "location" in out.columns else pd.Series("", index=out.index)
    descs = out["description"].fillna("").astype(str) if "description" in out.columns else pd.Series("", index=out.index)
    out["work_mode"] = [
        infer_work_mode(t, loc, d) for t, loc, d in zip(titles, locs, descs, strict=False)
    ]
    return out


def clean_jobs(df: pd.DataFrame) -> pd.DataFrame:
    title_col = _title_column(df)
    out = df.copy()

    for col in (title_col, "company", "location"):
        if col in out.columns:
            out[col] = clean_text_field(out[col])

    if "description" in out.columns:
        out["description"] = clean_description(out["description"])

    out = drop_duplicates_smart(out)

    out = standardize_salaries(out)

    # Tags use cleaned description text
    desc_for_tags = out["description"] if "description" in out.columns else pd.Series("", index=out.index)
    out = add_tags(out, desc_for_tags)

    out = add_work_mode_column(out, title_col)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean and enrich raw job listings CSV.")
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=_project_root() / "data" / "raw_jobs.csv",
        help="Input CSV path (default: data/raw_jobs.csv).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=_project_root() / "data" / "clean_jobs.csv",
        help="Output CSV path (default: data/clean_jobs.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.in_path.is_file():
        print(f"Input not found: {args.in_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.in_path)
    cleaned = clean_jobs(df)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out_path, index=False)
    print(f"Wrote {len(cleaned)} rows → {args.out_path}")


if __name__ == "__main__":
    main()

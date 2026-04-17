#!/usr/bin/env python3
"""
Load cleaned job CSV into SQLite: insert new rows, skip duplicates on
(source, api_job_id) or (source, job_url) per init_jobs_db.py indexes.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from init_jobs_db import init_db


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _title_column(df: pd.DataFrame) -> str:
    if "job_title" in df.columns:
        return "job_title"
    if "title" in df.columns:
        return "title"
    raise ValueError("Expected a title column named 'job_title' or 'title'.")


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and val != val:  # NaN
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _str_or_none(val: object) -> str | None:
    if _is_missing(val):
        return None
    s = str(val).strip()
    return s if s else None


def _float_or_none(val: object) -> float | None:
    if _is_missing(val):
        return None
    return float(val)


def _int_or_none(val: object) -> int | None:
    if _is_missing(val):
        return None
    return int(val)


def rows_from_dataframe(df: pd.DataFrame, *, source: str) -> list[tuple]:
    title_col = _title_column(df)
    out: list[tuple] = []
    for _, row in df.iterrows():
        out.append(
            (
                _str_or_none(row.get(title_col)),
                _str_or_none(row.get("company")),
                _str_or_none(row.get("location")),
                _str_or_none(row.get("work_mode")) if "work_mode" in df.columns else None,
                _float_or_none(row.get("salary_min")),
                _float_or_none(row.get("salary_max")),
                _float_or_none(row.get("salary_avg")),
                _str_or_none(row.get("tags")),
                _str_or_none(row.get("posting_date"))
                if "posting_date" in df.columns
                else _str_or_none(row.get("posted_date")),
                _str_or_none(row.get("description")),
                _str_or_none(row.get("job_url")),
                source,
                _int_or_none(row.get("job_id")),
            )
        )
    return out


INSERT_SQL = """
INSERT OR IGNORE INTO jobs (
    title,
    company,
    location,
    remote_status,
    salary_min,
    salary_max,
    salary_avg,
    tags,
    posted_date,
    description,
    job_url,
    source,
    api_job_id
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Insert cleaned jobs into SQLite (skip duplicates).")
    p.add_argument(
        "--csv",
        type=Path,
        default=_project_root() / "data" / "clean_jobs.csv",
        help="Cleaned CSV path (default: data/clean_jobs.csv).",
    )
    p.add_argument(
        "--db",
        type=Path,
        default=_project_root() / "data" / "jobs.db",
        help="SQLite database path (default: data/jobs.db).",
    )
    p.add_argument(
        "--source",
        default="adzuna",
        help="Value for the source column (default: adzuna).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.is_file():
        print(f"CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if df.empty:
        print("No rows in CSV; nothing to do.")
        return

    try:
        init_db(args.db)
    except OSError as e:
        print(f"Failed to initialize database: {e}", file=sys.stderr)
        sys.exit(1)

    rows = rows_from_dataframe(df, source=args.source)
    conn = sqlite3.connect(args.db)
    try:
        count_before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.executemany(INSERT_SQL, rows)
        conn.commit()
        count_after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()

    inserted = count_after - count_before
    skipped = len(rows) - inserted
    print(
        f"Read {len(rows)} rows from {args.csv}. "
        f"Inserted {inserted}, skipped {skipped} (duplicates or conflicts). "
        f"Total rows in jobs: {count_after}."
    )


if __name__ == "__main__":
    main()

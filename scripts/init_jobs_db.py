#!/usr/bin/env python3
"""
Create SQLite jobs.db with the jobs table and uniqueness for deduplication.

Uniqueness: (source, api_job_id) when the API provides an id (e.g. Adzuna),
and (source, job_url) when there is no api_job_id, so refetches with changed
URL query parameters still map to one row when api_job_id is set.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    company TEXT,
    location TEXT,
    remote_status TEXT,
    salary_min REAL,
    salary_max REAL,
    salary_avg REAL,
    tags TEXT,
    posted_date TEXT,
    description TEXT,
    job_url TEXT,
    source TEXT NOT NULL DEFAULT 'adzuna',
    fetched_at TEXT NOT NULL DEFAULT (datetime('now')),
    api_job_id INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_source_api_job_id
    ON jobs(source, api_job_id)
    WHERE api_job_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS ux_jobs_source_job_url
    ON jobs(source, job_url)
    WHERE api_job_id IS NULL AND job_url IS NOT NULL AND trim(job_url) != '';
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize SQLite jobs database.")
    p.add_argument(
        "--db",
        type=Path,
        default=_project_root() / "data" / "jobs.db",
        help="Path to SQLite file (default: data/jobs.db).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        init_db(args.db)
    except OSError as e:
        print(f"Failed to create database: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Initialized schema → {args.db}")


if __name__ == "__main__":
    main()

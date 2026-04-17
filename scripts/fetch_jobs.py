#!/usr/bin/env python3
"""
Fetch job listings from the Adzuna API, normalize to a table, and save to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Adzuna US search appears to cap results_per_page at 50.
MAX_PER_PAGE = 50
DEFAULT_BASE_URL = "https://api.adzuna.com/v1/api"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_credentials() -> tuple[str, str]:
    load_dotenv(_project_root() / ".env")
    app_id = os.environ.get("API_ID")
    app_key = os.environ.get("API_KEY")
    if not app_id or not app_key:
        print(
            "Missing API_ID or API_KEY. Set them in .env or the environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    return app_id, app_key


def _company_name(company: object) -> str | None:
    if isinstance(company, dict):
        return company.get("display_name")
    return None


def _location_label(location: object) -> str | None:
    if not isinstance(location, dict):
        return None
    display = location.get("display_name")
    if display:
        return display
    area = location.get("area")
    if isinstance(area, list) and area:
        return ", ".join(str(a) for a in area if a)
    return None


def normalize_jobs(raw_results: list[dict]) -> pd.DataFrame:
    rows = []
    for job in raw_results:
        rows.append(
            {
                "job_id": job.get("id"),
                "job_title": job.get("title"),
                "company": _company_name(job.get("company")),
                "location": _location_label(job.get("location")),
                "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"),
                "description": job.get("description"),
                "posting_date": job.get("created"),
                "job_url": job.get("redirect_url"),
            }
        )
    return pd.DataFrame(rows)


def fetch_jobs(
    app_id: str,
    app_key: str,
    *,
    keyword: str,
    where: str,
    limit: int,
    country: str,
    base_url: str,
    timeout: float,
) -> list[dict]:
    collected: list[dict] = []
    page = 1
    while len(collected) < limit:
        remaining = limit - len(collected)
        batch = min(MAX_PER_PAGE, remaining)
        url = f"{base_url}/jobs/{country}/search/{page}"
        response = requests.get(
            url,
            params={
                "app_id": app_id,
                "app_key": app_key,
                "results_per_page": batch,
                "what": keyword,
                "where": where,
            },
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        batch_results = payload.get("results") or []
        if not batch_results:
            break
        collected.extend(batch_results)
        if len(batch_results) < batch:
            break
        page += 1
    return collected[:limit]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch jobs from Adzuna and save raw outputs.")
    p.add_argument("--keyword", default="data scientist", help="Search keywords (Adzuna `what`).")
    p.add_argument(
        "--where",
        default="California",
        help='Location string (Adzuna `where`), e.g. "California" or "remote".',
    )
    p.add_argument("--limit", type=int, default=50, help="Max jobs to fetch (paginates past 50).")
    p.add_argument("--country", default="us", help="Adzuna country code, e.g. us, gb.")
    p.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="API base URL (default Adzuna v1).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_project_root() / "data",
        help="Directory for raw_jobs.csv and raw_jobs.json.",
    )
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app_id, app_key = load_credentials()

    try:
        raw_results = fetch_jobs(
            app_id,
            app_key,
            keyword=args.keyword,
            where=args.where,
            limit=args.limit,
            country=args.country,
            base_url=args.base_url.rstrip("/"),
            timeout=args.timeout,
        )
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        print(f"Adzuna API returned HTTP {status}.", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Request to Adzuna failed ({type(e).__name__}). Check network and credentials.", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "raw_jobs.json"
    csv_path = args.out_dir / "raw_jobs.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    df = normalize_jobs(raw_results)
    df.to_csv(csv_path, index=False)

    print(
        f"Fetched {len(raw_results)} jobs → {csv_path} and {json_path} "
        f"(keyword={args.keyword!r}, where={args.where!r}, limit={args.limit})."
    )


if __name__ == "__main__":
    main()

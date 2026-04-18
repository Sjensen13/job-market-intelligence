#!/usr/bin/env python3
"""
Fetch job listings from the Adzuna API, normalize to a table, and save to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Adzuna US search appears to cap results_per_page at 50.
MAX_PER_PAGE = 50
DEFAULT_BASE_URL = "https://api.adzuna.com/v1/api"

# Default mix of roles/industries so the board is not only “data scientist” results.
DEFAULT_DIVERSE_KEYWORDS: tuple[str, ...] = (
    "data scientist",
    "software engineer",
    "registered nurse",
    "project manager",
    "accountant",
    "sales representative",
    "customer service",
    "electrician",
    "marketing manager",
    "financial analyst",
    "teacher",
    "warehouse worker",
    "graphic designer",
    "human resources",
    "data analyst",
    "product manager",
    "cybersecurity analyst",
    "pharmacist",
    "mechanical engineer",
    "administrative assistant",
)


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


def _job_dedup_key(job: dict) -> tuple[str, object]:
    jid = job.get("id")
    if jid is not None:
        return ("id", jid)
    url = job.get("redirect_url") or ""
    return ("url", url)


def fetch_jobs_multi(
    app_id: str,
    app_key: str,
    keywords: list[str],
    *,
    where: str,
    limit_per_keyword: int,
    max_total: int | None,
    country: str,
    base_url: str,
    timeout: float,
    pause_s: float,
) -> list[dict]:
    """Run one search per keyword, merge, and dedupe by Adzuna job id (or URL)."""
    seen: set[tuple[str, object]] = set()
    merged: list[dict] = []
    for i, kw in enumerate(keywords):
        if i and pause_s > 0:
            time.sleep(pause_s)
        batch = fetch_jobs(
            app_id,
            app_key,
            keyword=kw,
            where=where,
            limit=limit_per_keyword,
            country=country,
            base_url=base_url,
            timeout=timeout,
        )
        for job in batch:
            key = _job_dedup_key(job)
            if key in seen:
                continue
            seen.add(key)
            merged.append(job)
            if max_total is not None and len(merged) >= max_total:
                return merged
    return merged


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch jobs from Adzuna and save raw outputs.")
    p.add_argument(
        "--diverse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch many role types and merge (default: on). Use --no-diverse for a single --keyword search.",
    )
    p.add_argument(
        "--keyword",
        default="data scientist",
        help="Single search keywords when --no-diverse (Adzuna `what`).",
    )
    p.add_argument(
        "--keywords-csv",
        default=None,
        metavar="STR",
        help='Comma-separated keywords when --diverse (overrides built-in mix), e.g. "nurse,chef,plumber".',
    )
    p.add_argument(
        "--where",
        default="California",
        help='Location string (Adzuna `where`), e.g. "California" or "remote".',
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="With --diverse: max jobs per keyword. With --no-diverse: max total jobs (default: 100).",
    )
    p.add_argument(
        "--max-total",
        type=int,
        default=2500,
        help="Cap merged unique jobs when --diverse (default: 2500). Use 0 for no cap.",
    )
    p.add_argument(
        "--pause",
        type=float,
        default=0.35,
        help="Seconds to sleep between keyword requests when --diverse (default: 0.35).",
    )
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
    base = args.base_url.rstrip("/")

    if args.diverse:
        if args.keywords_csv:
            keywords = [k.strip() for k in args.keywords_csv.split(",") if k.strip()]
            if not keywords:
                print("No keywords after parsing --keywords-csv.", file=sys.stderr)
                sys.exit(1)
        else:
            keywords = list(DEFAULT_DIVERSE_KEYWORDS)
        max_total = None if args.max_total == 0 else args.max_total
        try:
            raw_results = fetch_jobs_multi(
                app_id,
                app_key,
                keywords,
                where=args.where,
                limit_per_keyword=args.limit,
                max_total=max_total,
                country=args.country,
                base_url=base,
                timeout=args.timeout,
                pause_s=args.pause,
            )
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"Adzuna API returned HTTP {status}.", file=sys.stderr)
            sys.exit(1)
        except requests.RequestException as e:
            print(
                f"Request to Adzuna failed ({type(e).__name__}). Check network and credentials.",
                file=sys.stderr,
            )
            sys.exit(1)
        kw_note = f"{len(keywords)} keyword(s), up to {args.limit} each"
    else:
        keywords = [args.keyword]
        try:
            raw_results = fetch_jobs(
                app_id,
                app_key,
                keyword=args.keyword,
                where=args.where,
                limit=args.limit,
                country=args.country,
                base_url=base,
                timeout=args.timeout,
            )
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            print(f"Adzuna API returned HTTP {status}.", file=sys.stderr)
            sys.exit(1)
        except requests.RequestException as e:
            print(
                f"Request to Adzuna failed ({type(e).__name__}). Check network and credentials.",
                file=sys.stderr,
            )
            sys.exit(1)
        kw_note = f"single keyword {args.keyword!r}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "raw_jobs.json"
    csv_path = args.out_dir / "raw_jobs.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(raw_results, f, ensure_ascii=False, indent=2)

    df = normalize_jobs(raw_results)
    df.to_csv(csv_path, index=False)

    print(
        f"Fetched {len(raw_results)} unique jobs → {csv_path} and {json_path} "
        f"({kw_note}; where={args.where!r})."
    )


if __name__ == "__main__":
    main()

"""
Reddit Mental Health Sentiment Scraper
Collects stress-related posts and comments from Reddit via Arctic Shift API
(no API key required) for NLP sentiment analysis.

Target subreddits : r/college, r/students, r/mentalhealth
Default window    : 16-week academic semester (configurable)
Output            : /apps/data/reddit_raw.csv  (appended page-by-page, safe to interrupt)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 16-week semester window — adjust to your target semester
# Fall 2025: Sep 1 → Dec 21  |  Spring 2026: Jan 5 → Apr 26
SEMESTER_START = "2025-09-01"
SEMESTER_END   = "2025-12-21"

TARGET_SUBREDDITS = ["college", "students", "mentalhealth"]

SCRAPE_COMMENTS = True  # set False to skip comments (much faster)

# Keywords that signal stress / mental health content
STRESS_KEYWORDS = [
    "stress", "stressed", "anxiety", "anxious", "depressed", "depression",
    "overwhelmed", "burnout", "exhausted", "mental health", "panic",
    "struggling", "can't cope", "breakdown", "crying", "hopeless",
    "exam", "finals", "midterm", "deadline", "failing", "failed",
    "sleep deprived", "no motivation", "giving up", "drop out",
]

BASE_URL   = "https://arctic-shift.photon-reddit.com/api"
LIMIT      = 100   # per request (max 1000)
SLEEP_SEC  = 1.0   # polite delay between requests
OUTPUT_DIR = "/app/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "reddit_raw.csv")


# ---------------------------------------------------------------------------
# CSV helpers — append page-by-page so nothing is lost on interrupt
# ---------------------------------------------------------------------------

def load_existing_ids() -> set:
    """Return set of (id, type) already saved — used to skip duplicates on resume."""
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        df = pd.read_csv(OUTPUT_FILE, usecols=["id", "type"], dtype=str)
        return set(zip(df["id"], df["type"]))
    except Exception:
        return set()


def append_to_csv(records: list[dict], existing_ids: set) -> int:
    """Append new records to CSV, skipping duplicates. Returns count written."""
    if not records:
        return 0

    new_records = [r for r in records if (str(r["id"]), r["type"]) not in existing_ids]
    if not new_records:
        return 0

    df = pd.DataFrame(new_records)
    write_header = not os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode="a", header=write_header, index=False, encoding="utf-8")

    for r in new_records:
        existing_ids.add((str(r["id"]), r["type"]))

    return len(new_records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_stress_related(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in STRESS_KEYWORDS)


def fetch_page(endpoint: str, params: dict) -> list[dict]:
    url = f"{BASE_URL}/{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] {e}")
        return []


def fmt_record(raw: dict, record_type: str) -> dict:
    """Normalise a raw API post or comment into our CSV schema."""
    if record_type == "post":
        return {
            "id":           raw.get("id"),
            "subreddit":    raw.get("subreddit"),
            "type":         "post",
            "title":        raw.get("title"),
            "text":         raw.get("selftext"),
            "author":       raw.get("author"),
            "score":        raw.get("score"),
            "upvote_ratio": raw.get("upvote_ratio"),
            "num_comments": raw.get("num_comments"),
            "created_utc":  datetime.fromtimestamp(raw["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "url":          raw.get("url"),
            "permalink":    raw.get("permalink"),
            "parent_id":    None,
        }
    else:
        return {
            "id":           raw.get("id"),
            "subreddit":    raw.get("subreddit"),
            "type":         "comment",
            "title":        None,
            "text":         raw.get("body"),
            "author":       raw.get("author"),
            "score":        raw.get("score"),
            "upvote_ratio": None,
            "num_comments": None,
            "created_utc":  datetime.fromtimestamp(raw["created_utc"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "url":          None,
            "permalink":    raw.get("permalink"),
            "parent_id":    raw.get("link_id"),
        }


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def scrape(endpoint: str, record_type: str, subreddit: str,
           after: str, before: str, existing_ids: set, total_counter: list):
    """
    Generic paginator for posts or comments.
    Appends each page to CSV immediately — safe to interrupt at any time.
    """
    page_before  = before
    page_num     = 0
    session_saved = 0

    label = "posts" if record_type == "post" else "comments"
    print(f"\n[r/{subreddit}] {label} {after} → {before}")

    while True:
        params = {
            "subreddit": subreddit,
            "after":     after,
            "before":    page_before,
            "limit":     LIMIT,
        }
        page = fetch_page(endpoint, params)

        if not page:
            break

        page_num += 1

        # Filter and save this page immediately
        records = []
        for raw in page:
            text = ((raw.get("title") or "") + " " + (raw.get("selftext") or "")
                    if record_type == "post" else (raw.get("body") or ""))
            if text in ("[deleted]", "[removed]") or not text:
                continue
            if not is_stress_related(text):
                continue
            records.append(fmt_record(raw, record_type))

        saved = append_to_csv(records, existing_ids)
        session_saved += saved
        total_counter[0] += saved

        oldest_utc = min(r["created_utc"] for r in page)
        newest_utc = max(r["created_utc"] for r in page)
        date_range = (
            f"{datetime.fromtimestamp(oldest_utc, tz=timezone.utc).strftime('%Y-%m-%d')}"
            f" → {datetime.fromtimestamp(newest_utc, tz=timezone.utc).strftime('%Y-%m-%d')}"
        )

        print(f"  page {page_num:>4} | {date_range} | fetched {len(page):>4} | "
              f"matched {len(records):>4} | saved {saved:>4} | total {total_counter[0]:>7}")

        if len(page) < LIMIT:
            print(f"  -> done r/{subreddit} {label}: {session_saved} new records saved")
            break

        page_before = oldest_utc
        time.sleep(SLEEP_SEC)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load already-saved IDs so we skip duplicates if resuming
    existing_ids = load_existing_ids()
    resume_count = len(existing_ids)

    print("=" * 65)
    print(f"  Reddit Stress Scraper")
    print(f"  Window    : {SEMESTER_START} → {SEMESTER_END}")
    print(f"  Subreddits: {TARGET_SUBREDDITS}")
    print(f"  Comments  : {'yes' if SCRAPE_COMMENTS else 'no'}")
    print(f"  Output    : {OUTPUT_FILE}")
    print(f"  Resuming  : {resume_count} records already in file")
    print("=" * 65)

    total_counter = [resume_count]  # mutable so scrape() can update it

    for sub in TARGET_SUBREDDITS:
        try:
            scrape("posts/search",    "post",    sub, SEMESTER_START, SEMESTER_END, existing_ids, total_counter)
            if SCRAPE_COMMENTS:
                scrape("comments/search", "comment", sub, SEMESTER_START, SEMESTER_END, existing_ids, total_counter)
        except Exception as e:
            print(f"  [ERROR] r/{sub}: {e}")

    print("\n" + "=" * 65)
    print(f"  Finished. Total records in {OUTPUT_FILE}: {total_counter[0]}")
    print("=" * 65)


if __name__ == "__main__":
    main()

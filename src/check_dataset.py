"""
check_scrape_data.py — Scrape data quality & statistics report

Usage:
    python src/check_dataset.py --input data/1_reddit_raw.csv
    python src/check_dataset.py --input data/1_reddit_raw.csv --start 2024-01-15 --end 2025-12-20
    python src/check_dataset.py --help
"""

import argparse
import sys
from datetime import date, timedelta

import pandas as pd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print data quality statistics for a scraped Reddit CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/check_dataset.py --input data/1_reddit_raw.csv
  python src/check_dataset.py --input data/1_reddit_raw.csv --start 2024-01-15 --end 2025-12-20
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to scraped CSV file (output of 1_scrape_reddit.py)"
    )
    parser.add_argument(
        "--start", default="2024-01-15",
        help="Expected start date YYYY-MM-DD (default: 2024-01-15)"
    )
    parser.add_argument(
        "--end", default="2025-12-20",
        help="Expected end date YYYY-MM-DD (default: 2025-12-20)"
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP = "─" * 60

def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def warn(msg: str):
    print(f"  ⚠  WARNING: {msg}")

def ok(msg: str):
    print(f"  ✓  {msg}")


# ── Report sections ───────────────────────────────────────────────────────────

def report_overview(df: pd.DataFrame):
    section("1. FILE OVERVIEW")
    print(f"  Total rows       : {len(df):,}")
    print(f"  Posts            : {(df['type'] == 'post').sum():,}")
    print(f"  Comments         : {(df['type'] == 'comment').sum():,}")
    print(f"  Date range       : {df['date'].min()} → {df['date'].max()}")
    print(f"  Unique authors   : {df['author'].nunique():,}")
    dupes = df.duplicated(subset='id').sum()
    print(f"  Duplicate IDs    : {dupes:,}")
    if dupes > 0:
        warn(f"{dupes} duplicate post/comment IDs found — consider deduplication.")
    else:
        ok("No duplicate IDs.")


def report_missing_days(df: pd.DataFrame, start: date, end: date):
    section("2. MISSING DAYS")
    expected = set(start + timedelta(n) for n in range((end - start).days + 1))
    actual = set(df['date'].unique())
    missing_global = sorted(expected - actual)

    total_expected = len(expected)
    print(f"  Expected days    : {total_expected}")
    print(f"  Days with data   : {len(actual)}")

    if missing_global:
        warn(f"{len(missing_global)} day(s) missing from full dataset:")
        for d in missing_global:
            print(f"       {d}")
    else:
        ok("No missing days in full dataset.")

    # Per-subreddit missing days
    print(f"\n  Per-subreddit coverage:")
    for sub in sorted(df['subreddit'].unique()):
        sub_days = set(df[df['subreddit'] == sub]['date'].unique())
        sub_missing = sorted(expected - sub_days)
        flag = f"  ⚠  missing {len(sub_missing)}d: {sub_missing}" if sub_missing else "  ✓"
        print(f"    {sub:<25} {len(sub_days)}/{total_expected} days  {flag}")


def report_subreddit_distribution(df: pd.DataFrame):
    section("3. SUBREDDIT DISTRIBUTION")
    counts = df['subreddit'].value_counts()
    total = len(df)
    print(f"  {'Subreddit':<25} {'Rows':>8}  {'Share':>7}")
    print(f"  {'-'*25} {'-'*8}  {'-'*7}")
    for sub, cnt in counts.items():
        pct = cnt / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {sub:<25} {cnt:>8,}  {pct:>6.1f}%  {bar}")

    dominant = counts.index[0]
    dominant_pct = counts.iloc[0] / total * 100
    if dominant_pct > 40:
        warn(f"r/{dominant} dominates at {dominant_pct:.1f}% — may skew daily counts.")


def report_daily_counts(df: pd.DataFrame):
    section("4. DAILY COUNT STATISTICS")
    daily = df.groupby('date').size().rename('count')

    print(f"  Min   : {daily.min():,}  on {daily.idxmin()}")
    print(f"  Max   : {daily.max():,}  on {daily.idxmax()}")
    print(f"  Mean  : {daily.mean():.1f}")
    print(f"  Median: {daily.median():.0f}")
    print(f"  Std   : {daily.std():.1f}")

    low_days = daily[daily < 10].sort_values()
    if len(low_days):
        warn(f"{len(low_days)} day(s) with fewer than 10 posts:")
        for d, cnt in low_days.items():
            print(f"       {d}: {cnt} rows")
    else:
        ok("No days with fewer than 10 posts.")

    print(f"\n  Top 10 highest-volume days (likely exam/finals spikes):")
    print(f"  {'Date':<14} {'Rows':>6}")
    print(f"  {'-'*14} {'-'*6}")
    for d, cnt in daily.nlargest(10).items():
        print(f"  {str(d):<14} {cnt:>6,}")


def report_weekly_trend(df: pd.DataFrame):
    section("5. WEEKLY TREND (ISO WEEK)")
    df = df.copy()
    df['iso_week'] = df['created_utc'].dt.isocalendar().week.astype(int)
    df['iso_year'] = df['created_utc'].dt.isocalendar().year.astype(int)
    weekly = df.groupby(['iso_year', 'iso_week']).size()

    print(f"  {'Year':<6} {'Week':>6} {'Rows':>8}  Trend")
    print(f"  {'-'*6} {'-'*6} {'-'*8}  {'-'*30}")
    max_cnt = weekly.max()
    for (yr, wk), cnt in weekly.items():
        bar = "█" * int(cnt / max_cnt * 30)
        print(f"  {yr:<6} {wk:>6} {cnt:>8,}  {bar}")


def report_content_quality(df: pd.DataFrame):
    section("6. CONTENT QUALITY")
    removed_vals = {'[removed]', '[deleted]', 'nan', ''}

    def is_empty(val):
        return pd.isna(val) or str(val).strip().lower() in removed_vals

    # Check text column (comments) and title column (posts)
    text_empty = df['text'].apply(is_empty).sum()
    title_empty = df['title'].apply(is_empty).sum()
    text_pct = text_empty / len(df) * 100
    title_pct = title_empty / len(df) * 100

    print(f"  Empty/removed text  : {text_empty:,} ({text_pct:.1f}%)")
    print(f"  Empty/removed title : {title_empty:,} ({title_pct:.1f}%)")

    if text_pct > 30:
        warn(f"{text_pct:.1f}% of text fields are empty/removed — high removal rate.")
    else:
        ok(f"Text removal rate is acceptable ({text_pct:.1f}%).")

    print(f"\n  Score distribution:")
    print(f"    Min    : {df['score'].min():.0f}")
    print(f"    Median : {df['score'].median():.0f}")
    print(f"    Mean   : {df['score'].mean():.1f}")
    print(f"    Max    : {df['score'].max():.0f}")
    neg_pct = (df['score'] <= 0).sum() / len(df) * 100
    print(f"    Score ≤ 0: {neg_pct:.1f}%")


def report_flags(df: pd.DataFrame, start: date, end: date):
    section("7. SUMMARY FLAGS")
    flags = []

    # Dec 21 check
    if end not in set(df['date'].unique()):
        flags.append(f"End date {end} not found in data — scrape may have stopped at {df['date'].max()}.")

    # r/Students thin
    students = df[df['subreddit'] == 'Students']
    if len(students) > 0:
        avg_daily = len(students) / df['date'].nunique()
        if avg_daily < 10:
            flags.append(f"r/Students very thin ({avg_daily:.1f} rows/day avg) — low signal, note in methodology.")

    # Dominant subreddit
    counts = df['subreddit'].value_counts()
    top_pct = counts.iloc[0] / len(df) * 100
    if top_pct > 40:
        flags.append(f"r/{counts.index[0]} dominates at {top_pct:.1f}% — may skew aggregate daily counts.")

    # Duplicate IDs
    dupes = df.duplicated(subset='id').sum()
    if dupes > 0:
        flags.append(f"{dupes} duplicate IDs detected.")

    if flags:
        for f in flags:
            warn(f)
    else:
        ok("No flags — data looks clean and ready for Step 2 classifier.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    except ValueError as e:
        print(f"ERROR: Invalid date format — {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  SCRAPE DATA QUALITY REPORT")
    print(f"  Input : {args.input}")
    print(f"  Semester: {start} → {end}  ({(end - start).days + 1} days expected)")
    print(f"{'═' * 60}")

    try:
        df = pd.read_csv(args.input, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['date'] = df['created_utc'].dt.date

    report_overview(df)
    report_missing_days(df, start, end)
    report_subreddit_distribution(df)
    report_daily_counts(df)
    report_weekly_trend(df)
    report_content_quality(df)
    report_flags(df, start, end)

    print(f"\n{'═' * 60}\n")


if __name__ == "__main__":
    main()

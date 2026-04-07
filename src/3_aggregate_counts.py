"""
3_aggregate_counts.py — Daily stress count aggregation (Step 3)

Takes the classified CSV (output of Step 2) and produces a daily
time series of stress post counts — input for GLM and forecasting.

Features produced:
  - Core counts: total_posts, stressed, not_stressed, needs_review, stress_rate
  - Subreddit breakdown: stressed_college, stressed_GradSchool, stressed_mentalhealth, stressed_Students
  - Content type: post_ratio (proportion of posts vs comments)
  - Engagement: mean_score (all rows), mean_score_posts (posts only), mean_num_comments (posts only)
  - Academic calendar: is_exam_period, is_semester_break
  - Calendar: week_number, day_of_week, day_of_week_name, month, day_number
  - Time-series: rolling_7d, z_score, is_spike

Usage:
    python src/3_aggregate_counts.py --input data/2_reddit_labeled_1.csv
    python src/3_aggregate_counts.py --input data/2_reddit_labeled.csv --output data/3_daily_counts.csv
    python src/3_aggregate_counts.py --help
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Project root = parent of this script's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate daily stress counts from classified Reddit CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Columns in output CSV:
  date              — calendar date (YYYY-MM-DD)
  total_posts       — all posts/comments that day
  stressed          — is_stressed == 1  (both models agree: stressed)
  not_stressed      — is_stressed == 0  (both models agree: not stressed)
  needs_review      — is_stressed == -1 (models disagree)
  stress_rate       — stressed / (stressed + not_stressed)  [excludes needs_review]
  week_number       — ISO week number
  day_of_week       — 0=Monday … 6=Sunday
  day_of_week_name  — Monday … Sunday
  day_number        — 1–111 sequential day index (for GLM trend term)
  rolling_7d        — 7-day centred rolling mean of stressed count
  z_score           — (stressed - mean) / std  (spike detection)
  is_spike          — 1 if z_score > 2.0, else 0

Examples:
  python src/3_aggregate_counts.py --input data/2_reddit_labeled.csv
  python src/3_aggregate_counts.py --input data/2_reddit_labeled.csv --start 2024-01-15 --end 2025-12-20
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to classified CSV (output of 2_classify_sentiment.py)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: data/3_daily_counts.csv)"
    )
    parser.add_argument(
        "--start", default="2024-01-15",
        help="Start date YYYY-MM-DD (default: 2024-01-15)"
    )
    parser.add_argument(
        "--end", default="2025-12-20",
        help="End date YYYY-MM-DD (default: 2025-12-20)"
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    except ValueError as e:
        print(f"ERROR: Invalid date — {e}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(PROJECT_ROOT / "data" / "3_daily_counts.csv")

    print(f"\n{'═' * 60}")
    print(f"  DAILY STRESS COUNT AGGREGATION")
    print(f"  Input    : {args.input}")
    print(f"  Output   : {output_path}")
    print(f"  Semester : {start} → {end}")
    print(f"{'═' * 60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.input, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    required = {'created_utc', 'is_stressed', 'needs_review'}
    missing_cols = required - set(df.columns)
    if missing_cols:
        print(f"ERROR: Missing columns: {missing_cols}", file=sys.stderr)
        sys.exit(1)

    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df['date'] = df['created_utc'].dt.date

    # Filter to semester window
    df = df[(df['date'] >= start) & (df['date'] <= end)]
    print(f"\n  Rows in semester window : {len(df):,}")

    # ── Aggregate by day ──────────────────────────────────────────────────────

    # Core counts
    daily = df.groupby('date').agg(
        total_posts=('is_stressed', 'count'),
        stressed=('is_stressed', lambda x: (x == 1).sum()),
        not_stressed=('is_stressed', lambda x: (x == 0).sum()),
        needs_review=('is_stressed', lambda x: (x == -1).sum()),
    ).reset_index()

    daily['stress_rate'] = (
        daily['stressed'] / (daily['stressed'] + daily['not_stressed'])
    ).round(4)

    # ── Subreddit breakdown (stressed posts per subreddit per day) ────────
    stressed_df = df[df['is_stressed'] == 1]
    subreddit_cols = {}
    for sub in sorted(df['subreddit'].unique()):
        col = f"stressed_{sub}"
        sub_daily = stressed_df[stressed_df['subreddit'] == sub].groupby('date').size()
        subreddit_cols[col] = sub_daily
    sub_df = pd.DataFrame(subreddit_cols).fillna(0).astype(int)
    sub_df.index.name = 'date'
    daily = daily.merge(sub_df, on='date', how='left')
    for col in subreddit_cols:
        daily[col] = daily[col].fillna(0).astype(int)

    # ── Content type: post vs comment ratio ───────────────────────────────
    post_counts = df[df['type'] == 'post'].groupby('date').size().rename('n_posts')
    daily = daily.merge(post_counts, on='date', how='left')
    daily['n_posts'] = daily['n_posts'].fillna(0).astype(int)
    daily['post_ratio'] = (daily['n_posts'] / daily['total_posts']).round(4)
    daily.drop(columns=['n_posts'], inplace=True)

    # ── Engagement metrics ────────────────────────────────────────────────
    # mean_score: average Reddit score across ALL rows (posts + comments) per day
    score_daily = df.groupby('date')['score'].mean().rename('mean_score').round(2)
    daily = daily.merge(score_daily, on='date', how='left')

    # Posts-only engagement (num_comments is only available for posts)
    posts_only = df[df['type'] == 'post']
    post_engagement = posts_only.groupby('date').agg(
        mean_score_posts=('score', 'mean'),
        mean_num_comments=('num_comments', 'mean'),
    ).round(2)
    daily = daily.merge(post_engagement, on='date', how='left')

    # ── Fill any missing days with zeros ──────────────────────────────────────
    daily['date'] = pd.to_datetime(daily['date'])
    all_dates = pd.DataFrame({
        'date': pd.date_range(start, end, freq='D')
    })
    daily = all_dates.merge(daily, on='date', how='left').fillna(0)

    # Fix integer columns after fillna
    int_cols = ['total_posts', 'stressed', 'not_stressed', 'needs_review']
    int_cols += [c for c in daily.columns if c.startswith('stressed_')]
    for c in int_cols:
        if c in daily.columns:
            daily[c] = daily[c].astype(int)

    # Drop days with no scraped posts — scraper cut-off artifacts, not real zeros
    zero_days = daily[daily['total_posts'] == 0]
    if len(zero_days) > 0:
        print(f"  Dropped {len(zero_days)} day(s) with no scraped posts: {zero_days['date'].tolist()}")
        daily = daily[daily['total_posts'] > 0].reset_index(drop=True)

    # ── Academic calendar flags (multi-region) ───────────────────────────
    # Covers 4 major English-speaking higher education systems whose
    # students dominate the studied subreddits.  Flag = 1 if ANY region
    # has exams or break on that date.
    #
    # Sources:
    #   US/Canada — semester system (dominant on r/college, r/GradSchool)
    #     Exam periods : Apr 25 – May 15 (spring finals), Dec 1 – Dec 20 (fall finals)
    #     Breaks       : Jun 1 – Aug 15 (summer), Dec 21 – Jan 5 (winter holiday)
    #   UK — term system (significant presence on r/Students)
    #     Exam periods : Jan 10 – Jan 31 (winter exams), May 10 – Jun 20 (summer exams)
    #     Breaks       : Jun 21 – Sep 15 (summer), Dec 15 – Jan 9 (Christmas)
    #   Australia — southern-hemisphere semester system
    #     Exam periods : Jun 1 – Jun 25 (Sem 1 exams), Oct 20 – Nov 20 (Sem 2 exams)
    #     Breaks       : Nov 21 – Feb 20 (summer), Jul 1 – Jul 20 (mid-year)
    #
    # The union approach ensures that on any given day, if students from
    # at least one major region are under exam stress or on break, the
    # flag is active — reflecting the mixed international composition
    # of the subreddits.

    md = daily['date'].dt.month * 100 + daily['date'].dt.day  # MMDD

    daily['is_exam_period'] = (
        # US / Canada
        ((md >= 425) & (md <= 515)) |    # Spring finals
        ((md >= 1201) & (md <= 1220)) |  # Fall finals
        # UK
        ((md >= 110) & (md <= 131)) |    # Winter exams
        ((md >= 510) & (md <= 620)) |    # Summer exams
        # Australia
        ((md >= 601) & (md <= 625)) |    # Sem 1 exams
        ((md >= 1020) & (md <= 1120))    # Sem 2 exams
    ).astype(int)

    daily['is_semester_break'] = (
        # US / Canada
        ((md >= 601) & (md <= 815)) |    # Summer break
        ((md >= 1221) | (md <= 105)) |   # Winter holiday
        # UK
        ((md >= 621) & (md <= 915)) |    # Summer break
        ((md >= 1215) & (md <= 109)) |   # Christmas break
        # Australia
        ((md >= 1121) | (md <= 220)) |   # Southern-hemisphere summer
        ((md >= 701) & (md <= 720))      # Mid-year break
    ).astype(int)

    # ── Calendar features ─────────────────────────────────────────────────────
    daily['week_number'] = daily['date'].dt.isocalendar().week.astype(int)
    daily['day_of_week'] = daily['date'].dt.dayofweek          # 0=Mon
    daily['day_of_week_name'] = daily['date'].dt.day_name()
    daily['month'] = daily['date'].dt.month
    daily['day_number'] = range(1, len(daily) + 1)             # sequential for GLM

    # ── Time-series features ───────────────────────────────────────────────────
    s_vals = daily['stressed'].astype(float)

    # 7-day rolling average (centred, min 3 days at edges)
    daily['rolling_7d'] = s_vals.rolling(7, center=True, min_periods=3).mean().round(1)

    # Z-score for spike detection
    mean_, std_ = s_vals.mean(), s_vals.std()
    daily['z_score'] = ((s_vals - mean_) / std_).round(3)
    daily['is_spike'] = (daily['z_score'] > 2.0).astype(int)


    daily['date'] = daily['date'].dt.date                       # back to date

    # ── Print report ──────────────────────────────────────────────────────────
    section("OVERALL SUMMARY")
    total_days = len(daily)
    print(f"  Days in output      : {total_days}")
    print(f"  Total posts         : {daily['total_posts'].sum():,}")
    print(f"  Stressed (label=1)  : {daily['stressed'].sum():,}  ({daily['stressed'].sum()/len(df)*100:.1f}%)")
    print(f"  Not stressed (0)    : {daily['not_stressed'].sum():,}  ({daily['not_stressed'].sum()/len(df)*100:.1f}%)")
    print(f"  Needs review (-1)   : {daily['needs_review'].sum():,}  ({daily['needs_review'].sum()/len(df)*100:.1f}%)")
    print(f"  Avg stress rate/day : {daily['stress_rate'].mean()*100:.1f}%")

    section("DAILY STRESSED COUNT — STATS")
    s = daily['stressed']
    print(f"  Min    : {s.min():,}  on {daily.loc[s.idxmin(), 'date']}")
    print(f"  Max    : {s.max():,}  on {daily.loc[s.idxmax(), 'date']}")
    print(f"  Mean   : {s.mean():.1f}")
    print(f"  Median : {s.median():.0f}")
    print(f"  Std    : {s.std():.1f}")

    section("TOP 10 HIGHEST STRESS DAYS")
    top10 = daily.nlargest(10, 'stressed')[['date', 'stressed', 'total_posts', 'stress_rate', 'day_of_week_name']]
    print(f"  {'Date':<12} {'Stressed':>9} {'Total':>7} {'Rate':>7}  Day")
    print(f"  {'-'*12} {'-'*9} {'-'*7} {'-'*7}  {'-'*10}")
    for _, row in top10.iterrows():
        print(f"  {str(row['date']):<12} {row['stressed']:>9,} {row['total_posts']:>7,} {row['stress_rate']:>7.1%}  {row['day_of_week_name']}")

    section("WEEKLY STRESS TOTALS  (Semester Weeks 1–16)")
    daily['sem_week'] = ((daily['day_number'] - 1) // 7) + 1
    weekly = daily.groupby('sem_week').agg(
        date_from=('date', 'min'),
        date_to=('date', 'max'),
        days=('stressed', 'count'),
        stressed=('stressed', 'sum'),
        total=('total_posts', 'sum'),
    )
    weekly['stress_rate'] = (weekly['stressed'] / weekly['total']).round(4)
    max_stressed = weekly['stressed'].max()
    print(f"  {'Wk':>4}  {'Date Range':<23} {'Days':>4} {'Stressed':>9} {'Total':>8} {'Rate':>7}  Trend")
    print(f"  {'-'*4}  {'-'*23} {'-'*4} {'-'*9} {'-'*8} {'-'*7}  {'-'*30}")
    for wk, row in weekly.iterrows():
        bar = "█" * int(row['stressed'] / max_stressed * 30)
        print(f"  {wk:>4}  {str(row['date_from'])} → {str(row['date_to'])}  {int(row['days']):>4} {row['stressed']:>9,} {row['total']:>8,} {row['stress_rate']:>7.1%}  {bar}")

    section("DAY-OF-WEEK PATTERN")
    dow = daily.groupby('day_of_week_name').agg(
        avg_stressed=('stressed', 'mean'),
        avg_stress_rate=('stress_rate', 'mean'),
    ).round(2)
    # Sort Mon → Sun
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow = dow.reindex([d for d in order if d in dow.index])
    print(f"  {'Day':<12} {'Avg Stressed':>13} {'Avg Rate':>10}")
    print(f"  {'-'*12} {'-'*13} {'-'*10}")
    for day, row in dow.iterrows():
        print(f"  {day:<12} {row['avg_stressed']:>13.1f} {row['avg_stress_rate']:>10.1%}")

    # ── ASCII daily chart with 7-day rolling average ───────────────────────────
    section("DAILY TIME SERIES (ASCII CHART)")
    chart_h = 16
    vals = daily['stressed'].tolist()
    roll = daily['rolling_7d'].tolist()
    chart_w = min(len(vals), 200)
    v_max = max(vals)
    def scale(v): return int(round(v / v_max * (chart_h - 1)))

    # Build grid
    grid = [[' '] * chart_w for _ in range(chart_h)]
    for x, (v, r) in enumerate(zip(vals[:chart_w], roll[:chart_w])):
        sv = scale(v)
        grid[chart_h - 1 - sv][x] = '▪'        # daily count dot
        if not np.isnan(r):
            sr = scale(r)
            grid[chart_h - 1 - sr][x] = '─'    # rolling average line

    print(f"  (▪ = daily stressed count  ─ = 7-day rolling avg  max={v_max:,})\n")
    for i, row in enumerate(grid):
        level = int(v_max * (chart_h - 1 - i) / (chart_h - 1))
        print(f"  {level:>4} │{''.join(row)}")
    # X-axis: mark week numbers
    print(f"       └{'─' * chart_w}")
    week_labels = ' ' * 5
    for x in range(chart_w):
        wk = daily.iloc[x]['sem_week']
        if x % 7 == 0:
            week_labels += f"W{wk:<6}"
    print(f"       {week_labels}")

    # ── Spike detection ────────────────────────────────────────────────────────
    section("SPIKE DETECTION  (z-score > 2.0)")
    spikes = daily[daily['is_spike'] == 1][['date', 'stressed', 'z_score', 'day_of_week_name']]
    if len(spikes):
        print(f"  {len(spikes)} spike day(s) detected:\n")
        print(f"  {'Date':<12} {'Stressed':>9} {'Z-score':>8}  {'Day':<12}")
        print(f"  {'-'*12} {'-'*9} {'-'*8}  {'-'*12}")
        for _, row in spikes.sort_values('z_score', ascending=False).iterrows():
            print(f"  {str(row['date']):<12} {row['stressed']:>9,} {row['z_score']:>8.2f}  {row['day_of_week_name']:<12}")
    else:
        print("  No spike days detected.")

    # ── Autocorrelation ────────────────────────────────────────────────────────
    section("AUTOCORRELATION  (signal for ARIMA/Prophet)")
    series = daily['stressed'].astype(float)
    print(f"  Lag   ACF     Interpretation")
    print(f"  {'-'*5} {'-'*7} {'-'*35}")
    for lag in [1, 2, 3, 7, 14, 21]:
        acf = series.autocorr(lag=lag)
        if abs(acf) >= 0.5:   interp = "strong correlation"
        elif abs(acf) >= 0.3: interp = "moderate correlation"
        elif abs(acf) >= 0.1: interp = "weak correlation"
        else:                  interp = "negligible"
        bar = ('▓' if acf > 0 else '░') * int(abs(acf) * 20)
        print(f"  {lag:<5} {acf:>+.3f}  {bar:<20} {interp}")
    lag1  = series.autocorr(lag=1)
    lag7  = series.autocorr(lag=7)
    notes = []
    if abs(lag7) >= 0.4:
        notes.append("Lag-7 ≥ 0.4 ✓ → weekly seasonality detected (use Prophet/SARIMA)")
    if abs(lag1) >= 0.5:
        notes.append("Lag-1 ≥ 0.5 ✓ → strong trend component (consider differencing for ARIMA)")
    elif abs(lag1) >= 0.3:
        notes.append(f"Lag-1 = {lag1:+.3f} (moderate persistence, no strong trend)")
    if notes:
        print()
        for n in notes:
            print(f"  → {n}")

    # ── Stationarity check ─────────────────────────────────────────────────────
    section("STATIONARITY CHECK  (first half vs second half)")
    half = len(daily) // 2
    h1 = daily['stressed'].iloc[:half]
    h2 = daily['stressed'].iloc[half:]
    print(f"  {'Metric':<18} {'First half':>12} {'Second half':>12} {'Change':>10}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Mean':<18} {h1.mean():>12.1f} {h2.mean():>12.1f} {(h2.mean()-h1.mean()):>+10.1f}")
    print(f"  {'Std dev':<18} {h1.std():>12.1f} {h2.std():>12.1f} {(h2.std()-h1.std()):>+10.1f}")
    print(f"  {'Variance':<18} {h1.var():>12.1f} {h2.var():>12.1f} {(h2.var()-h1.var()):>+10.1f}")
    mean_unstable = h2.mean() > h1.mean() * 1.3 or h2.mean() < h1.mean() * 0.7
    var_ratio = h2.var() / h1.var() if h1.var() > 0 else 1.0
    var_unstable = var_ratio < 0.6 or var_ratio > 1.67  # variance changes by >40%

    if mean_unstable and var_unstable:
        print(f"\n  ⚠  Non-stationary: mean and variance both change substantially.")
        print(f"     → Apply log transform + differencing before ARIMA.")
    elif mean_unstable:
        print(f"\n  ⚠  Non-stationary: mean shifts sharply between halves.")
        print(f"     → Apply differencing before ARIMA.")
    elif var_unstable:
        print(f"\n  ⚠  Heteroscedastic: variance changed {var_ratio:.1f}x between halves.")
        print(f"     → Apply log transform before ARIMA (Prophet handles this natively).")
    else:
        print(f"\n  ✓  Mean and variance both stable — series likely weakly stationary.")

    # ── Save ──────────────────────────────────────────────────────────────────
    daily.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved → {output_path}  ({len(daily)} rows × {len(daily.columns)} columns)\n")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()

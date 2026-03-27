"""
3_aggregate_counts.py — Daily stress count aggregation (Step 3)

Takes the classified CSV (output of Step 2) and produces a 111-row
daily time series of stress post counts — input for GLM and forecasting.

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
    daily = df.groupby('date').agg(
        total_posts=('is_stressed', 'count'),
        stressed=('is_stressed', lambda x: (x == 1).sum()),
        not_stressed=('is_stressed', lambda x: (x == 0).sum()),
        needs_review=('is_stressed', lambda x: (x == -1).sum()),
    ).reset_index()

    daily['stress_rate'] = (
        daily['stressed'] / (daily['stressed'] + daily['not_stressed'])
    ).round(4)

    # ── Fill any missing days with zeros ──────────────────────────────────────
    all_dates = pd.DataFrame({
        'date': [start + timedelta(n) for n in range((end - start).days + 1)]
    })
    daily = all_dates.merge(daily, on='date', how='left').fillna(0)
    daily[['total_posts', 'stressed', 'not_stressed', 'needs_review']] = \
        daily[['total_posts', 'stressed', 'not_stressed', 'needs_review']].astype(int)

    # ── Calendar features ─────────────────────────────────────────────────────
    daily['date'] = pd.to_datetime(daily['date'])
    daily['week_number'] = daily['date'].dt.isocalendar().week.astype(int)
    daily['day_of_week'] = daily['date'].dt.dayofweek          # 0=Mon
    daily['day_of_week_name'] = daily['date'].dt.day_name()
    daily['day_number'] = range(1, len(daily) + 1)             # 1–111 for GLM

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

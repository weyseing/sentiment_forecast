"""
generate_synthetic_2yr.py — Generate realistic 2-year synthetic stress count data.

Simulates the output of the full pipeline (Steps 1-3) for 2 academic years
(2024-01-15 to 2025-12-20) with realistic academic calendar stress patterns.

Usage:
    python src/generate_synthetic_2yr.py
    python src/generate_synthetic_2yr.py --output data/3_daily_counts_2yr.csv --seed 42
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Academic Calendar ──────────────────────────────────────────────────────────
# Each entry: (start, end, multiplier_label)
# Multipliers applied to base daily stress count

ACADEMIC_PERIODS = [
    # Spring 2024
    ("2024-01-15", "2024-02-09", "early_semester"),   # weeks 1-4
    ("2024-02-10", "2024-03-01", "mid_semester"),      # weeks 5-7
    ("2024-03-02", "2024-03-15", "midterm"),           # midterms
    ("2024-03-16", "2024-03-22", "break"),             # spring break
    ("2024-03-23", "2024-04-19", "post_midterm"),      # weeks 11-14
    ("2024-04-20", "2024-05-10", "finals"),            # finals
    # Summer 2024
    ("2024-05-11", "2024-08-31", "summer_break"),
    # Fall 2024
    ("2024-09-02", "2024-10-04", "early_semester"),    # weeks 1-4
    ("2024-10-05", "2024-10-11", "mid_semester"),      # weeks 5-6
    ("2024-10-12", "2024-10-25", "midterm"),           # midterms
    ("2024-10-26", "2024-11-22", "post_midterm"),      # weeks 9-12
    ("2024-11-23", "2024-12-01", "thanksgiving"),      # thanksgiving week
    ("2024-12-02", "2024-12-20", "finals"),            # finals
    # Winter break
    ("2024-12-21", "2025-01-14", "winter_break"),
    # Spring 2025
    ("2025-01-15", "2025-02-07", "early_semester"),    # weeks 1-4
    ("2025-02-08", "2025-02-28", "mid_semester"),      # weeks 5-7
    ("2025-03-01", "2025-03-14", "midterm"),           # midterms
    ("2025-03-15", "2025-03-21", "break"),             # spring break
    ("2025-03-22", "2025-04-18", "post_midterm"),      # weeks 11-14
    ("2025-04-19", "2025-05-09", "finals"),            # finals
    # Summer 2025
    ("2025-05-10", "2025-08-31", "summer_break"),
    # Fall 2025
    ("2025-09-01", "2025-10-03", "early_semester"),    # weeks 1-4
    ("2025-10-04", "2025-10-10", "mid_semester"),      # weeks 5-6
    ("2025-10-11", "2025-10-24", "midterm"),           # midterms
    ("2025-10-25", "2025-11-21", "post_midterm"),      # weeks 9-12
    ("2025-11-22", "2025-11-30", "thanksgiving"),      # thanksgiving week
    ("2025-12-01", "2025-12-20", "finals"),            # finals
]

# Stress multipliers by period type
PERIOD_MULTIPLIERS = {
    "early_semester":  1.0,
    "mid_semester":    1.3,
    "midterm":         2.0,
    "break":           0.5,
    "post_midterm":    1.1,
    "finals":          2.5,
    "summer_break":    0.45,
    "winter_break":    0.35,
    "thanksgiving":    0.8,
}

# Day-of-week multipliers (0=Mon, 6=Sun)
DOW_MULTIPLIERS = {0: 1.0, 1: 1.05, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.70, 6: 0.65}

BASE_STRESS_MEAN = 18  # avg daily stressed posts during regular semester


def get_period_multiplier(d: date) -> float:
    ds = d.strftime("%Y-%m-%d")
    for start, end, label in ACADEMIC_PERIODS:
        if start <= ds <= end:
            return PERIOD_MULTIPLIERS[label]
    return 0.4  # outside defined periods (between semesters)


def generate(start: str, end: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []

    for ts in dates:
        d = ts.date()
        period_mult = get_period_multiplier(d)
        dow_mult = DOW_MULTIPLIERS[d.weekday()]

        # Weekly upward trend within semester (students accumulate fatigue)
        week_of_year = ts.isocalendar()[1]
        trend_factor = 1.0 + 0.005 * (week_of_year % 20)  # small positive trend

        mu = BASE_STRESS_MEAN * period_mult * dow_mult * trend_factor

        # Negative Binomial: dispersion parameter controls variance
        # mu = r * p/(1-p), variance = mu + mu^2/r
        r = 3.5  # overdispersion (lower = more variance)
        p = r / (r + mu)
        stressed = int(rng.negative_binomial(r, p))

        # not_stressed: roughly 3x stressed on average
        not_mu = max(stressed * rng.uniform(2.0, 4.5), 1)
        not_stressed = int(rng.negative_binomial(r, r / (r + not_mu)))

        # needs_review: ~15% of total
        total = stressed + not_stressed
        needs_review = int(rng.binomial(max(total // 5, 1), 0.12))

        rows.append({
            "date": d,
            "stressed": stressed,
            "not_stressed": not_stressed,
            "needs_review": needs_review,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["total_posts"] = df["stressed"] + df["not_stressed"] + df["needs_review"]
    df["stress_rate"] = df["stressed"] / (df["stressed"] + df["not_stressed"]).clip(lower=1)
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_week_name"] = df["date"].dt.day_name()
    df["day_number"] = (df["date"] - df["date"].min()).dt.days + 1
    df["rolling_7d"] = df["stressed"].rolling(7, center=True, min_periods=1).mean().round(2)
    mean_s = df["stressed"].mean()
    std_s = df["stressed"].std()
    df["z_score"] = ((df["stressed"] - mean_s) / std_s).round(3)
    df["is_spike"] = (df["z_score"] > 2.0).astype(int)

    col_order = [
        "date", "total_posts", "stressed", "not_stressed", "needs_review",
        "stress_rate", "week_number", "day_of_week", "day_of_week_name",
        "day_number", "rolling_7d", "z_score", "is_spike",
    ]
    return df[col_order]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic 2-year stress count data")
    parser.add_argument("--start",  default="2024-01-15")
    parser.add_argument("--end",    default="2025-12-20")
    parser.add_argument("--output", default="data/3_daily_counts_2yr.csv")
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic 2-year data: {args.start} → {args.end}")
    df = generate(args.start, args.end, args.seed)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"\nSummary:")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total days : {len(df)}")
    print(f"  Stress posts — mean={df['stressed'].mean():.1f}, "
          f"std={df['stressed'].std():.1f}, "
          f"min={df['stressed'].min()}, max={df['stressed'].max()}")
    print(f"  Spikes detected: {df['is_spike'].sum()}")


if __name__ == "__main__":
    main()

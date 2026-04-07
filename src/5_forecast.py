"""
5_forecast.py — Time Series Forecasting with Walk-Forward Validation

Models:
    - Prophet  (primary, with academic calendar events)
    - SARIMA   (baseline, seasonal order 7)

Validation strategy:
    Walk-forward validation — train on expanding window, predict next N days.
    Multiple windows give a more honest picture than a single train/test split.

Usage:
    python src/5_forecast.py
    python src/5_forecast.py --input data/2years/3_daily_counts.csv
    python src/5_forecast.py --input data/2years/3_daily_counts.csv --output-dir outputs --horizon 21 --windows 4

Outputs:
    outputs/5_cv_scores.csv         — MAE / RMSE / MAPE per model per window
    outputs/5_cv_summary.csv        — Mean scores across all windows
    outputs/5_cv_plot.png           — Actual vs predicted for each window
    outputs/5_final_forecast.png    — Final forecast trained on all data
    outputs/5_final_forecast.csv    — Final forecast values
    outputs/5_yearly_pattern.png    — Monthly average stressed posts (academic calendar)
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-forward forecast: Prophet vs SARIMA on daily stressed post counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      default="data/2years/3_daily_counts.csv",
                        help="Path to daily counts CSV (output of step 3)")
    parser.add_argument("--output-dir", default="/app/data/2years",
                        help="Directory to save outputs (CSVs and plots)")
    parser.add_argument("--horizon",    type=int, default=21,
                        help="Forecast horizon in days")
    parser.add_argument("--windows",    type=int, default=4,
                        help="Number of walk-forward validation windows")
    parser.add_argument("--min-train",  type=int, default=180,
                        help="Minimum training days before first forecast window")
    parser.add_argument("--window-type", default="expanding", choices=["expanding", "sliding"],
                        help="expanding: train grows each window | sliding: fixed train size (min-train days)")
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def mape(actual, predicted):
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


# ── Academic calendar events ──────────────────────────────────────────────────

def build_academic_events(start_year=2024, end_year=2026):
    """
    Build a DataFrame of recurring academic calendar events for Prophet.

    Covers 4 major English-speaking higher education systems whose
    students dominate the studied subreddits (r/college, r/Students,
    r/GradSchool, r/mentalhealth with student-context filters):

      US / Canada (semester system):
        - us_spring_finals     : Apr 25 – May 15
        - us_fall_finals       : Dec 1 – Dec 20
        - us_winter_holiday    : Dec 21 – Jan 5
        - us_summer_break      : Jun 1 – Aug 15

      UK (term system):
        - uk_winter_exams      : Jan 10 – Jan 31
        - uk_summer_exams      : May 10 – Jun 20
        - uk_summer_break      : Jun 21 – Sep 15
        - uk_christmas_break   : Dec 15 – Jan 9

      Australia (southern-hemisphere semester system):
        - au_sem1_exams        : Jun 1 – Jun 25
        - au_sem2_exams        : Oct 20 – Nov 20
        - au_summer_break      : Nov 21 – Feb 20
        - au_midyear_break     : Jul 1 – Jul 20

    The union of these calendars ensures the model accounts for stress
    peaks from any major English-speaking student population present
    in the subreddits.
    """
    # (event_name, month_start, day_start, month_end, day_end, wraps_year)
    events = [
        # US / Canada
        ("us_spring_finals",    4, 25,  5, 15, False),
        ("us_fall_finals",     12,  1, 12, 20, False),
        ("us_winter_holiday",  12, 21,  1,  5, True),
        ("us_summer_break",     6,  1,  8, 15, False),
        # UK
        ("uk_winter_exams",     1, 10,  1, 31, False),
        ("uk_summer_exams",     5, 10,  6, 20, False),
        ("uk_summer_break",     6, 21,  9, 15, False),
        ("uk_christmas_break", 12, 15,  1,  9, True),
        # Australia
        ("au_sem1_exams",       6,  1,  6, 25, False),
        ("au_sem2_exams",      10, 20, 11, 20, False),
        ("au_summer_break",    11, 21,  2, 20, True),
        ("au_midyear_break",    7,  1,  7, 20, False),
    ]

    rows = []
    for year in range(start_year, end_year + 1):
        for name, ms, ds, me, de, wraps in events:
            try:
                if wraps:
                    date_range = pd.date_range(f"{year}-{ms}-{ds}", f"{year+1}-{me}-{de}")
                else:
                    date_range = pd.date_range(f"{year}-{ms}-{ds}", f"{year}-{me}-{de}")
                for d in date_range:
                    rows.append({"holiday": name, "ds": d,
                                 "lower_window": 0, "upper_window": 0})
            except ValueError:
                pass

    return pd.DataFrame(rows).drop_duplicates(subset=["holiday", "ds"])


# ── Yearly seasonality analysis ───────────────────────────────────────────────

def analyze_yearly_pattern(df, output_path):
    """
    Print and plot monthly average stressed posts to reveal the academic
    calendar cycle embedded in the yearly seasonality.
    """
    df = df.copy()
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly = df.groupby(["year", "month"])["stressed"].mean().reset_index()
    monthly.columns = ["year", "month", "mean_stressed"]

    period_label = {
        1: "UK winter exams / US-UK winter break",
        2: "Spring semester (US/UK/AU)",
        3: "Spring semester (US/UK/AU)",
        4: "Spring semester (US/UK/AU)",
        5: "Exams: US spring finals + UK summer exams",
        6: "Exams: UK summer + AU Sem 1 / US-UK break starts",
        7: "Break: US/UK summer + AU mid-year",
        8: "Break: US/UK summer",
        9: "Fall semester start (US/UK)",
        10: "AU Sem 2 exams / US-UK fall semester",
        11: "AU Sem 2 exams / US-UK fall semester",
        12: "US/CA fall finals / AU-UK break starts",
    }
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    print("\n  Monthly average stressed posts/day:")
    print(f"  {'Year':<6} {'Month':<8} {'Mean posts/day':>15}  {'Academic Period'}")
    print(f"  {'-'*58}")
    for _, row in monthly.iterrows():
        m = int(row.month)
        print(f"  {int(row.year):<6} {month_names[m]:<8} {row.mean_stressed:>15.1f}  {period_label[m]}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = {2024: "steelblue", 2025: "darkorange"}
    for year, grp in monthly.groupby("year"):
        ax.plot(grp["month"], grp["mean_stressed"],
                marker="o", linewidth=2, label=str(year),
                color=colors.get(year, "gray"))

    # Shade multi-region academic periods (2 colors: exam=red, break=blue)
    # Each span: (month_start, month_end, type, region_label)
    # Months are approximate since this chart shows monthly granularity
    exam_color  = "#e74c3c"
    break_color = "#3498db"

    multi_region_spans = [
        # Exams
        (11.5, 12.5, "exam",  "US/CA"),     # US/CA fall finals (Dec)
        ( 0.5,  1.5, "exam",  "UK"),         # UK winter exams (Jan)
        ( 4.5,  5.5, "exam",  "US/CA"),      # US/CA spring finals (May)
        ( 4.5,  6.5, "exam",  "UK"),         # UK summer exams (May–Jun)
        ( 5.5,  6.5, "exam",  "AU"),         # AU Sem 1 exams (Jun)
        ( 9.5, 11.5, "exam",  "AU"),         # AU Sem 2 exams (Oct–Nov)
        # Breaks
        (12.5, 13.0, "break", "US/UK"),      # Winter holiday (late Dec → Jan)
        ( 0.5,  1.0, "break", "US/UK"),      # Winter holiday (Jan portion)
        ( 5.5,  8.5, "break", "US/UK"),      # US/UK summer break (Jun–Aug)
        ( 6.5,  7.5, "break", "AU"),         # AU mid-year break (Jul)
    ]

    exam_labeled = False
    break_labeled = False
    for m_start, m_end, etype, region in multi_region_spans:
        color = exam_color if etype == "exam" else break_color
        if etype == "exam" and not exam_labeled:
            lbl = "Exam period"
            exam_labeled = True
        elif etype == "break" and not break_labeled:
            lbl = "Semester break"
            break_labeled = True
        else:
            lbl = "_nolegend_"
        ax.axvspan(m_start, m_end, alpha=0.12, color=color, label=lbl, zorder=0)
        # Region label at top
        ax.text((m_start + m_end) / 2, 0.96, region, ha="center", va="top",
                fontsize=7, color=color, alpha=0.8, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean stressed posts / day")
    ax.set_title("Yearly Seasonality — Monthly Average Stressed Posts\n"
                 "(Multi-region academic calendar: US/Canada, UK, Australia)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return monthly


# ── Academic events → exogenous matrix (for SARIMAX) ─────────────────────────

def build_exog(dates, academic_events):
    """Convert academic_events DataFrame into a binary indicator matrix.

    Each academic event type becomes one column (0/1) aligned to `dates`.
    This is the exogenous regressor format required by SARIMAX.
    """
    event_types = sorted(academic_events["holiday"].unique())
    exog = pd.DataFrame({"date": pd.to_datetime(dates)})
    for ev in event_types:
        ev_dates = set(academic_events.loc[academic_events["holiday"] == ev, "ds"].dt.normalize())
        exog[ev] = exog["date"].dt.normalize().isin(ev_dates).astype(float)
    return exog.set_index("date")[event_types]


# ── Prophet forecast ──────────────────────────────────────────────────────────

def forecast_prophet(train_df, horizon, academic_events=None):
    """Fit Prophet on train_df, return predictions for next `horizon` days.

    academic_events: if provided, passed as Prophet holidays DataFrame so the
    model can learn the effect of exam periods and holiday breaks on post volume.
    """
    prophet_df = train_df[["date", "stressed"]].rename(columns={"date": "ds", "stressed": "y"})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        holidays=academic_events,
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    preds = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    preds["yhat"] = preds["yhat"].clip(lower=0)
    return preds, model


# ── ARIMA forecast ────────────────────────────────────────────────────────────

def forecast_sarima(train_df, horizon, academic_events=None):
    """Fit SARIMAX(1,1,1)(1,1,1,7) on train_df, return predictions for next `horizon` days.

    Reindexes to a continuous daily date range before fitting so that any
    missing dates don't misalign the weekly seasonal period.
    Missing days are filled with the 7-day rolling mean.

    academic_events: if provided, binary exogenous regressors for each academic
    event type are passed to SARIMAX so both models receive the same calendar
    information, ensuring a fair comparison with Prophet.
    """
    s = train_df.set_index("date")["stressed"].astype(float)
    s.index = pd.to_datetime(s.index)
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    if s.isna().any():
        s = s.fillna(s.rolling(7, min_periods=1, center=True).mean())

    exog_train = None
    exog_fc    = None
    if academic_events is not None:
        forecast_dates = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        exog_train = build_exog(full_idx, academic_events)
        exog_fc    = build_exog(forecast_dates, academic_events)

    model  = SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), exog=exog_train)
    result = model.fit(disp=False)
    forecast = result.forecast(steps=horizon, exog=exog_fc)
    forecast = np.clip(np.asarray(forecast), 0, None)
    return forecast


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validation(df, horizon, n_windows, min_train, window_type="expanding",
                             academic_events=None):
    """
    Walk-forward validation — two modes:

    expanding: train always starts from row 0, grows each window
        Window 1: train[0:split1]            → test[split1 : split1+horizon]
        Window 2: train[0:split2]            → test[split2 : split2+horizon]

    sliding: fixed train size (min_train days), window slides forward
        Window 1: train[split1-min_train:split1] → test[split1 : split1+horizon]
        Window 2: train[split2-min_train:split2] → test[split2 : split2+horizon]

    Splits are evenly spaced from min_train to (len(df) - horizon).
    """
    n = len(df)
    max_split = n - horizon
    split_points = np.linspace(min_train, max_split, n_windows, dtype=int)

    results = []
    window_details = []

    for i, split in enumerate(split_points):
        win_num = i + 1
        if window_type == "sliding":
            train = df.iloc[split - min_train:split].copy()
        else:
            train = df.iloc[:split].copy()
        test  = df.iloc[split:split + horizon].copy()

        print(f"\n  Window {win_num}/{n_windows}")
        print(f"    Train: {train['date'].iloc[0].date()} → {train['date'].iloc[-1].date()}  ({len(train)} days)")
        print(f"    Test : {test['date'].iloc[0].date()}  → {test['date'].iloc[-1].date()}  ({len(test)} days)")

        actual = test["stressed"].values

        # Prophet
        try:
            prophet_preds, _ = forecast_prophet(train, horizon, academic_events)
            p_pred = prophet_preds["yhat"].values
            p_mae  = mae(actual, p_pred)
            p_rmse = rmse(actual, p_pred)
            p_mape = mape(actual, p_pred)
            print(f"    Prophet  — MAE={p_mae:.1f}  RMSE={p_rmse:.1f}  MAPE={p_mape:.1f}%")
        except Exception as e:
            print(f"    Prophet  — FAILED: {e}")
            p_pred = np.full(horizon, np.nan)
            p_mae = p_rmse = p_mape = np.nan

        # SARIMA
        try:
            a_pred = forecast_sarima(train, horizon, academic_events)
            a_mae  = mae(actual, a_pred)
            a_rmse = rmse(actual, a_pred)
            a_mape = mape(actual, a_pred)
            print(f"    SARIMA   — MAE={a_mae:.1f}  RMSE={a_rmse:.1f}  MAPE={a_mape:.1f}%")
        except Exception as e:
            print(f"    SARIMA   — FAILED: {e}")
            a_pred = np.full(horizon, np.nan)
            a_mae = a_rmse = a_mape = np.nan

        results.append({"window": win_num, "model": "Prophet", "MAE": p_mae, "RMSE": p_rmse, "MAPE": p_mape})
        results.append({"window": win_num, "model": "SARIMA",  "MAE": a_mae, "RMSE": a_rmse, "MAPE": a_mape})

        window_details.append({
            "window":       win_num,
            "train_end":    train["date"].iloc[-1],
            "test_dates":   test["date"].values,
            "actual":       actual,
            "prophet_pred": p_pred,
            "sarima_pred":  a_pred,
            "prophet_ci_low":  prophet_preds["yhat_lower"].values if not np.all(np.isnan(p_pred)) else np.full(horizon, np.nan),
            "prophet_ci_high": prophet_preds["yhat_upper"].values if not np.all(np.isnan(p_pred)) else np.full(horizon, np.nan),
        })

    return pd.DataFrame(results), window_details


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cv_windows(window_details, df, output_path):
    """One subplot per walk-forward window showing actual vs predicted."""
    n = len(window_details)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Walk-Forward Validation: Actual vs Predicted\nInput: {df.attrs.get('source', 'unknown')}", fontsize=14, fontweight="bold")

    for i, wd in enumerate(window_details):
        ax = axes[i]
        dates = pd.to_datetime(wd["test_dates"])

        # Show some training tail for context (last 30 days)
        train_tail = df[df["date"] <= wd["train_end"]].tail(30)
        ax.plot(train_tail["date"], train_tail["stressed"],
                color="gray", linewidth=1, alpha=0.5, label="Training (tail)")

        # Actual test
        ax.plot(dates, wd["actual"], color="black", linewidth=1.8,
                marker="o", markersize=3, label="Actual", zorder=5)

        # Prophet
        if not np.all(np.isnan(wd["prophet_pred"])):
            ax.plot(dates, wd["prophet_pred"], color="steelblue",
                    linewidth=1.5, linestyle="--", label="Prophet")
            ax.fill_between(dates, wd["prophet_ci_low"], wd["prophet_ci_high"],
                            alpha=0.15, color="steelblue")

        # SARIMA
        if not np.all(np.isnan(wd["sarima_pred"])):
            ax.plot(dates, wd["sarima_pred"], color="darkorange",
                    linewidth=1.5, linestyle="--", label="SARIMA")

        ax.axvline(wd["train_end"], color="red", linewidth=1,
                   linestyle=":", alpha=0.7, label="Train/Test split")

        ax.set_title(f"Window {wd['window']}  |  Test: {dates[0].date()} → {dates[-1].date()}", fontsize=10)
        ax.set_ylabel("Stressed posts/day")
        ax.legend(fontsize=8, loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_final_forecast(df, prophet_forecast, horizon, output_path):
    """Full history + final Prophet forecast with academic period annotations."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Full history
    ax.plot(df["date"], df["stressed"], color="black", linewidth=0.8,
            alpha=0.7, label="Historical (actual)")

    # Forecast
    fc = prophet_forecast[prophet_forecast["ds"] > df["date"].max()]
    ax.plot(fc["ds"], fc["yhat"], color="steelblue", linewidth=2,
            linestyle="--", label="Prophet forecast")
    ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                    alpha=0.2, color="steelblue", label="95% CI")

    ax.axvline(df["date"].max(), color="red", linewidth=1.2,
               linestyle=":", label="Forecast start")

    # Shade recurring academic periods across the full date range
    all_dates_min = df["date"].min()
    all_dates_max = fc["ds"].max() if len(fc) > 0 else df["date"].max()

    # Two-color scheme: red = exam stress, blue = break/low activity
    # Region shown via text annotation inside each band (not legend)
    # Stacked vertically by region so overlaps don't hide each other
    academic_shading = [
        # (month_start, day_start, month_end, day_end, type, region_label)
        # Exams (red tones)
        (12,  1, 12, 20, "exam",  "US/CA"),
        ( 1, 10,  1, 31, "exam",  "UK"),
        ( 4, 25,  5, 15, "exam",  "US/CA"),
        ( 5, 10,  6, 20, "exam",  "UK"),
        ( 6,  1,  6, 25, "exam",  "AU"),
        (10, 20, 11, 20, "exam",  "AU"),
        # Breaks (blue tones)
        (12, 21,  1,  9, "break", "US/UK"),
        ( 6, 21,  8, 15, "break", "US/UK"),
        ( 7,  1,  7, 20, "break", "AU"),
        (11, 21,  2, 20, "break", "AU"),
    ]

    exam_color  = "#e74c3c"   # red
    break_color = "#3498db"   # blue

    # Track legend entries
    exam_labeled = False
    break_labeled = False

    for year in range(all_dates_min.year, all_dates_max.year + 2):
        for ms, ds, me, de, etype, region in academic_shading:
            try:
                if ms <= me:
                    span_start = pd.Timestamp(year, ms, ds)
                    span_end   = pd.Timestamp(year, me, de)
                else:
                    span_start = pd.Timestamp(year,   ms, ds)
                    span_end   = pd.Timestamp(year+1, me, de)
                span_start = max(span_start, all_dates_min)
                span_end   = min(span_end,   all_dates_max)
                if span_start >= span_end:
                    continue

                color = exam_color if etype == "exam" else break_color
                alpha = 0.15

                # Legend: one entry per type
                if etype == "exam" and not exam_labeled:
                    lbl = "Exam period"
                    exam_labeled = True
                elif etype == "break" and not break_labeled:
                    lbl = "Semester break"
                    break_labeled = True
                else:
                    lbl = "_nolegend_"

                ax.axvspan(span_start, span_end, alpha=alpha, color=color,
                           label=lbl, zorder=0)

                # Small region label at top of band (use axes transform for y)
                mid = span_start + (span_end - span_start) / 2
                ax.text(mid, 0.96, region, ha="center", va="top",
                        fontsize=6, color=color, alpha=0.8, fontweight="bold",
                        transform=ax.get_xaxis_transform())
            except ValueError:
                pass

    ax.set_title("Final Prophet Forecast — Trained on Full Dataset\n"
                 "(Multi-region academic calendar: US/Canada, UK, Australia)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stressed posts/day")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    section("1. LOAD DATA")
    df = pd.read_csv(args.input, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.attrs["source"] = args.input
    print(f"  Rows       : {len(df)}")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stressed   : min={df['stressed'].min()}  max={df['stressed'].max()}  mean={df['stressed'].mean():.1f}")

    # ── 2. Yearly seasonality analysis ───────────────────────────────────────
    section("2. YEARLY SEASONALITY ANALYSIS (Academic Calendar)")
    yearly_plot_path = os.path.join(args.output_dir, "5_yearly_pattern.png")
    monthly_df = analyze_yearly_pattern(df, yearly_plot_path)
    print(f"\n  Saved: {yearly_plot_path}")

    # ── 3. Build academic calendar events for Prophet ─────────────────────────
    section("3. ACADEMIC CALENDAR EVENTS (Prophet holidays)")
    start_yr = df["date"].dt.year.min()
    end_yr   = df["date"].dt.year.max() + 1
    academic_events = build_academic_events(start_year=start_yr, end_year=end_yr)
    print(f"  Built {len(academic_events)} event-day rows across {start_yr}–{end_yr}")
    print(f"  Event types: {sorted(academic_events['holiday'].unique())}")
    print()
    print("  Both models receive the same academic calendar information:")
    print("  Prophet — via the holidays DataFrame (built-in support)")
    print("  SARIMAX  — via binary exogenous regressors (one column per event type)")

    # ── 4. Walk-forward validation ────────────────────────────────────────────
    section(f"4. WALK-FORWARD VALIDATION  (horizon={args.horizon} days, {args.windows} windows, type={args.window_type})")
    print(f"  Window type : {args.window_type}")
    print(f"  Train days  : {'fixed ' + str(args.min_train) if args.window_type == 'sliding' else 'expanding from ' + str(args.min_train)}")

    cv_scores, window_details = walk_forward_validation(
        df, args.horizon, args.windows, args.min_train, args.window_type,
        academic_events=academic_events,
    )

    # ── 5. Summary scores ─────────────────────────────────────────────────────
    section("5. CV SCORES SUMMARY")
    summary = cv_scores.groupby("model")[["MAE", "RMSE", "MAPE"]].mean().round(2)
    summary.columns = ["Mean MAE", "Mean RMSE", "Mean MAPE (%)"]
    print(summary.to_string())

    winner = summary["Mean MAE"].idxmin()
    print(f"\n  → Best model (lowest mean MAE): {winner}")

    scores_path = os.path.join(args.output_dir, "5_cv_scores.csv")
    summary_path = os.path.join(args.output_dir, "5_cv_summary.csv")
    cv_scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path)
    print(f"  Saved: {scores_path}")
    print(f"  Saved: {summary_path}")

    # ── 6. CV plot ────────────────────────────────────────────────────────────
    section("6. PLOTTING WALK-FORWARD WINDOWS")
    cv_plot_path = os.path.join(args.output_dir, "5_cv_plot.png")
    plot_cv_windows(window_details, df, cv_plot_path)
    print(f"  Saved: {cv_plot_path}")

    # ── 7. Final forecast (Prophet on full data) ──────────────────────────────
    section("7. FINAL FORECAST — Prophet trained on full dataset")
    print(f"  Forecasting {args.horizon} days beyond {df['date'].max().date()}")
    final_preds, final_model = forecast_prophet(df, args.horizon, academic_events)

    # Full forecast df (history + future)
    future_full = final_model.make_future_dataframe(periods=args.horizon)
    final_forecast = final_model.predict(future_full)

    forecast_future = final_forecast[final_forecast["ds"] > df["date"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    forecast_future["yhat"] = forecast_future["yhat"].clip(lower=0)
    forecast_future.columns = ["date", "forecast", "forecast_low", "forecast_high"]
    forecast_future = forecast_future.reset_index(drop=True)

    print(forecast_future[["date", "forecast", "forecast_low", "forecast_high"]].to_string(index=False))

    fc_csv_path = os.path.join(args.output_dir, "5_final_forecast.csv")
    forecast_future.to_csv(fc_csv_path, index=False)
    print(f"\n  Saved: {fc_csv_path}")

    fc_plot_path = os.path.join(args.output_dir, "5_final_forecast.png")
    plot_final_forecast(df, final_forecast, args.horizon, fc_plot_path)
    print(f"  Saved: {fc_plot_path}")

    print(f"\n{'═' * 60}")
    print(f"  Step 5 complete. Best model: {winner}")
    print(f"  Yearly pattern → {yearly_plot_path}")
    print(f"  CV scores      → {scores_path}")
    print(f"  CV plot        → {cv_plot_path}")
    print(f"  Final forecast → {fc_csv_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()

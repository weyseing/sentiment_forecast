"""
5_forecast.py — Time-Series Forecasting on Daily Stress Post Counts

Models: Prophet (primary) + ARIMA (baseline)
Evaluation: Hold-out last 12 weeks (84 days)
Metrics: MAE, RMSE, MAPE

Usage:
    python src/5_forecast.py
    python src/5_forecast.py --input data/3_daily_counts_2yr.csv --horizon 28
    python src/5_forecast.py --input data/3_daily_counts_2yr.csv --output-dir data
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from prophet import Prophet
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEP = "─" * 65


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Metrics ────────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp   = mape(y_true, y_pred)
    print(f"  {name:<10}  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mp:.1f}%")
    return {"model": name, "MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE_%": round(mp, 1)}


# ── Data Loader ────────────────────────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # Fill any missing dates with 0
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range).fillna(0).reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    return df


# ── Prophet ────────────────────────────────────────────────────────────────────

def run_prophet(train: pd.DataFrame, test: pd.DataFrame, horizon: int,
                output_dir: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Fit Prophet, predict test period + horizon days ahead. Returns (test_preds, future_forecast)."""

    # Prophet expects ds/y columns
    prophet_train = train[["date", "stressed"]].rename(columns={"date": "ds", "stressed": "y"})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10,
        interval_width=0.95,
    )

    # Add academic calendar regressors as custom seasonalities
    m.add_seasonality(name="academic_biannual", period=182.5, fourier_order=5)

    m.fit(prophet_train)

    # Forecast: test period + additional horizon
    total_days = len(test) + horizon
    future = m.make_future_dataframe(periods=total_days, freq="D")
    forecast = m.predict(future)

    # Test predictions (last len(test) rows of training-period predictions)
    test_forecast = forecast[forecast["ds"].isin(test["date"])].copy()
    test_preds = test_forecast["yhat"].clip(lower=0).values

    # Future forecast (beyond training data)
    last_train_date = train["date"].max()
    future_forecast = forecast[forecast["ds"] > last_train_date].copy()
    future_forecast = future_forecast.rename(columns={"ds": "date", "yhat": "forecast",
                                                       "yhat_lower": "lower_95",
                                                       "yhat_upper": "upper_95"})
    future_forecast = future_forecast[["date", "forecast", "lower_95", "upper_95"]]
    future_forecast["forecast"] = future_forecast["forecast"].clip(lower=0).round(1)
    future_forecast["lower_95"] = future_forecast["lower_95"].clip(lower=0).round(1)
    future_forecast["upper_95"] = future_forecast["upper_95"].clip(lower=0).round(1)

    # Save Prophet components plot
    fig = m.plot_components(forecast)
    fig.savefig(output_dir / "5_prophet_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return pd.Series(test_preds, index=test.index), future_forecast


# ── ARIMA ──────────────────────────────────────────────────────────────────────

def run_arima(train: pd.DataFrame, test: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    """Fit auto-ARIMA, predict test period + future horizon."""
    y_train = train["stressed"].values

    model = auto_arima(
        y_train,
        seasonal=True,
        m=7,            # weekly seasonality
        d=None,         # auto-determine differencing
        max_p=3, max_q=3,
        max_P=2, max_Q=2,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    print(f"\n  ARIMA order selected: {model.order}  seasonal: {model.seasonal_order}")

    # Predict test window
    test_preds_raw, _ = model.predict(n_periods=len(test), return_conf_int=True)
    test_preds = np.clip(test_preds_raw, 0, None)

    # Extend forecast into future horizon (refit on train+test)
    y_all = np.concatenate([y_train, test["stressed"].values])
    model.update(test["stressed"].values)
    future_preds_raw, future_ci = model.predict(n_periods=horizon, return_conf_int=True)
    future_preds = np.clip(future_preds_raw, 0, None)

    last_date = test["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_df = pd.DataFrame({
        "date": future_dates,
        "forecast": future_preds.round(1),
        "lower_95": np.clip(future_ci[:, 0], 0, None).round(1),
        "upper_95": future_ci[:, 1].round(1),
    })

    return pd.Series(test_preds, index=test.index), future_df


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(train: pd.DataFrame, test: pd.DataFrame,
                 prophet_test_preds: pd.Series, arima_test_preds: pd.Series,
                 prophet_future: pd.DataFrame, arima_future: pd.DataFrame,
                 output_dir: Path):
    """Main forecast plot: historical + test predictions + future."""

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle("Mental Health Stress Post Forecasting — 2-Year Reddit Dataset\n"
                 "r/college, r/students, r/GradSchool, r/mentalhealth",
                 fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1: Full 2-year history overview ──────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(train["date"], train["stressed"], alpha=0.3, color="steelblue")
    ax1.plot(train["date"], train["stressed"], color="steelblue", linewidth=0.8, label="Training data")
    ax1.plot(train["date"], train["rolling_7d"], color="navy", linewidth=1.5, label="7-day rolling mean")
    ax1.fill_between(test["date"], test["stressed"], alpha=0.3, color="orange")
    ax1.plot(test["date"], test["stressed"], color="darkorange", linewidth=0.8, label="Test data (held out)")
    ax1.set_title("Full 2-Year History: Daily Stressed Posts", fontweight="bold")
    ax1.set_ylabel("Stressed Posts / Day")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Test period — actual vs predicted ─────────────────────────────
    ax2 = axes[1]
    ax2.plot(test["date"], test["stressed"], color="black", linewidth=1.5,
             label="Actual", zorder=5)
    ax2.plot(test["date"], prophet_test_preds.values, color="crimson", linewidth=1.5,
             linestyle="--", label="Prophet forecast", zorder=4)
    ax2.plot(test["date"], arima_test_preds.values, color="forestgreen", linewidth=1.5,
             linestyle="-.", label="ARIMA forecast", zorder=3)
    ax2.set_title("Test Period (Last 12 Weeks): Actual vs Forecast", fontweight="bold")
    ax2.set_ylabel("Stressed Posts / Day")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: Future forecast beyond dataset ────────────────────────────────
    ax3 = axes[2]
    # Show last 30 days of actual data for context
    context = pd.concat([train.tail(30), test])
    ax3.plot(context["date"], context["stressed"], color="black", linewidth=1.2,
             label="Historical (context)", zorder=5)

    # Prophet future
    ax3.fill_between(prophet_future["date"], prophet_future["lower_95"],
                     prophet_future["upper_95"], alpha=0.2, color="crimson", label="Prophet 95% CI")
    ax3.plot(prophet_future["date"], prophet_future["forecast"], color="crimson",
             linewidth=2, linestyle="--", label="Prophet forecast")

    # ARIMA future
    ax3.fill_between(arima_future["date"], arima_future["lower_95"],
                     arima_future["upper_95"], alpha=0.15, color="forestgreen", label="ARIMA 95% CI")
    ax3.plot(arima_future["date"], arima_future["forecast"], color="forestgreen",
             linewidth=2, linestyle="-.", label="ARIMA forecast")

    # Divider line
    ax3.axvline(test["date"].max(), color="gray", linestyle=":", linewidth=1.5, label="Forecast start")
    ax3.set_title(f"Future Forecast: {len(prophet_future)}-Day Horizon Beyond Dataset", fontweight="bold")
    ax3.set_ylabel("Stressed Posts / Day")
    ax3.legend(fontsize=9, ncol=2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%Y"))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = output_dir / "5_forecast_results.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved forecast plot → {out_path}")


def plot_weekly_pattern(df: pd.DataFrame, output_dir: Path):
    """Weekly stress pattern across 2 years."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stress Patterns from 2-Year Dataset", fontweight="bold")

    # Day-of-week pattern
    ax = axes[0]
    dow_mean = df.groupby("day_of_week_name")["stressed"].mean()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_mean = dow_mean.reindex([d for d in order if d in dow_mean.index])
    colors = ["steelblue"] * 5 + ["salmon"] * 2
    ax.bar(dow_mean.index, dow_mean.values, color=colors[:len(dow_mean)])
    ax.set_title("Avg Stressed Posts by Day of Week")
    ax.set_ylabel("Mean Stressed Posts")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # Monthly pattern
    ax2 = axes[1]
    df2 = df.copy()
    df2["month"] = df2["date"].dt.month
    df2["month_name"] = df2["date"].dt.strftime("%b")
    monthly = df2.groupby(["month", "month_name"])["stressed"].mean().reset_index()
    monthly = monthly.sort_values("month")
    bar_colors = ["coral" if m in [3, 4, 5, 10, 11, 12] else "steelblue" for m in monthly["month"]]
    ax2.bar(monthly["month_name"], monthly["stressed"], color=bar_colors)
    ax2.set_title("Avg Stressed Posts by Month (2 Years)")
    ax2.set_ylabel("Mean Stressed Posts")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "5_stress_patterns.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pattern plot  → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Time-series forecasting on daily stress counts")
    parser.add_argument("--input",      default="data/3_daily_counts_2yr.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--horizon",    type=int, default=56,
                        help="Days to forecast beyond the dataset end (default: 56 = 8 weeks)")
    parser.add_argument("--test-weeks", type=int, default=12,
                        help="Weeks to hold out for evaluation (default: 12)")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_path  = PROJECT_ROOT / args.input
    output_dir  = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    section("1. LOAD DATA")
    df = load_data(input_path)
    print(f"  Rows loaded  : {len(df)}")
    print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stressed mean: {df['stressed'].mean():.1f} ± {df['stressed'].std():.1f}")
    print(f"  Total spikes : {df['is_spike'].sum() if 'is_spike' in df.columns else 'N/A'}")

    # ── 2. Train / Test split ──────────────────────────────────────────────────
    section("2. TRAIN / TEST SPLIT")
    test_days = args.test_weeks * 7
    split_idx = len(df) - test_days
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    print(f"  Train: {len(train)} days  ({train['date'].min().date()} → {train['date'].max().date()})")
    print(f"  Test : {len(test)} days   ({test['date'].min().date()} → {test['date'].max().date()})")
    print(f"  Horizon (future): {args.horizon} days ({args.horizon//7} weeks)")

    # ── 3. Prophet ────────────────────────────────────────────────────────────
    section("3. PROPHET FORECASTING")
    print("  Fitting Prophet (yearly + weekly + academic seasonality)...")
    prophet_test_preds, prophet_future = run_prophet(train, test, args.horizon, output_dir)

    # ── 4. ARIMA ──────────────────────────────────────────────────────────────
    section("4. ARIMA FORECASTING (auto-selection)")
    print("  Fitting auto-ARIMA (seasonal, m=7 for weekly)...")
    arima_test_preds, arima_future = run_arima(train, test, args.horizon)

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    section("5. MODEL EVALUATION (Test Period)")
    print(f"  Evaluating on {len(test)} held-out test days:\n")
    results = []
    results.append(evaluate(test["stressed"].values, prophet_test_preds.values, "Prophet"))
    results.append(evaluate(test["stressed"].values, arima_test_preds.values,   "ARIMA"))

    results_df = pd.DataFrame(results)
    winner = results_df.loc[results_df["MAE"].idxmin(), "model"]
    print(f"\n  Best model (lowest MAE): {winner}")

    results_df.to_csv(output_dir / "5_model_comparison.csv", index=False)
    print(f"  Saved metrics → {output_dir / '5_model_comparison.csv'}")

    # ── 6. Future Forecast Summary ─────────────────────────────────────────────
    section("6. FUTURE FORECAST SUMMARY")
    print(f"\n  Prophet — next {args.horizon} days forecast:")
    weekly = prophet_future.copy()
    weekly["week"] = (weekly["date"] - weekly["date"].min()).dt.days // 7 + 1
    weekly_summary = weekly.groupby("week").agg(
        week_start=("date", "first"),
        avg_forecast=("forecast", "mean"),
        avg_lower=("lower_95", "mean"),
        avg_upper=("upper_95", "mean"),
    ).reset_index()
    weekly_summary["avg_forecast"] = weekly_summary["avg_forecast"].round(1)
    weekly_summary["avg_lower"] = weekly_summary["avg_lower"].round(1)
    weekly_summary["avg_upper"] = weekly_summary["avg_upper"].round(1)

    for _, row in weekly_summary.iterrows():
        bar_len = int(row["avg_forecast"] / 2)
        bar = "█" * min(bar_len, 40)
        print(f"  Week {int(row['week']):2d} ({row['week_start'].strftime('%Y-%m-%d')}): "
              f"{row['avg_forecast']:5.1f}/day  [{row['avg_lower']:.0f}–{row['avg_upper']:.0f}]  {bar}")

    # Save future forecasts
    prophet_future_out = output_dir / "5_prophet_forecast.csv"
    arima_future_out   = output_dir / "5_arima_forecast.csv"
    prophet_future.to_csv(prophet_future_out, index=False)
    arima_future.to_csv(arima_future_out, index=False)
    print(f"\n  Prophet forecast saved → {prophet_future_out}")
    print(f"  ARIMA forecast saved   → {arima_future_out}")

    # ── 7. Plots ───────────────────────────────────────────────────────────────
    section("7. GENERATING PLOTS")
    plot_results(train, test, prophet_test_preds, arima_test_preds,
                 prophet_future, arima_future, output_dir)
    plot_weekly_pattern(df, output_dir)

    # ── 8. Final Summary ───────────────────────────────────────────────────────
    section("8. FINAL SUMMARY")
    print(f"""
  Dataset      : {len(df)} days ({df['date'].min().date()} → {df['date'].max().date()})
  Training     : {len(train)} days
  Test (held)  : {len(test)} days ({args.test_weeks} weeks)
  Horizon      : {args.horizon} days ({args.horizon//7} weeks ahead)

  Prophet  — MAE={results[0]['MAE']:.2f}, RMSE={results[0]['RMSE']:.2f}, MAPE={results[0]['MAPE_%']:.1f}%
  ARIMA    — MAE={results[1]['MAE']:.2f}, RMSE={results[1]['RMSE']:.2f}, MAPE={results[1]['MAPE_%']:.1f}%
  Winner   : {winner}

  Output files:
    data/5_forecast_results.png      — main forecast chart (3 panels)
    data/5_prophet_components.png    — Prophet trend + seasonality components
    data/5_stress_patterns.png       — weekly/monthly stress patterns
    data/5_prophet_forecast.csv      — daily Prophet forecast ({args.horizon} days)
    data/5_arima_forecast.csv        — daily ARIMA forecast ({args.horizon} days)
    data/5_model_comparison.csv      — MAE/RMSE/MAPE comparison table
""")


if __name__ == "__main__":
    main()

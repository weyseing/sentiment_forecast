"""
5_forecast.py — Time Series Forecasting with Walk-Forward Validation

Models:
    - Prophet  (primary)
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
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

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


# ── Prophet forecast ──────────────────────────────────────────────────────────

def forecast_prophet(train_df, horizon):
    """Fit Prophet on train_df, return predictions for next `horizon` days."""
    prophet_df = train_df[["date", "stressed"]].rename(columns={"date": "ds", "stressed": "y"})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    preds = forecast.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    preds["yhat"] = preds["yhat"].clip(lower=0)
    return preds, model


# ── ARIMA forecast ────────────────────────────────────────────────────────────

def forecast_sarima(train_series, horizon):
    """Fit SARIMA(1,1,1)(1,1,1,7) on train_series, return predictions for next `horizon` days."""
    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    result = model.fit(disp=False)
    forecast = result.forecast(steps=horizon)
    forecast = np.clip(np.asarray(forecast), 0, None)
    return forecast


# ── Walk-forward validation ───────────────────────────────────────────────────

def walk_forward_validation(df, horizon, n_windows, min_train, window_type="expanding"):
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
            prophet_preds, _ = forecast_prophet(train, horizon)
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
            a_pred = forecast_sarima(train["stressed"].values, horizon)
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

    fig.suptitle("Walk-Forward Validation: Actual vs Predicted", fontsize=14, fontweight="bold")

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
    """Full history + final Prophet forecast."""
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

    ax.set_title("Final Prophet Forecast — Trained on Full Dataset", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stressed posts/day")
    ax.legend()
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
    print(f"  Rows       : {len(df)}")
    print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stressed   : min={df['stressed'].min()}  max={df['stressed'].max()}  mean={df['stressed'].mean():.1f}")

    # ── 2. Walk-forward validation ────────────────────────────────────────────
    section(f"2. WALK-FORWARD VALIDATION  (horizon={args.horizon} days, {args.windows} windows, type={args.window_type})")
    print(f"  Window type : {args.window_type}")
    print(f"  Train days  : {'fixed ' + str(args.min_train) if args.window_type == 'sliding' else 'expanding from ' + str(args.min_train)}")

    cv_scores, window_details = walk_forward_validation(
        df, args.horizon, args.windows, args.min_train, args.window_type
    )

    # ── 3. Summary scores ─────────────────────────────────────────────────────
    section("3. CV SCORES SUMMARY")
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

    # ── 4. CV plot ────────────────────────────────────────────────────────────
    section("4. PLOTTING WALK-FORWARD WINDOWS")
    cv_plot_path = os.path.join(args.output_dir, "5_cv_plot.png")
    plot_cv_windows(window_details, df, cv_plot_path)
    print(f"  Saved: {cv_plot_path}")

    # ── 5. Final forecast (Prophet on full data) ──────────────────────────────
    section("5. FINAL FORECAST — Prophet trained on full dataset")
    print(f"  Forecasting {args.horizon} days beyond {df['date'].max().date()}")
    final_preds, final_model = forecast_prophet(df, args.horizon)

    # Full forecast df (history + future)
    prophet_full_df = df[["date", "stressed"]].rename(columns={"date": "ds", "stressed": "y"})
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
    print(f"  CV scores    → {scores_path}")
    print(f"  CV plot      → {cv_plot_path}")
    print(f"  Final forecast → {fc_csv_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()

"""
4_model_glm.py — Poisson vs Negative Binomial GLM on daily stressed post counts

Usage:
    python src/4_model_glm.py
    python src/4_model_glm.py --input data/3_daily_counts_1sem.csv
    python src/4_model_glm.py --input data/3_daily_counts.csv --output-dir outputs

Outputs:
    outputs/4_irr_table.csv      — Incidence Rate Ratios with CI and p-values
    outputs/4_model_comparison.csv — AIC / BIC / log-likelihood comparison
    outputs/4_residuals.png      — Deviance residuals plot (NB model)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GLM: Poisson vs Negative Binomial on daily stress counts")
    parser.add_argument("--input",      default="data/3_daily_counts_1sem.csv")
    parser.add_argument("--output-dir", default="/app/data")
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

SEP = "─" * 60

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    section("1. LOAD DATA")
    df = pd.read_csv(args.input, parse_dates=["date"])
    print(f"  Rows         : {len(df)}")
    print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Stressed col : min={df['stressed'].min()}  max={df['stressed'].max()}  mean={df['stressed'].mean():.1f}")

    # ── 2. Overdispersion check ───────────────────────────────────────────────
    section("2. OVERDISPERSION CHECK")
    mean_s = df["stressed"].mean()
    var_s  = df["stressed"].var()
    ratio  = var_s / mean_s
    print(f"  Mean     : {mean_s:.2f}")
    print(f"  Variance : {var_s:.2f}")
    print(f"  Ratio (variance / mean) : {ratio:.2f}")
    if ratio > 1.5:
        print(f"  → Overdispersion confirmed (ratio >> 1). NB model expected to win.")
    else:
        print(f"  → No strong overdispersion. Poisson may be adequate.")

    # ── 3. Feature engineering ────────────────────────────────────────────────
    section("3. FEATURE ENGINEERING")

    # Set reference categories (absorbed into intercept)
    # semester_period: "early" is reference  → IRRs show change vs early semester
    # day_of_week_name: "Monday" is reference → IRRs show change vs Monday
    df["semester_period"] = pd.Categorical(
        df["semester_period"],
        categories=["early", "mid", "pre_finals", "finals"],
        ordered=False,
    )
    df["day_of_week_name"] = pd.Categorical(
        df["day_of_week_name"],
        categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=False,
    )

    # sem_week excluded: it is derived directly from semester_period (weeks 1-4=early,
    # 5-8=mid, 9-12=pre_finals, 13-16=finals), so including both causes multicollinearity
    # and produces counterintuitive negative IRRs for the later periods.
    formula = "stressed ~ C(semester_period, Treatment('early')) + C(day_of_week_name, Treatment('Monday'))"
    print(f"  Formula : {formula}")
    print(f"  Reference category — semester_period: 'early'  |  day_of_week: 'Monday'")
    print(f"  Note: sem_week excluded (collinear with semester_period)")

    # ── 4. Fit Poisson GLM ────────────────────────────────────────────────────
    section("4. FIT POISSON GLM")
    poisson_model = smf.glm(formula=formula, data=df, family=Poisson()).fit()
    print(poisson_model.summary())

    # ── 5. Fit Negative Binomial GLM ─────────────────────────────────────────
    section("5. FIT NEGATIVE BINOMIAL GLM")
    # smf.negativebinomial() properly estimates the dispersion parameter (alpha)
    # from the data via MLE, unlike smf.glm(..., NegativeBinomial()) which
    # requires alpha to be pre-specified.
    nb_model = smf.negativebinomial(formula=formula, data=df).fit(disp=False)
    print(nb_model.summary())

    # ── 6. Model comparison ───────────────────────────────────────────────────
    section("6. MODEL COMPARISON (AIC / BIC / Log-Likelihood)")
    comparison = pd.DataFrame({
        "Model":          ["Poisson", "Negative Binomial"],
        "AIC":            [round(poisson_model.aic, 2), round(nb_model.aic, 2)],
        "BIC":            [round(poisson_model.bic, 2), round(nb_model.bic, 2)],
        "Log-Likelihood": [round(poisson_model.llf, 2), round(nb_model.llf, 2)],
    })
    print(comparison.to_string(index=False))
    winner = "Negative Binomial" if nb_model.aic < poisson_model.aic else "Poisson"
    print(f"\n  → Winner (lower AIC): {winner}")

    comp_path = os.path.join(args.output_dir, "4_model_comparison.csv")
    comparison.to_csv(comp_path, index=False)
    print(f"  Saved: {comp_path}")

    # ── 7. IRR table (from winning model) ────────────────────────────────────
    section("7. INCIDENCE RATE RATIOS (IRR)")
    best = nb_model if winner == "Negative Binomial" else poisson_model

    coef   = best.params
    ci     = best.conf_int()
    pvals  = best.pvalues

    irr_df = pd.DataFrame({
        "term":    coef.index,
        "IRR":     np.exp(coef).round(3),
        "CI_low":  np.exp(ci[0]).round(3),
        "CI_high": np.exp(ci[1]).round(3),
        "p_value": pvals.round(4),
    }).reset_index(drop=True)

    # Significance stars
    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    irr_df["sig"] = irr_df["p_value"].apply(stars)

    print(irr_df.to_string(index=False))
    irr_path = os.path.join(args.output_dir, "4_irr_table.csv")
    irr_df.to_csv(irr_path, index=False)
    print(f"\n  Saved: {irr_path}")

    # ── 8. Residual plot ──────────────────────────────────────────────────────
    section("8. RESIDUAL DIAGNOSTICS")
    fitted    = best.fittedvalues
    residuals = best.resid_deviance if hasattr(best, "resid_deviance") else best.resid_response

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"GLM Residual Diagnostics — {winner} Model", fontsize=13)

    # Residuals vs fitted
    axes[0].scatter(fitted, residuals, alpha=0.6, edgecolors="white", linewidths=0.3)
    axes[0].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Fitted values (expected stressed posts/day)")
    axes[0].set_ylabel("Deviance residuals")
    axes[0].set_title("Residuals vs Fitted")

    # Residuals over time
    axes[1].plot(df["date"], residuals, linewidth=0.8, color="steelblue")
    axes[1].axhline(0, color="red", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Deviance residuals")
    axes[1].set_title("Residuals over Time")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "4_residuals.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {plot_path}")

    print(f"\n{'═' * 60}")
    print(f"  Step 4 complete. Winner: {winner}")
    print(f"  IRR table  → {irr_path}")
    print(f"  Residuals  → {plot_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()

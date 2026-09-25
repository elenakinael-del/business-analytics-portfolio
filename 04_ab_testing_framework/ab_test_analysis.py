"""
A/B Testing Framework: Statistical Significance -> Business Decision
========================================================================
Business question: We ran a pricing/checkout-flow experiment. Is the lift
real, and what's it worth if we roll it out to 100% of traffic?

Method:
  1. Simulate a realistic A/B test (control vs. treatment) on conversion
     rate AND revenue-per-visitor (two-metric test, since conversion rate
     alone can be misleading).
  2. Run a proper two-proportion z-test + Welch's t-test, with confidence
     intervals — not just "p < 0.05".
  3. Run a sequential/power check: was the test even adequately powered?
  4. Translate the lift into annualized revenue impact if rolled out.

Author: Elena | github.com/elenakinael-del
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

sns.set_theme(style="whitegrid", palette="deep")
OUT_DIR = "outputs"
DATA_DIR = "data"


def simulate_experiment(n_control=8000, n_treatment=8000, seed=99):
    rng = np.random.default_rng(seed)

    # True underlying rates (unknown to the "analyst" in a real scenario)
    p_control, p_treatment = 0.082, 0.093  # ~13% relative lift in conversion
    control_conv = rng.binomial(1, p_control, n_control)
    treatment_conv = rng.binomial(1, p_treatment, n_treatment)

    # Revenue per visitor: converters spend, non-converters spend 0
    control_rev = np.where(control_conv == 1, rng.gamma(4, 22, n_control), 0)
    treatment_rev = np.where(treatment_conv == 1, rng.gamma(4, 23.5, n_treatment), 0)

    control = pd.DataFrame({"group": "control", "converted": control_conv, "revenue": control_rev})
    treatment = pd.DataFrame({"group": "treatment", "converted": treatment_conv, "revenue": treatment_rev})
    return pd.concat([control, treatment], ignore_index=True)


def analyze_conversion(df):
    counts = df.groupby("group")["converted"].agg(["sum", "count"])
    successes = counts.loc[["control", "treatment"], "sum"].values
    nobs = counts.loc[["control", "treatment"], "count"].values

    z_stat, p_val = proportions_ztest(successes, nobs)
    ci_control = proportion_confint(successes[0], nobs[0], method="wilson")
    ci_treatment = proportion_confint(successes[1], nobs[1], method="wilson")

    rate_control = successes[0] / nobs[0]
    rate_treatment = successes[1] / nobs[1]
    relative_lift = (rate_treatment - rate_control) / rate_control * 100

    return {
        "rate_control": rate_control, "rate_treatment": rate_treatment,
        "ci_control": ci_control, "ci_treatment": ci_treatment,
        "z_stat": z_stat, "p_value": p_val, "relative_lift_pct": relative_lift,
        "n_control": nobs[0], "n_treatment": nobs[1],
    }


def analyze_revenue(df):
    control_rev = df.loc[df.group == "control", "revenue"]
    treatment_rev = df.loc[df.group == "treatment", "revenue"]
    t_stat, p_val = stats.ttest_ind(treatment_rev, control_rev, equal_var=False)

    lift = treatment_rev.mean() - control_rev.mean()
    relative_lift = lift / control_rev.mean() * 100

    return {
        "rpv_control": control_rev.mean(), "rpv_treatment": treatment_rev.mean(),
        "t_stat": t_stat, "p_value": p_val, "relative_lift_pct": relative_lift,
    }


def power_check(conv_result, alpha=0.05):
    """Was the test adequately powered to detect the lift it found?"""
    effect_size = 2 * (np.arcsin(np.sqrt(conv_result["rate_treatment"])) -
                        np.arcsin(np.sqrt(conv_result["rate_control"])))
    analysis = NormalIndPower()
    achieved_power = analysis.power(effect_size=abs(effect_size), nobs1=conv_result["n_control"],
                                     alpha=alpha, ratio=1.0)
    required_n = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=0.8, ratio=1.0)
    return achieved_power, required_n


def business_impact(conv_result, rev_result, annual_visitors=2_000_000):
    """If rolled out to 100% of traffic, what's the annualized revenue impact?"""
    incremental_rpv = rev_result["rpv_treatment"] - rev_result["rpv_control"]
    annual_impact = incremental_rpv * annual_visitors
    return incremental_rpv, annual_impact


def make_charts(df, conv_result, rev_result, power, required_n):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Conversion rate with CI
    groups = ["Control", "Treatment"]
    rates = [conv_result["rate_control"] * 100, conv_result["rate_treatment"] * 100]
    ci_low = [conv_result["ci_control"][0] * 100, conv_result["ci_treatment"][0] * 100]
    ci_high = [conv_result["ci_control"][1] * 100, conv_result["ci_treatment"][1] * 100]
    errs = [[r - l for r, l in zip(rates, ci_low)], [h - r for r, h in zip(rates, ci_high)]]
    bars = axes[0].bar(groups, rates, yerr=errs, capsize=8, color=["#999999", "#2c7fb8"])
    axes[0].set_title(f"Conversion Rate (95% CI)\nLift: {conv_result['relative_lift_pct']:.1f}% | "
                       f"p={conv_result['p_value']:.4f}", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Conversion Rate (%)")
    for bar, v in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.15, f"{v:.2f}%", ha="center", fontweight="bold")

    # Revenue per visitor
    rpv = [rev_result["rpv_control"], rev_result["rpv_treatment"]]
    bars2 = axes[1].bar(groups, rpv, color=["#999999", "#d95f02"])
    axes[1].set_title(f"Revenue per Visitor\nLift: {rev_result['relative_lift_pct']:.1f}% | "
                       f"p={rev_result['p_value']:.4f}", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Revenue per Visitor ($)")
    for bar, v in zip(bars2, rpv):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"${v:.2f}", ha="center", fontweight="bold")

    # Power / sample size
    axes[2].bar(["Achieved Power", "Target Power (80%)"], [power * 100, 80],
                color=["#2c7fb8", "#cccccc"])
    axes[2].set_title(f"Statistical Power Check\nSample used: {conv_result['n_control']:,}/arm | "
                       f"Needed for 80% power: {int(required_n):,}/arm", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Power (%)")
    axes[2].axhline(80, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ab_test_results.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Simulating A/B test data...")
    df = simulate_experiment()
    df.to_csv(f"{DATA_DIR}/experiment_data.csv", index=False)

    conv_result = analyze_conversion(df)
    rev_result = analyze_revenue(df)
    power, required_n = power_check(conv_result)
    incremental_rpv, annual_impact = business_impact(conv_result, rev_result)

    print("\n=== CONVERSION RATE TEST ===")
    print(f"Control: {conv_result['rate_control']*100:.2f}%  |  Treatment: {conv_result['rate_treatment']*100:.2f}%")
    print(f"Relative lift: {conv_result['relative_lift_pct']:.1f}%  |  p-value: {conv_result['p_value']:.5f}")

    print("\n=== REVENUE PER VISITOR TEST ===")
    print(f"Control: ${rev_result['rpv_control']:.2f}  |  Treatment: ${rev_result['rpv_treatment']:.2f}")
    print(f"Relative lift: {rev_result['relative_lift_pct']:.1f}%  |  p-value: {rev_result['p_value']:.5f}")

    print(f"\n=== STATISTICAL POWER ===")
    print(f"Achieved power: {power*100:.1f}% | Required sample size for 80% power: {int(required_n):,}/arm")

    print(f"\n=== BUSINESS IMPACT (if rolled out to 2M annual visitors) ===")
    print(f"Incremental revenue per visitor: ${incremental_rpv:.3f}")
    print(f"Projected annual revenue impact: ${annual_impact:,.0f}")

    verdict = "SHIP IT" if (conv_result["p_value"] < 0.05 and rev_result["p_value"] < 0.05) else "NEEDS MORE DATA"
    print(f"\nRECOMMENDATION: {verdict}")

    make_charts(df, conv_result, rev_result, power, required_n)
    print(f"\nSaved: {OUT_DIR}/ab_test_results.png")


if __name__ == "__main__":
    main()

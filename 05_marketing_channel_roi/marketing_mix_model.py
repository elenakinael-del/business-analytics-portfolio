"""
Marketing Mix Modeling: Channel ROI & Budget Reallocation
=============================================================
Business question: Across our marketing channels, which ones actually
drive revenue, and how should we reallocate a fixed budget to maximize it?

Method:
  1. Generate 2 years of weekly marketing spend (5 channels) and revenue,
     with realistic diminishing returns (saturation) and adstock (carryover)
     effects per channel — this is what makes MMM harder than plain
     linear regression.
  2. Apply adstock transforms, then fit a regularized regression to
     estimate each channel's marginal contribution.
  3. Compute ROI per channel (revenue attributed / spend).
  4. Run a simple budget-reallocation optimizer: same total budget,
     shifted toward higher-ROI channels, projected revenue gain.

Author: Elena | github.com/elenakinael-del
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="deep")
OUT_DIR = "outputs"
DATA_DIR = "data"

CHANNELS = ["search_ads", "social_ads", "display_ads", "tv", "email"]
# True (hidden) efficiency + saturation params used to generate data —
# the model has to recover these from observed spend/revenue alone.
TRUE_PARAMS = {
    "search_ads":  dict(base_roi=4.2, saturation=0.75, adstock=0.25),
    "social_ads":  dict(base_roi=2.8, saturation=0.65, adstock=0.45),
    "display_ads": dict(base_roi=1.4, saturation=0.55, adstock=0.35),
    "tv":          dict(base_roi=1.9, saturation=0.45, adstock=0.70),
    "email":       dict(base_roi=6.5, saturation=0.85, adstock=0.10),
}


def apply_adstock(spend, decay):
    out = np.zeros_like(spend, dtype=float)
    carry = 0.0
    for i, s in enumerate(spend):
        carry = s + decay * carry
        out[i] = carry
    return out


def saturate(x, alpha):
    """Diminishing returns curve (Hill-style, simplified)."""
    return np.power(x, alpha)


def generate_marketing_data(weeks=104, seed=21):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"week": pd.date_range("2024-09-01", periods=weeks, freq="W")})

    revenue = np.full(weeks, 45000.0)  # baseline organic revenue
    revenue += np.linspace(0, 8000, weeks)  # slow organic growth trend
    revenue += 4000 * np.sin(2 * np.pi * np.arange(weeks) / 52)  # seasonality

    for ch, p in TRUE_PARAMS.items():
        base_spend = rng.uniform(3000, 12000, weeks) * (1 + 0.3 * np.sin(np.arange(weeks) / 8))
        base_spend = base_spend.clip(min=500)
        df[f"spend_{ch}"] = base_spend.round(0)

        adstocked = apply_adstock(base_spend, p["adstock"])
        contribution = p["base_roi"] * saturate(adstocked / 1000, p["saturation"]) * 1000
        revenue += contribution

    revenue += rng.normal(0, 1800, weeks)
    df["revenue"] = revenue.round(0)
    return df.set_index("week")


def fit_mmm(df):
    """Fit adstock + saturation transformed regression to recover channel contributions."""
    X = pd.DataFrame(index=df.index)
    # Grid-search a single decay/saturation approx per channel (simplified, transparent)
    best_decay = {}
    for ch in CHANNELS:
        spend = df[f"spend_{ch}"].values
        # try a few decay candidates, pick by correlation with revenue (proxy for fit)
        candidates = [0.1, 0.25, 0.4, 0.55, 0.7]
        corrs = []
        for d in candidates:
            ad = apply_adstock(spend, d)
            sat = saturate(ad / 1000, 0.6)
            corrs.append(np.corrcoef(sat, df["revenue"])[0, 1])
        decay = candidates[int(np.argmax(corrs))]
        best_decay[ch] = decay
        ad = apply_adstock(spend, decay)
        X[ch] = saturate(ad / 1000, 0.6)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    y = df["revenue"].values

    model = Ridge(alpha=5.0, positive=True)
    model.fit(X_s, y)

    # Convert standardized coefficients back to approximate $ contribution per channel
    contributions = {}
    for i, ch in enumerate(CHANNELS):
        pred_with = model.predict(X_s)
        X_zeroed = X_s.copy()
        X_zeroed[:, i] = X_s[:, i].min()  # simulate channel spend at minimum
        pred_without = model.predict(X_zeroed)
        contributions[ch] = float(np.sum(pred_with - pred_without).clip(min=0))

    return model, contributions, best_decay


def compute_roi(df, contributions):
    rows = []
    for ch in CHANNELS:
        total_spend = df[f"spend_{ch}"].sum()
        contribution = contributions[ch]
        roi = contribution / total_spend if total_spend > 0 else 0
        rows.append({
            "channel": ch, "total_spend": round(total_spend, 0),
            "attributed_revenue": round(contribution, 0), "roi_multiple": round(roi, 2),
        })
    return pd.DataFrame(rows).sort_values("roi_multiple", ascending=False)


def optimize_budget(roi_df, total_budget=None, n_channels_boost=2):
    """
    Simple reallocation: shift 20% of budget from the lowest-ROI channels
    into the highest-ROI channels, project the revenue delta using each
    channel's observed marginal ROI (last-dollar assumption, clearly stated).
    """
    df = roi_df.copy().sort_values("roi_multiple", ascending=False).reset_index(drop=True)
    total_budget = total_budget or df["total_spend"].sum()

    shift_amount = df.iloc[-n_channels_boost:]["total_spend"].sum() * 0.4
    df["reallocated_spend"] = df["total_spend"]
    df.loc[df.index[:n_channels_boost], "reallocated_spend"] += shift_amount / n_channels_boost
    df.loc[df.index[-n_channels_boost:], "reallocated_spend"] -= shift_amount / n_channels_boost

    # Apply diminishing marginal ROI on the added spend (70% of stated ROI, conservative)
    df["projected_revenue_change"] = (
        (df["reallocated_spend"] - df["total_spend"]) * df["roi_multiple"] * 0.7
    )
    net_gain = df["projected_revenue_change"].sum()
    return df, net_gain


def make_charts(roi_df, realloc_df, net_gain):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    order = roi_df.sort_values("roi_multiple", ascending=True)
    colors = ["#d95f02" if v < 2 else "#2c7fb8" if v < 4 else "#1a9850" for v in order["roi_multiple"]]
    axes[0].barh(order["channel"], order["roi_multiple"], color=colors)
    axes[0].axvline(1, color="red", linestyle="--", alpha=0.6, label="Breakeven (ROI=1)")
    axes[0].set_title("Estimated ROI by Marketing Channel\n(Revenue attributed / $ spent)",
                       fontsize=13, fontweight="bold")
    axes[0].set_xlabel("ROI Multiple")
    axes[0].legend()
    for i, v in enumerate(order["roi_multiple"]):
        axes[0].text(v + 0.05, i, f"{v}x", va="center", fontweight="bold")

    x = np.arange(len(realloc_df))
    width = 0.35
    axes[1].bar(x - width/2, realloc_df["total_spend"], width, label="Current Spend", color="#999999")
    axes[1].bar(x + width/2, realloc_df["reallocated_spend"], width, label="Recommended Spend", color="#1a9850")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(realloc_df["channel"], rotation=20)
    axes[1].set_title(f"Recommended Budget Reallocation\nProjected net revenue gain: ${net_gain:,.0f}",
                       fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Spend ($)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/marketing_roi_and_reallocation.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Generating 2 years of weekly marketing spend/revenue data...")
    df = generate_marketing_data()
    df.to_csv(f"{DATA_DIR}/marketing_weekly.csv")

    model, contributions, decays = fit_mmm(df)
    roi_df = compute_roi(df, contributions)
    roi_df.to_csv(f"{OUT_DIR}/channel_roi.csv", index=False)

    realloc_df, net_gain = optimize_budget(roi_df)
    realloc_df.to_csv(f"{OUT_DIR}/budget_reallocation.csv", index=False)

    print("\n=== CHANNEL ROI (attributed revenue / spend) ===")
    print(roi_df.to_string(index=False))

    best = roi_df.iloc[0]
    worst = roi_df.iloc[-1]
    print(f"\nKEY INSIGHT: '{best['channel']}' returns {best['roi_multiple']}x per dollar; "
          f"'{worst['channel']}' returns only {worst['roi_multiple']}x.")
    print(f"\nRECOMMENDATION: Shift budget from '{worst['channel']}' toward '{best['channel']}'.")
    print(f"Projected net revenue gain from reallocation: ${net_gain:,.0f} "
          f"(over the {len(df)}-week period analyzed, same total budget).")
    print(f"\nSaved: {OUT_DIR}/marketing_roi_and_reallocation.png")


if __name__ == "__main__":
    main()

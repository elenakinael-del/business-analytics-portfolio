"""
Customer Segmentation & Lifetime Value (CLV) Analysis
=======================================================
Business question: Which customer segments drive the most revenue, and how
much should we spend to acquire/retain each one?

Method:
  1. Generate a realistic e-commerce transaction log (synthetic, but with
     realistic RFM distributions — see data_generator.py for assumptions).
  2. Compute RFM (Recency, Frequency, Monetary) features per customer.
  3. Cluster customers with K-Means into actionable segments.
  4. Project 12-month CLV per segment using a simple BG/NBD-style heuristic
     (purchase frequency x average order value x expected lifetime).
  5. Output: segment profile chart, CLV-by-segment chart, and a ranked
     business recommendation table.

Author: Elena | github.com/elenakinael-del
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", palette="deep")
RNG = np.random.default_rng(42)

OUT_DIR = "outputs"
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# 1. Data generation (synthetic transaction log with realistic structure)
# ---------------------------------------------------------------------------
def generate_transactions(n_customers=2500, seed=42):
    """
    Simulates 18 months of e-commerce transactions.
    Customer behavior is drawn from a mixture of archetypes (champions,
    loyal, at-risk, one-time, low-value) so clustering has real signal
    to find rather than pure noise.
    """
    rng = np.random.default_rng(seed)
    archetypes = rng.choice(
        ["champion", "loyal", "at_risk", "new", "low_value"],
        size=n_customers,
        p=[0.08, 0.22, 0.15, 0.20, 0.35],
    )

    params = {
        "champion":  dict(freq_lambda=1.8, aov_mean=180, aov_sd=40, recency_max=15),
        "loyal":     dict(freq_lambda=1.0, aov_mean=110, aov_sd=30, recency_max=40),
        "at_risk":   dict(freq_lambda=0.6, aov_mean=95,  aov_sd=35, recency_max=150),
        "new":       dict(freq_lambda=0.5, aov_mean=70,  aov_sd=25, recency_max=30),
        "low_value": dict(freq_lambda=0.3, aov_mean=45,  aov_sd=20, recency_max=200),
    }

    rows = []
    today = pd.Timestamp("2026-09-01")
    for cid, arch in enumerate(archetypes):
        p = params[arch]
        n_orders = max(1, rng.poisson(p["freq_lambda"] * 12))
        recency = rng.integers(1, p["recency_max"] + 1)
        order_days_ago = np.sort(rng.integers(recency, 540, size=n_orders))
        for d in order_days_ago:
            order_value = max(10, rng.normal(p["aov_mean"], p["aov_sd"]))
            rows.append({
                "customer_id": cid,
                "archetype_true": arch,  # kept only for validation, not used in clustering
                "order_date": today - pd.Timedelta(days=int(d)),
                "order_value": round(order_value, 2),
            })
    return pd.DataFrame(rows)


def compute_rfm(df, snapshot_date):
    grouped = df.groupby("customer_id").agg(
        last_order=("order_date", "max"),
        frequency=("order_date", "count"),
        monetary=("order_value", "mean"),
        total_spent=("order_value", "sum"),
        archetype_true=("archetype_true", "first"),
    )
    grouped["recency_days"] = (snapshot_date - grouped["last_order"]).dt.days
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# 2. Clustering
# ---------------------------------------------------------------------------
def segment_customers(rfm, k=5):
    features = rfm[["recency_days", "frequency", "monetary"]].copy()
    features["recency_days"] = np.log1p(features["recency_days"])
    features["frequency"] = np.log1p(features["frequency"])
    features["monetary"] = np.log1p(features["monetary"])

    X = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm["segment"] = km.fit_predict(X)

    # Label segments by business meaning using cluster centroids (not hardcoded order)
    profile = rfm.groupby("segment")[["recency_days", "frequency", "monetary"]].mean()
    profile["score"] = profile["frequency"].rank() + profile["monetary"].rank() - profile["recency_days"].rank()
    order = profile.sort_values("score", ascending=False).index.tolist()
    labels = ["Champions", "Loyal", "Potential Loyalists", "At Risk", "Low Value"][:k]
    label_map = dict(zip(order, labels))
    rfm["segment_label"] = rfm["segment"].map(label_map)
    return rfm


# ---------------------------------------------------------------------------
# 3. CLV projection (simple, transparent heuristic — documented, not a black box)
# ---------------------------------------------------------------------------
def project_clv(rfm, horizon_months=12, gross_margin=0.35):
    """
    CLV = expected orders in horizon x AOV x gross margin
    Expected orders scaled down for higher recency (churn-adjusted).
    This is intentionally simple/explainable for a business audience —
    a full BG/NBD or Pareto/NBD model would replace this in production.
    """
    annual_order_rate = rfm["frequency"] / 1.5  # observed window ~18 months -> annualize
    churn_adjustment = np.exp(-rfm["recency_days"] / 180)  # decays with inactivity
    expected_orders = annual_order_rate * (horizon_months / 12) * churn_adjustment
    rfm["clv_12m"] = (expected_orders * rfm["monetary"] * gross_margin).round(2)
    return rfm


# ---------------------------------------------------------------------------
# 4. Reporting
# ---------------------------------------------------------------------------
def build_report(rfm):
    seg_summary = rfm.groupby("segment_label").agg(
        customers=("customer_id", "count"),
        avg_recency=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_order_value=("monetary", "mean"),
        total_clv_12m=("clv_12m", "sum"),
        avg_clv_12m=("clv_12m", "mean"),
    ).sort_values("total_clv_12m", ascending=False)

    seg_summary["pct_of_customers"] = (seg_summary["customers"] / seg_summary["customers"].sum() * 100).round(1)
    seg_summary["pct_of_clv"] = (seg_summary["total_clv_12m"] / seg_summary["total_clv_12m"].sum() * 100).round(1)
    return seg_summary.round(2)


def make_charts(rfm, seg_summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    order = seg_summary.index.tolist()
    sns.barplot(x=seg_summary.index, y=seg_summary["pct_of_clv"], ax=axes[0],
                order=order, palette="viridis")
    axes[0].set_title("Share of Projected 12-Month CLV by Segment", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("% of Total CLV")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(seg_summary["pct_of_clv"]):
        axes[0].text(i, v + 0.5, f"{v}%", ha="center", fontweight="bold")

    sns.scatterplot(
        data=rfm, x="frequency", y="monetary", hue="segment_label",
        size="clv_12m", sizes=(20, 300), alpha=0.6, ax=axes[1], palette="viridis"
    )
    axes[1].set_title("Customer Map: Frequency vs. Avg Order Value\n(bubble size = projected CLV)",
                       fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Purchase Frequency (18mo)")
    axes[1].set_ylabel("Average Order Value ($)")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/segmentation_clv_overview.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Generating transaction data...")
    tx = generate_transactions()
    tx.to_csv(f"{DATA_DIR}/transactions.csv", index=False)

    snapshot = tx["order_date"].max() + pd.Timedelta(days=1)
    rfm = compute_rfm(tx, snapshot)
    rfm = segment_customers(rfm, k=5)
    rfm = project_clv(rfm)
    rfm.to_csv(f"{DATA_DIR}/customer_rfm_segments.csv", index=False)

    seg_summary = build_report(rfm)
    seg_summary.to_csv(f"{OUT_DIR}/segment_summary.csv")

    make_charts(rfm, seg_summary)

    print("\n=== SEGMENT SUMMARY ===")
    print(seg_summary[["customers", "pct_of_customers", "avg_clv_12m", "pct_of_clv"]].to_string())

    top_seg = seg_summary.index[0]
    top_pct = seg_summary.loc[top_seg, "pct_of_customers"]
    top_clv_pct = seg_summary.loc[top_seg, "pct_of_clv"]
    print(f"\nKEY INSIGHT: '{top_seg}' = {top_pct}% of customers but "
          f"{top_clv_pct}% of projected 12-month CLV.")
    print("BUSINESS RECOMMENDATION: Reallocate retention budget toward this "
          "segment and build a lookalike acquisition model from its RFM profile.")
    print(f"\nSaved: {OUT_DIR}/segmentation_clv_overview.png, {OUT_DIR}/segment_summary.csv")


if __name__ == "__main__":
    main()

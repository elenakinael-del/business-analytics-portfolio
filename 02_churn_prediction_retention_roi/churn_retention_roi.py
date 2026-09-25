"""
Churn Prediction & Retention Campaign ROI
==========================================
Business question: Which customers are about to churn, and is it actually
profitable to spend money trying to keep them?

Method:
  1. Generate a realistic subscription-business dataset (usage, tenure,
     support tickets, plan tier -> churn label).
  2. Train a classifier (Logistic Regression + Random Forest comparison)
     to predict churn probability.
  3. Rank customers by churn risk x their value, not risk alone.
  4. Simulate a retention campaign and compute expected ROI: cost of
     outreach vs. expected revenue saved, to show the model actually
     changes a spending decision.

Author: Elena | github.com/elenakinael-del
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

sns.set_theme(style="whitegrid", palette="deep")
OUT_DIR = "outputs"
DATA_DIR = "data"


def generate_subscribers(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    tenure_months = rng.gamma(3, 6, n).clip(1, 60)
    monthly_spend = rng.normal(45, 15, n).clip(9, 200)
    logins_last_30d = rng.poisson(8, n)
    support_tickets_90d = rng.poisson(0.6, n)
    plan_tier = rng.choice(["basic", "pro", "enterprise"], n, p=[0.55, 0.35, 0.10])
    nps_score = rng.normal(30, 25, n).clip(-100, 100)

    # Churn probability driven by realistic causal structure, not random noise
    logit = (
        -0.6
        - 0.07 * tenure_months
        - 0.14 * logins_last_30d
        + 0.5 * support_tickets_90d
        - 0.018 * nps_score
        + np.where(plan_tier == "basic", 0.55, np.where(plan_tier == "pro", 0.0, -0.65))
        + rng.normal(0, 0.4, n)
    )
    churn_prob_true = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_prob_true)

    df = pd.DataFrame({
        "customer_id": range(n),
        "tenure_months": tenure_months.round(1),
        "monthly_spend": monthly_spend.round(2),
        "logins_last_30d": logins_last_30d,
        "support_tickets_90d": support_tickets_90d,
        "plan_tier": plan_tier,
        "nps_score": nps_score.round(1),
        "churned": churned,
    })
    return df


def train_models(df):
    X = pd.get_dummies(df.drop(columns=["customer_id", "churned"]), columns=["plan_tier"], drop_first=True)
    y = df["churned"]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(X_train, y_train)

    results = {}
    for name, model, Xte in [("Logistic Regression", log_reg, X_test_s), ("Random Forest", rf, X_test)]:
        proba = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(y_test, proba)
        results[name] = {"model": model, "proba": proba, "auc": auc}

    best_name = max(results, key=lambda k: results[k]["auc"])
    return results, best_name, X_test, y_test, idx_test, rf, X.columns


def retention_roi_simulation(df, idx_test, proba, y_test, outreach_cost=15, save_rate_if_contacted=0.35,
                              retained_value_months=6):
    """
    Simulates: if we contact the top-N highest-risk-x-value customers,
    what's the expected ROI, and where's the break-even point?
    """
    sim = df.loc[idx_test].copy()
    sim["churn_proba"] = proba
    sim["expected_value_at_risk"] = sim["churn_proba"] * sim["monthly_spend"] * retained_value_months
    sim = sim.sort_values("expected_value_at_risk", ascending=False).reset_index(drop=True)

    n_customers = len(sim)
    contact_fractions = np.arange(0.02, 0.51, 0.02)
    rows = []
    for frac in contact_fractions:
        n_contacted = max(1, int(n_customers * frac))
        cohort = sim.iloc[:n_contacted]
        cost = n_contacted * outreach_cost
        revenue_saved = (cohort["expected_value_at_risk"] * save_rate_if_contacted).sum()
        roi = (revenue_saved - cost) / cost
        rows.append({"pct_contacted": frac * 100, "n_contacted": n_contacted,
                      "cost": cost, "revenue_saved": round(revenue_saved, 2), "roi": roi})
    roi_df = pd.DataFrame(rows)
    return sim, roi_df


def make_charts(results, y_test, roi_df, rf_model, feature_names):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # ROC curves
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("Model Comparison: ROC Curve", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)

    # Feature importance (RF)
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    importances.plot(kind="barh", ax=axes[1], color=sns.color_palette("viridis", len(importances)))
    axes[1].set_title("What Predicts Churn?\n(Random Forest Feature Importance)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Importance")

    # ROI curve
    axes[2].plot(roi_df["pct_contacted"], roi_df["roi"] * 100, marker="o", color="#2c7fb8", linewidth=2)
    axes[2].axhline(0, color="red", linestyle="--", alpha=0.6, label="Break-even")
    best = roi_df.loc[roi_df["roi"].idxmax()]
    axes[2].scatter([best["pct_contacted"]], [best["roi"] * 100], color="green", s=100, zorder=5,
                     label=f"Optimal: contact top {best['pct_contacted']:.0f}%")
    axes[2].set_title("Retention Campaign ROI\nby % of At-Risk Customers Contacted", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("% of Customers Contacted (ranked by risk x value)")
    axes[2].set_ylabel("ROI (%)")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/churn_model_and_roi.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Generating subscriber data...")
    df = generate_subscribers()
    df.to_csv(f"{DATA_DIR}/subscribers.csv", index=False)
    print(f"Base churn rate: {df['churned'].mean()*100:.1f}%")

    results, best_name, X_test, y_test, idx_test, rf_model, feature_names = train_models(df)
    print(f"\nModel AUCs: " + ", ".join(f"{k}={v['auc']:.3f}" for k, v in results.items()))
    print(f"Best model: {best_name}")

    best_proba = results[best_name]["proba"]
    sim, roi_df = retention_roi_simulation(df, idx_test, best_proba, y_test)
    roi_df.to_csv(f"{OUT_DIR}/retention_roi_curve.csv", index=False)

    best_row = roi_df.loc[roi_df["roi"].idxmax()]
    make_charts(results, y_test, roi_df, rf_model, feature_names)

    print(f"\n=== RETENTION CAMPAIGN RECOMMENDATION ===")
    print(f"Optimal outreach: contact top {best_row['pct_contacted']:.0f}% of at-risk customers "
          f"({int(best_row['n_contacted'])} customers)")
    print(f"Campaign cost: ${best_row['cost']:,.0f} | Expected revenue saved: ${best_row['revenue_saved']:,.0f}")
    print(f"Expected ROI: {best_row['roi']*100:.0f}%")
    print(f"\nBEYOND this point, ROI declines — contacting low-risk customers wastes spend.")
    print(f"\nSaved: {OUT_DIR}/churn_model_and_roi.png, {OUT_DIR}/retention_roi_curve.csv")


if __name__ == "__main__":
    main()

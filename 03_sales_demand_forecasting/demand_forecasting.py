"""
Sales / Demand Forecasting with Inventory Cost Impact
========================================================
Business question: How much inventory should we hold next quarter, and
what does a better forecast save us in carrying + stockout costs?

Method:
  1. Generate 3 years of daily sales data with trend, weekly seasonality,
     yearly seasonality, and promo spikes (realistic retail pattern).
  2. Forecast with 3 models: Naive (last-value), SARIMA, and Holt-Winters
     Exponential Smoothing — compared on held-out test data.
  3. Translate forecast error (MAPE) directly into $ cost using a simple
     newsvendor-style inventory cost model (overstock vs. stockout cost).

Author: Elena | github.com/elenakinael-del
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
OUT_DIR = "outputs"
DATA_DIR = "data"


def generate_sales(days=1095, seed=11):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-09-01", periods=days, freq="D")
    t = np.arange(days)

    trend = 200 + 0.08 * t
    weekly = 40 * np.sin(2 * np.pi * t / 7 - 1.2)
    yearly = 60 * np.sin(2 * np.pi * t / 365.25 + 1.5)
    promo_days = rng.choice(days, size=days // 45, replace=False)
    promo_spike = np.zeros(days)
    promo_spike[promo_days] = rng.uniform(80, 160, size=len(promo_days))
    noise = rng.normal(0, 18, days)

    sales = (trend + weekly + yearly + promo_spike + noise).clip(min=20)
    df = pd.DataFrame({"date": dates, "units_sold": sales.round().astype(int)})
    return df.set_index("date")


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def run_forecasts(df, test_days=60):
    train, test = df.iloc[:-test_days], df.iloc[-test_days:]
    y_train, y_test = train["units_sold"], test["units_sold"]

    # 1. Naive baseline
    naive_pred = pd.Series(y_train.iloc[-1], index=test.index)

    # 2. Holt-Winters (additive trend + weekly seasonality)
    hw_model = ExponentialSmoothing(
        y_train, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    hw_pred = hw_model.forecast(test_days)

    # 3. SARIMA
    sarima_model = SARIMAX(
        y_train, order=(2, 1, 1), seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)
    sarima_pred = sarima_model.forecast(test_days)

    results = {
        "Naive": {"pred": naive_pred, "mape": mape(y_test, naive_pred)},
        "Holt-Winters": {"pred": hw_pred, "mape": mape(y_test, hw_pred)},
        "SARIMA": {"pred": sarima_pred, "mape": mape(y_test, sarima_pred)},
    }
    return train, test, results


def inventory_cost_impact(y_test, results, unit_cost=12, stockout_penalty=28, holding_cost=2.5):
    """
    Newsvendor-style cost model: forecasting too LOW -> stockouts (lost margin),
    forecasting too HIGH -> excess holding cost. Translates MAPE into real $.
    """
    rows = []
    for name, r in results.items():
        pred = r["pred"]
        error = pred.values - y_test.values
        overstock = np.clip(error, 0, None).sum() * holding_cost
        understock = np.clip(-error, 0, None).sum() * stockout_penalty
        total_cost = overstock + understock
        rows.append({
            "model": name, "mape_pct": round(r["mape"], 2),
            "overstock_cost": round(overstock, 0),
            "stockout_cost": round(understock, 0),
            "total_inventory_cost": round(total_cost, 0),
        })
    cost_df = pd.DataFrame(rows).sort_values("total_inventory_cost")
    return cost_df


def make_charts(train, test, results, cost_df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(train.index[-120:], train["units_sold"].iloc[-120:], color="gray",
                 label="Historical (train)", linewidth=1.2)
    axes[0].plot(test.index, test["units_sold"], color="black", label="Actual", linewidth=2)
    colors = {"Naive": "#999999", "Holt-Winters": "#2c7fb8", "SARIMA": "#d95f02"}
    for name, r in results.items():
        axes[0].plot(test.index, r["pred"], label=f"{name} (MAPE={r['mape']:.1f}%)",
                     linewidth=2, linestyle="--", color=colors.get(name))
    axes[0].set_title("Demand Forecast: Actual vs. Model Predictions (60-day holdout)",
                       fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Units Sold / Day")
    axes[0].legend(fontsize=9)

    bars = axes[1].bar(cost_df["model"], cost_df["total_inventory_cost"],
                        color=[colors.get(m) for m in cost_df["model"]])
    axes[1].set_title("Inventory Cost Impact by Forecast Model (60-day period)",
                       fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Total Inventory Cost ($)")
    for bar, val in zip(bars, cost_df["total_inventory_cost"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 50, f"${val:,.0f}",
                     ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/forecast_and_cost_impact.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Generating 3 years of daily sales data...")
    df = generate_sales()
    df.to_csv(f"{DATA_DIR}/daily_sales.csv")

    train, test, results = run_forecasts(df)
    cost_df = inventory_cost_impact(test["units_sold"], results)
    cost_df.to_csv(f"{OUT_DIR}/model_cost_comparison.csv", index=False)

    make_charts(train, test, results, cost_df)

    print("\n=== MODEL COMPARISON ===")
    print(cost_df.to_string(index=False))

    best = cost_df.iloc[0]
    worst = cost_df.iloc[-1]
    savings = worst["total_inventory_cost"] - best["total_inventory_cost"]
    print(f"\nBUSINESS IMPACT: Switching from '{worst['model']}' to '{best['model']}' "
          f"cuts inventory cost by ${savings:,.0f} over 60 days "
          f"(~${savings/60*365:,.0f}/year annualized) at this order volume.")
    print(f"\nSaved: {OUT_DIR}/forecast_and_cost_impact.png, {OUT_DIR}/model_cost_comparison.csv")


if __name__ == "__main__":
    main()

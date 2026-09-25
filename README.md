# Business Analytics Portfolio 

Quantitative research and business analytics case studies focused on one thing: **turning data into decisions that move revenue, cost, or risk.**

Each project follows the same structure, a real business question, a transparent method (no black boxes), a chart you can read in 10 seconds, and a dollar denominated recommendation. Every script runs end-to-end and regenerates its own data, analysis, and charts from scratch.

> **Note on data:** the datasets are synthetically generated with realistic statistical structure (documented in each script's data generator) rather than pulled from a live business, since this is a public portfolio. The methods, models, and business logic are exactly what I'd apply to real transaction/CRM/marketing data, swap in a real dataset with matching columns and the pipeline runs unchanged.

## Projects

| # | Project | Business Question | Techniques | Key Result |
|---|---------|-------------------|------------|------------|
| 01 | [Customer Segmentation & CLV](01_customer_segmentation_clv/) | Which customers should we spend retention budget on? | RFM analysis, K-Means clustering, CLV projection | Top 19.6% of customers drive 66% of projected 12-month value |
| 02 | [Churn Prediction & Retention ROI](02_churn_prediction_retention_roi/) | Is a retention campaign actually profitable? | Logistic Regression, Random Forest, ROC/AUC, ROI simulation | Optimal outreach strategy returns 200%+ ROI; over-contacting destroys it |
| 03 | [Sales / Demand Forecasting](03_sales_demand_forecasting/) | How much inventory should we hold? | SARIMA, Holt-Winters, naive baseline, cost-of-error modeling | Better model choice saves ~$122K/year in inventory cost at this volume — and the model with the *best* accuracy isn't always the cheapest |
| 04 | [A/B Testing Framework](04_ab_testing_framework/) | Should we ship this experiment? | Two-proportion z-test, Welch's t-test, power analysis, CIs | Statistically robust "ship it" call with a $5.5M annualized revenue projection |
| 05 | [Marketing Channel ROI](05_marketing_channel_roi/) | How should we reallocate marketing budget? | Adstock/saturation transforms, regularized regression, budget optimization | Reallocating spend toward the top channel projects $1M+ incremental revenue for the same total budget |

## What each project folder contains

```
0X_project_name/
├── *.py              # the full analysis — data generation, modeling, charting
├── README.md          # business framing, method notes, how to run it
├── data/               # generated input data (CSV)
└── outputs/            # charts (PNG) and result tables (CSV)
```

## Stack

Python · pandas · numpy · scikit-learn · statsmodels · scipy · matplotlib · seaborn

## How to run any project

```bash
pip install -r requirements.txt
cd 0X_project_name
python *.py
```

Each script prints its business recommendation to the console and saves charts/tables to `outputs/`.

## About me

Independent quantitative researcher and systematic trader (COMEX Gold Futures), currently completing an MSc in Financial Engineering (WorldQuant University) with CME Group certifications. This portfolio applies the same rigor I use in trading research — transparent methodology, honest error reporting, and results tied to a real decision — to general business analytics problems.

More research: [github.com/elenakinael-del](https://github.com/elenakinael-del)

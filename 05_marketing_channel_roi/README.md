# Marketing Mix Modeling: Channel ROI & Budget Reallocation

**Business question:** Across our marketing channels, which ones actually drive revenue — and how should we reallocate budget?

## Approach
1. Generate 2 years of weekly spend across 5 channels (search, social, display, TV, email) and total revenue, with realistic **adstock** (a channel's effect carries over into future weeks) and **saturation** (diminishing returns as spend increases) baked into the data-generating process — the two effects that make naive "revenue vs. spend" correlation misleading for marketing data.
2. Fit adstock transforms per channel, then a positive-constrained Ridge regression to estimate each channel's incremental (not just correlated) contribution to revenue.
3. Compute ROI per channel = attributed revenue ÷ spend.
4. Run a simple budget reallocation: shift spend from the lowest-ROI channels to the highest-ROI ones (same total budget), applying a conservative 70% discount to marginal ROI to account for diminishing returns on the added spend.

## Result

![Marketing ROI and reallocation](outputs/marketing_roi_and_reallocation.png)

Email (3.19x) and search ads (1.99x) are the clear performers. TV and social ads hover near breakeven, and display ads show effectively zero attributable incremental revenue in this data — its apparent effect is fully explained by other channels running concurrently.

**Recommendation:** Reallocate budget from display ads toward email and search. Projected net revenue gain from reallocation, same total budget: **+$1.0M** over the analyzed period.

## Run it
```bash
python marketing_mix_model.py
```
Outputs: `outputs/marketing_roi_and_reallocation.png`, `outputs/channel_roi.csv`, `outputs/budget_reallocation.csv`

# DSN Mart Sales Modeling Write-up

## EDA Insights
- Store overlap across train and test is complete, while there are 4 unseen products in test.
- `store_size` is missing by store, not row noise, which supports store-level imputation.
- The raw target is right-skewed, but `log1p` underperformed raw CV RMSE here.

## Feature List
- OOF product and store aggregates to reduce leakage.
- Smoothing for sparse product-format and product-tier interactions.
- Ordinal encodings for ordered store attributes.

## Model Progression
- Trivial baseline RMSE: 1481.748558
- LightGBM baseline RMSE: 1115.239757
- Tuned models: scaffolded in notebook, but not finalized in this environment.
- Ensemble: scaffolded in notebook, but not finalized in this environment.

## Leakage Safety
- Aggregates are built out-of-fold in the notebook; the product-store raw pair feature is omitted because max_pair_count = 1.

## Business Recommendation
- Store format and location tier are both predictive enough to matter, so DSN Mart should keep them as first-class merchandising and planning signals.

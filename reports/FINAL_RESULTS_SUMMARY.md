# Final ML Results Summary

## Dataset
- 2 synthetic demo PHCs × 10 medicine/form entries × 90 days.
- 1,800 daily inventory rows; 1,780 next-day forecasting examples.
- Chronological splits: 1,240 train / 260 validation / 280 test.
- All PHC-level visits, demand, inventory, orders, delivery delays, expiry batches, stock-outs and crisis flags are synthetic. Medicine/form names are grounded in the official NHM PHC essential medicine list; the public aggregate datasets were not used as row-level training observations because their grain and labels do not match the day-level PHC–medicine target.

## Demand regression model
- Model: `ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.8, random_state=42)`, with imputation and one-hot encoding inside a fitted pipeline.
- Selection: validation-only comparison using macro mean per-medicine normalized MAE. Final model refit on train + validation.
- Final test: MAE 2.501, RMSE 3.604, pooled R² 0.655, macro normalized MAE 26.49%.
- 7-day-average baseline: MAE 2.406, RMSE 3.397, pooled R² 0.693, macro normalized MAE 26.29%.
- Conclusion: model beats previous-day and 30-day-average baselines but does not beat the 7-day-average baseline on this held-out synthetic period.

## Next-day stock-out classifier
- Model: class-weighted Logistic Regression with imputers and one-hot encoding.
- Target: synthetic `Stockout_Flag` on target date. Model selection used average precision on validation, and a 0.65 decision threshold was chosen by validation F1; test data was not used to select the threshold.
- Final test support: 280 rows, 14 positive labels. Accuracy 98.57%, balanced accuracy 99.25%, precision 77.78%, recall 100%, F1 87.50%, average precision 0.901, ROC-AUC 0.995. Confusion counts: 262 true negatives, 4 false positives, 0 false negatives, 14 true positives.
- Baseline that predicts no stock-out: accuracy 95%, precision 0%, recall 0%, F1 0%.
- Caution: only 14 positive test examples; metrics are unstable and entirely synthetic. The reported classifier score is uncalibrated and is not evidence of live PHC risk accuracy.

## Inventory planning demo
The package includes a recursive multi-day demand forecast and a deterministic inventory projection for stock sufficiency, stock-out dates, order-up-to quantity and expiry risk. The reorder calculation is an explicit prototype policy (including a configured 25-day demand cover), not a government procurement rule.

## Data audit and smoke checks
- Synthetic data audit: PASS, zero balance/row alignment/duplicate/delivery-delay errors.
- Generator reproduces the included main CSV datasets byte-for-byte with the included seed.
- Package smoke test: both saved models load; finite non-negative demand predictions, next-day classifier score, recursive 7-day demand forecast and inventory projection all passed.
- Crisis-scenario sensitivity: the demand model does not increase forecasts for every medicine when the declared crisis flag is toggled, so crisis performance is not validated.

## Responsible claim
This is a complete runnable educational ML prototype built and evaluated on synthetic data. It is not production-ready and has not been validated on real PHC operations. To establish real-world performance, obtain authorized, de-identified daily facility-level medicine-demand, inventory, receipts, stock-out/unmet-demand and batch-expiry records; then retrain and evaluate on later dates and preferably held-out facilities.

# Model Card — PHC Medicine Next-Day Demand Model

## Intended use
Educational/research prototype for next-day medicine-demand regression and scenario-based inventory planning using synthetic PHC-like data. Not for operational procurement or clinical decisions without validation against genuine facility data.

## Model
- Estimator: scikit-learn `ExtraTreesRegressor` inside a fitted preprocessing pipeline.
- Target: `Next_Day_Demand`, numerical units specific to each medicine form.
- Fit: 1,500 chronological training + validation rows after selecting the model on validation data.
- Final test: 280 later-date rows, target dates 2025-03-18 to 2025-03-31.
- Selection criterion: macro mean per-medicine normalized MAE on the validation split.

## Inputs
Recent medicine-demand history (lagged and rolling features), current-day patient visits and flags, past visit-history summaries, medicine and PHC IDs, target weekday/month, estimated future target-day visits based on historical weekday means, and an explicit target-day crisis scenario flag.

The target-day crisis flag is valid as a feature only when the scenario is known/declared at forecast time. It must not be set from an unknown future event label in a real application.

## Evaluation
See `reports/test_metrics_comparison.csv` and `reports/test_metrics_by_medicine.csv`. On the held-out synthetic test period, the Extra Trees model's pooled MAE was 2.501 and RMSE 3.604. The 7-day-average baseline had a slightly lower MAE (2.406) and RMSE (3.397), so the ML model does not establish superior test performance over that baseline.

## Training data
All daily demand, inventory, visits, delivery, stock-out, batch, expiry and crisis values in the prototype were generated synthetically. Medicine/form names are referenced to the NHM Essential Medicine List for SHC & PHC Level (updated March 2020). Other public source files were not used as row-level training data due to different aggregation/time periods or label definitions.

## Limitations
- Small 90-day, 2-facility, 10-medicine simulation.
- Random noise and demand patterns are author-selected; no empirical calibration against real daily PHC data.
- Crisis periods are not represented in the final test period, so crisis generalization is not tested. A model-response sensitivity check also found that toggling the target crisis flag does not increase forecasts for every medicine; the current fitted model has no guaranteed monotonic crisis response.
- Regression metrics do not prove the model predicts actual real-world inventory stock-outs.
- Recursive multi-day forecasts assume future visit patterns can be approximated by historical same-weekday averages and assume no unknown future emergencies.

## Appropriate wording
“Built a synthetic-data prototype for next-day medicine-demand forecasting using an Extra Trees regression pipeline, compared it with rolling-average baselines, and implemented scenario-based stock sufficiency/reorder/expiry projections. Evaluation found the 7-day-average baseline slightly outperformed the ML model on the held-out synthetic test period.”

## Secondary model: next-day stock-out classifier
- Estimator: class-weighted Logistic Regression in a fitted preprocessing pipeline.
- Target: synthetic `Stockout_Flag` on `Target_Date`, joined by PHC/medicine/date. Features use prediction-date inventory and historical-demand information plus an explicitly declared target-day crisis scenario.
- Selection: highest validation average precision among tested classifiers; decision threshold selected by F1 on validation only.
- Held-out synthetic test: 280 rows, 14 positive labels; accuracy 98.57%, balanced accuracy 99.25%, precision 77.78%, recall 100%, F1 87.50%, average precision 0.901, ROC-AUC 0.995 (threshold 0.65). Confusion counts: 14 TP, 4 FP, 0 FN, 262 TN.
- Major caution: only 14 positive test labels make estimates unstable, and labels are simulated. `predict_proba` scores are not calibrated probabilities and do not establish real-world stock-out predictive performance.

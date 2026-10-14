# PHC Medicine Demand Forecasting & Inventory Planning — ML Package

## Status at a glance

This package contains two fitted models—a next-day medicine-demand regression pipeline and a next-day stock-out classifier—plus reproducible training/evaluation code, synthetic datasets, test reports, and demand/inventory-planning demos.

**Critical limitation:** all PHC-level demand, patient visits, inventory, deliveries, expiry events, stock-outs and crisis events in this package are **synthetic**. This is a software/model prototype, not a model validated on real PHC operations.

The final held-out test results do **not** show a clear advantage over a simple 7-day-average baseline. The test report preserves that result; no claims of real-world accuracy are made.

## Files you should know

| Path | Purpose |
|---|---|
| `models/medicine_demand_model.joblib` | Trained fitted preprocessing + Extra Trees regression pipeline. |
| `models/model_metadata.json` | Demand model type, feature list, split periods, software versions and limitations. |
| `models/stockout_next_day_classifier.joblib` | Fitted next-day stock-out classifier preprocessing + Logistic Regression pipeline. |
| `models/stockout_classifier_metadata.json` | Classifier target, input features, selected threshold, support counts and limitations. |
| `train_and_evaluate.py` | Demand-regression model comparison, validation-only model selection, final refit and held-out test evaluation. |
| `train_stockout_classifier.py` | Trains and evaluates a separate next-day stock-out classifier with precision/recall/F1 and imbalance-aware baselines. |
| `predict_stockout.py` | Scores next-day synthetic stock-out risk for a chosen demo PHC/medicine. |
| `forecast_inventory.py` | CLI for arbitrary-horizon demand forecasts and inventory/reorder/expiry scenario calculations. |
| `verify_package.py` | Smoke test for both fitted models, demand forecasting and inventory projection. |
| `src/phc_ml.py` | Shared feature engineering, recursive forecasting and inventory simulation code. |
| `data/ml_train.csv` | Training split (1,240 rows). |
| `data/ml_validation.csv` | Chronological validation split (260 rows). |
| `data/ml_test.csv` | Later chronological test split (280 rows). |
| `data/ml_training_full.csv` | Full set of 1,780 next-day forecast records. |
| `data/DATA_GENERATION_README.md` | Configuration, field definitions and internal data validation rules. |
| `data/phc_daily_inventory.csv` | 1,800 synthetic PHC–medicine–date inventory records. |
| `data/purchase_orders.csv` | Simulated purchase orders and delivery dates/delays. |
| `data/medicine_batches.csv` | Simulated batch, dispensing, remaining quantity and expiry records. |
| `data/demand_generation_audit.csv` | Generator-internal factor log. **Diagnostic only; not used as ML input.** |
| `data/crisis_counterfactual_comparison.csv` | Simulated demand with/without crisis multiplier under the same sampled conditions. |
| `reports/noise_sensitivity_summary.csv`, `reports/noise_sensitivity_by_medicine.csv`, and `reports/noise_sensitivity_report.txt` | Diagnostic comparison of noise assumptions. |
| `data/medicine_catalogue.csv` | Ten selected medicine/form names plus explicitly synthetic demand/multiplier assumptions. |
| `data/reference_sources_register.csv` | References reviewed and an explanation of whether/how each was used. |
| `reports/validation_model_comparison.csv` | Candidate model selection results on validation data only. |
| `reports/test_metrics_comparison.csv` | Final test metrics for the selected model and simple baselines. |
| `reports/test_metrics_by_medicine.csv` | Final test error metrics by medicine. |
| `reports/test_predictions.csv` | Actual vs predicted held-out demand test values. |
| `reports/stockout_classifier_test_metrics.csv` and `reports/stockout_classifier_evaluation.txt` | Classifier precision/recall/F1, PR-AUC, ROC-AUC, confusion counts and no-stock-out baseline. |
| `reports/stockout_classifier_test_predictions.csv` | Held-out stock-out labels, classifier scores and thresholded predictions. |
| `reports/stockout_classifier_test_by_medicine.csv` | Per-medicine classifier results (small per-medicine supports; interpret cautiously). |
| `reports/crisis_scenario_forecast_sensitivity.csv` and `.txt` | Diagnostic of how the demand model responds when the declared crisis scenario is toggled. |
| `reports/model_evaluation_summary.txt` | Human-readable demand-regression evaluation summary. |
| `reports/FINAL_RESULTS_SUMMARY.md` | Compact summary of both models, metrics, data provenance and limitations. |
| `reports/feature_importance.csv` | Tree-model feature importance (associational, not causal). |
| `reports/crisis_scenario_forecast_sensitivity.csv` and `.txt` | Diagnostic of how the fitted model responds when the declared crisis scenario is toggled. |
| `reports/synthetic_data_audit_report.txt` | Data consistency audit. |
| `reference/NHM_Essential_Medicine_List_SHC_PHC_March_2020.pdf` | Medicine-list reference copy. |

## Backend API and UI integration

Use **`api.py`** as the backend bridge between the ML models and a UI. It exposes JSON endpoints for health checks, PHC/medicine catalogues, demand forecasts with inventory planning, and next-day stock-out scoring. The underlying forecasting functions are in `src/phc_ml.py`; the serialized models are in `models/`.

See [`BACKEND_UI_INTEGRATION.md`](BACKEND_UI_INTEGRATION.md) for Windows commands, request examples, and how to connect a React/Vite UI or an existing Node.js/Express backend. A minimal React example is in `frontend/PHCForecastExample.jsx`. If your application already uses Node/Express, see `integrations/express_ml_proxy.js` and `BACKEND_UI_INTEGRATION.md` for a proxy-route example.

Run the API from the package root after installing the requirements:

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` to try the endpoints. Run `python verify_api.py` for API smoke tests.

## Setup (Windows / VS Code)

Use Python 3.12 or another version compatible with the pinned requirements. From the extracted package root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The already-trained model is included; you do **not** need to train again to run the demo.

Run the package checks:

```powershell
python verify_package.py
```

Reproduce demand-model selection, training and evaluation (this overwrites demand model/report outputs):

```powershell
python train_and_evaluate.py
```

Reproduce the next-day stock-out classifier training and evaluation:

```powershell
python train_stockout_classifier.py
```

Re-run data-audit, noise-sensitivity, and crisis-response diagnostics:

```powershell
python scripts/audit_synthetic_data.py
python scripts/analyze_noise_sensitivity.py
python scripts/evaluate_crisis_scenario_response.py
```

Example 14-day demand and stock-planning forecast:

```powershell
python forecast_inventory.py --phc-id PHC-DEMO-001 --medicine-id MED001 --horizon 14
```

Example 14-day scenario where a crisis is declared for every forecast target date:

```powershell
python forecast_inventory.py --phc-id PHC-DEMO-002 --medicine-id MED001 --horizon 14 --crisis
```

Example next-day stock-out classification score (default crisis scenario off):

```powershell
python predict_stockout.py --phc-id PHC-DEMO-001 --medicine-id MED001
```

Add `--crisis` to score a declared-crisis scenario for the next target day. The reported score is not probability-calibrated.

Available demo PHCs are `PHC-DEMO-001` and `PHC-DEMO-002`. Medicine IDs are `MED001`–`MED010`. The bundled history ends on **2025-03-31**, so the bundled demo forecasts begin on **2025-04-01**. These are illustrative synthetic dates—not live current PHC forecasts.

## How the model was trained

1. Loaded and checked the synthetic chronological training, validation and test CSVs.
2. Engineered past-only visit-history and medicine-demand features. Calendar features are derived from the target date because the target weekday/month are known at prediction time.
3. Included `Target_Crisis_Flag` as an **explicit known/declared scenario input**. During training it comes from the synthetic scenario schedule. In a real application it must be set by a user/known planning scenario; never use an unknown future event label as though it were known.
4. Kept imputation and one-hot encoding inside the fitted scikit-learn pipeline to avoid fitting preprocessing on validation/test targets.
5. Compared Extra Trees, Random Forest and Gradient Boosting candidates on validation data using **macro mean per-medicine normalized MAE**: compute each medicine's MAE divided by that medicine's mean actual demand, then average across medicines. This prevents the higher-volume Paracetamol series from dominating selection.
6. Selected `ExtraTrees_400_leaf5` by validation score, then refit that configuration using training + validation rows.
7. Evaluated it on the later test set and compared against previous-day, 7-day-average and 30-day-average baselines.

### Selected model

`ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.8, random_state=42, n_jobs=-1)`, with median imputation (including missing indicators) for numerical features and most-frequent imputation + one-hot encoding for categorical features.

The regression model forecasts **next-day total demand**, including unmet demand in the synthetic labels. The inventory planner then calculates projected stock-out events by simulating that demand against usable stock, scheduled synthetic deliveries and known batch expiry. A separate classifier estimates next-day stock-out status from prediction-date features, but its output score is **not probability-calibrated** and does not establish a real-world stock-out probability.

## Final evaluation — read this before making claims

The reported test rows are 280 records with target dates 2025-03-18 through 2025-03-31. Final results:

| Method | Test MAE (pooled units; caution) | Test RMSE (pooled units; caution) | Macro mean per-medicine normalized MAE |
|---|---:|---:|---:|
| Extra Trees model | 2.501 | 3.604 | 26.49% |
| Previous-day baseline | 3.036 | 4.222 | 33.85% |
| 7-day-average baseline | **2.406** | **3.397** | **26.29%** |
| 30-day-average baseline | 2.512 | 3.527 | 27.67% |

Pooled MAE/RMSE combine tablets, capsules and sachets, so the pooled values are not a pure apples-to-apples physical-unit measure. The macro normalized measure averages relative errors across medicines. The ML model beat the previous-day and 30-day-average baselines, but did **not** beat the 7-day-average baseline on this held-out period. Treat this as a research finding/limitation, not something to hide.

### Regression metrics vs stock-out classification metrics

The demand forecaster predicts a numerical quantity, so MAE/RMSE/R² and per-medicine errors are appropriate—not classification precision/recall. Separately, the package contains a binary next-day stock-out classifier, so it reports accuracy, balanced accuracy, precision, recall, F1, average precision (PR-AUC), ROC-AUC, and confusion counts.

#### Next-day stock-out classifier — held-out synthetic test

The classifier was selected using average precision on the chronological validation set, and its decision threshold was selected on validation only by maximizing F1 on a coarse threshold grid. It was then refit on train + validation and evaluated on the later test set. Test support was only 14 positive labels among 280 rows.

| Metric | Logistic Regression (class-weight balanced) | Baseline: predict no stock-out |
|---|---:|---:|
| Accuracy | 98.57% | 95.00% |
| Balanced accuracy | 99.25% | 50.00% |
| Precision | 77.78% | 0.00% |
| Recall | 100.00% | 0.00% |
| F1 | 87.50% | 0.00% |
| Average precision (PR-AUC) | 0.901 | 0.050 |
| ROC-AUC | 0.995 | 0.500 |

The classifier recorded 14 true positives, 4 false positives and 0 false negatives on this tiny synthetic test. These high scores are not proof of real-world stock-out prediction: labels are generated by a simulator, features include current stock/reorder context, and only 14 positive test cases exist. Scores are not calibrated probabilities. The deterministic stock-planning projection remains a separate calculation from this classifier.

## Forecasting and inventory assumptions

- Recursive multi-day forecasts use a one-day model repeatedly. Future patient visits are estimated using observed historical same-weekday means; future emergency events are unknown and assumed absent during steps beyond the historical origin.
- The `--crisis` flag applies the declared crisis condition to every forecast target date. This interface makes the scenario explicit, but crisis response has not been validated on real data, and the single synthetic crisis period falls outside the final test dates. A package diagnostic found the fitted model does not increase forecasts for every medicine when the crisis flag is toggled: several medicines showed small decreases. This shows that the current model has not learned a consistently monotonic crisis response; do not rely on it for crisis forecasting without additional scenario training and evaluation against suitable data.
- Projected daily consumption is rounded up to whole units for conservative inventory simulation.
- FEFO is used for available, known batches. The planning tool conservatively treats a known batch as unavailable on/after its expiry date. Pending deliveries have synthetic scheduled receipt dates, but their batch expiry dates are unknown and are excluded from expiry-risk estimates.
- The reorder recommendation is a transparent prototype policy: flag reorder when a stock-out is projected within supplier lead time, projected stock at the lead-time boundary is at/below the reorder level, or current stock is already at/below the reorder level. It recommends order-up-to quantity based on an assumed 25-day demand cover. This is a configurable demo policy, not a validated government procurement rule.

## Data provenance

The ten medicine/form names were selected from the uploaded NHM Essential Medicine List for SHC & PHC Level (March 2020). PHC-level visit counts, daily demand, stock, deliveries, supplier delays, batch expiry, stock-outs and crisis flags were generated by the included simulator with seed `20261012`. The other public reference datasets were not merged into the model training rows because their reporting granularity/time period or medicine category labels did not match the daily PHC–medicine target. See `data/reference_sources_register.csv`.

## Limitations and responsible claims

- Only 2 simulated PHCs, 10 medicine/form entries and 90 simulated days are included. The total 1,780 ML rows are not equivalent to 1,780 independent facilities or real observations.
- Synthetic metrics for both models measure behaviour on the simulator's patterns. They do not establish real-world PHC accuracy. The stock-out classifier test set contains only 14 positive labels, so its precision/recall estimates are especially uncertain.
- The simulated demand assumptions and crisis multipliers were not calibrated from facility-level observed records.
- The test-set result did not outperform the 7-day-average baseline. The current model is a complete runnable prototype, but should not be described as production-ready or as proven to prevent real medicine stock-outs.
- For a defensible real-world model, replace/augment the synthetic labels with authorized, appropriately de-identified daily PHC medicine demand/dispensing records and actual stock, deliveries and expiry batches; define how unmet demand during stock-outs is measured; retrain; and validate on later real dates/facilities.

# Interview / Viva Notes

## Problem
A PHC needs to plan medicine replenishment and reduce shortages/expiry. The prototype forecasts next-day total demand with an Extra Trees regressor, and separately classifies next-day stock-out status with a class-weighted Logistic Regression model. It also projects inventory using forecast demand, stock, planned deliveries and batch expiry information.

## Why regression?
The target (`Next_Day_Demand`) is a numerical quantity, so this is a regression problem and MAE/RMSE are appropriate. We also trained a separate next-day stock-out classifier, for which precision, recall, F1, balanced accuracy and PR-AUC are appropriate. Its labels are synthetic and test positives are few.

## Why Extra Trees?
Candidate Extra Trees, Random Forest and Gradient Boosting models were compared on a chronological validation period. ExtraTrees_400_leaf5 had the lowest macro per-medicine normalized MAE among the candidates tested. Extra Trees averages predictions from many randomized decision trees, which can reduce the sensitivity of a single tree. It was not proven superior to the 7-day-average baseline on the final test period.

## Why chronological splits?
Forecasting means predicting future dates from past observations. Randomly shuffling dates can leak future patterns into training; train, validation and test periods are separated by target date.

## What is MAE?
Mean absolute error: mean of absolute differences between predicted and actual demand. It is in the medicine's unit (tablets/capsules/sachets).

## What is RMSE?
Root mean squared error; larger misses receive more weight because errors are squared before averaging and taking the square root.

## What is macro normalized MAE?
For each medicine, divide that medicine's MAE by its mean actual demand, then average those values across medicines. It helps avoid a high-volume medicine dominating model selection, but is a project-specific selection metric rather than a standard “accuracy percentage.”

## What about test results?
The test MAE/RMSE for Extra Trees were 2.501/3.604 pooled across mixed units. The 7-day average baseline was slightly better at 2.406/3.397. The correct interpretation is that the model beats the previous-day and 30-day baselines, but does not demonstrate an improvement over the 7-day baseline in this synthetic holdout. Do not hide that result.

## How is stock-out projected?
The tool recursively forecasts demand and simulates consumption against usable batches and pending scheduled deliveries. If forecasted demand exceeds projected usable stock, it marks a projected stock-out. This is a deterministic scenario, not a calibrated stock-out probability.

## What would be needed for a real deployment?
Actual daily medicine-level PHC demand/dispensing records, current stock, stock-out/unmet-demand data, genuine supplier lead times, batch/expiry records, and a temporal/external-facility validation. Synthetic metrics alone do not establish real-world performance.

## Crisis scenario limitation
A forecast sensitivity check toggles the declared `Target_Crisis_Flag` for a 14-day future horizon using the same starting history. The fitted model raises the mean forecast for some medicines but slightly lowers it for others. This shows that the fitted regression model does not implement a guaranteed monotonic crisis uplift. The diagnostic is a synthetic model-response check only; it does not validate crisis demand. A future version could compare scenario-specific models or impose a transparent policy adjustment, but any adjustment must be justified and evaluated—not added just to force an expected output.

## Stock-out classifier results and cautions
The classifier was selected on validation average precision and its threshold was selected by validation F1. On the final synthetic test (280 rows, only 14 positive stock-out labels), it achieved precision 0.778, recall 1.000, F1 0.875, balanced accuracy 0.992 and average precision 0.901. This is a small, synthetic test and is not evidence of real PHC performance; the positive-class score is not calibrated. Accuracy by itself is misleading because stock-outs are uncommon in the simulator.

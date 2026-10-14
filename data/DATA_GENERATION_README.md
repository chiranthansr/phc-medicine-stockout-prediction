# PHC Medicine Stock-Out Prediction — Synthetic Prototype Data

**IMPORTANT: All facility-level dates, demand, inventory, patients, orders, delays, batches, expiry events, stock-outs and crisis events in this package are synthetic demo data. They are not actual records from real PHCs.**

## Configuration
- PHCs: 2 demo facilities
- Medicines: 10 selected medicine/form entries
- Simulation period: 2025-01-01 through 2025-03-31 (90 days)
- Daily inventory rows: 1,800
- Next-day ML rows: 1,780
- Seed: 20261012 (repeatable generation)
- Crisis scenario: PHC-DEMO-002, 2025-02-15 through 2025-02-27
- Runtime: Python 3.9+; standard library only

## Files
- `phc_daily_inventory.csv`: one row per PHC, medicine and date; includes stock balance, received quantities, total demand, quantity dispensed, unmet demand, stock-outs, patient visits, emergency/crisis flags and expiry indicators.
- `purchase_orders.csv`: simulated replenishment orders, supplier lead times, expected/simulated delivery dates, quantities received and pending status.
- `medicine_batches.csv`: batch-level quantities received, receipt/expiry dates, dispensing, expired quantity and remaining quantity. FEFO is used for issuing stock.
- `crisis_counterfactual_comparison.csv`: paired synthetic comparison of demand with and without the crisis multiplier while holding other sampled factors constant.
- `demand_generation_audit.csv`: logs the factors used to generate each synthetic demand value, including random noise. **Diagnostic only; do not use as model inputs because it documents how the target was generated.**
- `ml_training_full.csv`: next-day demand forecasting dataset with features known on `Prediction_Date` and target `Next_Day_Demand`.
- `ml_train.csv`, `ml_validation.csv`, `ml_test.csv`: chronological splits grouped by target date; records are not randomly shuffled.
- `generate_phc_synthetic_data.py`: reproducible generator.

## Medicine catalogue
The ten medicine/form entries are selected from the PHC section of the uploaded NHM *Essential Medicine List for SHC & PHC Level* (updated March 2020): Paracetamol 500 mg tablet; Amoxicillin 500 mg capsule; ORS; Zinc sulphate 20 mg dispersible tablet; Amlodipine 5 mg tablet; Metformin 500 mg tablet; Albendazole 400 mg tablet; Ferrous salt 60 mg + folic acid 500 mcg tablet; Omeprazole 20 mg capsule; Salbutamol 2 mg tablet.

## Key field definitions
- `Total_Demand`: simulated required quantity, including quantity that could not be dispensed. This avoids teaching the model that low dispensing during a stock-out means low demand.
- `Dispensed_Quantity`: quantity issued from usable on-hand stock.
- `Unmet_Demand`: `Total_Demand - Dispensed_Quantity`.
- `Opening_Stock`: previous closing stock before today's expired stock is removed.
- `Expired_Quantity`: quantity removed from stock because the batch's expiry date passed. A batch remains usable through its expiry date in this simulator and is removed on the following day.
- `Available_Stock`: `Opening_Stock - Expired_Quantity + Quantity_Received`.
- `Closing_Stock`: `Available_Stock - Dispensed_Quantity`.
- `Stockout_Flag`: 1 when demand exceeds what can be dispensed.
- `Persistent_Demand_Shift_Flag`: a simple simulation indicator when demand deviates at least 25% from the prior seven-day average for five consecutive days. It is a prototype heuristic, not a validated clinical rule.
- `Next_Day_Demand`: target for forecasting; each feature row uses information available no later than `Prediction_Date`.
- `Source_Data_Type`: all rows are marked `SYNTHETIC`.

## Important limitations
1. This is a small 90-day logic prototype, not enough on its own to claim reliable real-world forecasting accuracy. Validate the rules, then scale the simulation and seek real facility data if available.
2. The PHC IDs and patient-visit figures are invented. Public state-level OPD/IPD totals were not converted directly into daily PHC patient counts because the reporting granularity/periods do not match.
3. Demand patterns, crisis multipliers, starting stock, lead times, delivery delays and shelf lives are configurable simulation assumptions, not measured ground truth.
4. The ML CSV contains a synthetic demand target. Real-world training requires trustworthy historical demand/dispensing data and an explicit method for estimating unmet demand during stock-outs.
5. Keep these synthetic files separate from all public source datasets and clearly label them as synthetic in your report.

## Validation performed
- Inventory balance: Opening Stock - Expired Quantity + Quantity Received - Dispensed Quantity = Closing Stock.
- Demand balance: Dispensed Quantity + Unmet Demand = Total Demand.
- Batch balance: Batch Quantity Received = Quantity Dispensed + Quantity Expired + Closing Quantity.
- Checked for non-negative quantities, duplicate PHC-medicine-date keys and expected record counts.

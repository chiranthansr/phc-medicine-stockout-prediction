#!/usr/bin/env python3
"""Audit consistency and simulated patterns in the PHC prototype data.

This script checks internal rules and summarises patterns. It does NOT prove
that synthetic values match real PHC operations; that requires compatible real data.
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'data'
REPORTS = ROOT / 'reports'
DAILY_FILE = BASE / 'phc_daily_inventory.csv'
ML_FILE = BASE / 'ml_training_full.csv'
ORDERS_FILE = BASE / 'purchase_orders.csv'
BATCHES_FILE = BASE / 'medicine_batches.csv' 


def main():
    missing_files = [p.name for p in [DAILY_FILE, ML_FILE, ORDERS_FILE, BATCHES_FILE] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(f'Missing required files in {BASE}: {missing_files}')

    REPORTS.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(DAILY_FILE)
    ml = pd.read_csv(ML_FILE)
    orders = pd.read_csv(ORDERS_FILE)
    batches = pd.read_csv(BATCHES_FILE)

    for c in ['Date', 'Prediction_Date', 'Target_Date', 'Order_Date', 'Expected_Delivery_Date', 'Simulated_Delivery_Date', 'Actual_Delivery_Date', 'Receipt_Date', 'Expiry_Date']:
        for df in [daily, ml, orders, batches]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors='coerce')

    errors = []
    notes = []

    # Structural integrity checks
    key_cols = ['PHC_ID', 'Medicine_ID', 'Date']
    duplicate_keys = int(daily.duplicated(key_cols).sum())
    if duplicate_keys:
        errors.append(f'Duplicate PHC-medicine-date rows: {duplicate_keys}')
    expected_rows = daily['PHC_ID'].nunique() * daily['Medicine_ID'].nunique() * daily['Date'].nunique()
    if len(daily) != expected_rows:
        errors.append(f'Daily rows={len(daily)}, expected complete panel size={expected_rows}')
    source_types = sorted(daily['Source_Data_Type'].dropna().astype(str).unique().tolist())
    if source_types != ['SYNTHETIC']:
        notes.append(f'Source_Data_Type values are {source_types}; inspect provenance labels.')

    # Inventory and demand balance checks
    avail_calc = daily['Opening_Stock'] - daily['Expired_Quantity'] + daily['Quantity_Received']
    close_calc = daily['Available_Stock'] - daily['Dispensed_Quantity']
    demand_calc = daily['Dispensed_Quantity'] + daily['Unmet_Demand']
    stockout_calc = (daily['Unmet_Demand'] > 0).astype(int)
    checks = {
        'Available_Stock equation failures': int((avail_calc != daily['Available_Stock']).sum()),
        'Closing_Stock equation failures': int((close_calc != daily['Closing_Stock']).sum()),
        'Demand balance failures': int((demand_calc != daily['Total_Demand']).sum()),
        'Stockout flag failures': int((stockout_calc != daily['Stockout_Flag']).sum()),
    }
    for label, n in checks.items():
        if n:
            errors.append(f'{label}: {n}')
    qty_cols = ['Opening_Stock','Expired_Quantity','Quantity_Received','Available_Stock','Total_Demand','Dispensed_Quantity','Unmet_Demand','Closing_Stock']
    negative_count = int((daily[qty_cols] < 0).sum().sum())
    if negative_count:
        errors.append(f'Negative inventory/demand values: {negative_count}')

    continuity_failures = 0
    for _, g in daily.sort_values('Date').groupby(['PHC_ID','Medicine_ID']):
        g = g.sort_values('Date')
        if len(g) > 1:
            continuity_failures += int((g['Opening_Stock'].iloc[1:].to_numpy() != g['Closing_Stock'].iloc[:-1].to_numpy()).sum())
    if continuity_failures:
        errors.append(f'Daily opening/previous closing stock continuity failures: {continuity_failures}')

    batch_balance = batches['Quantity_Dispensed'] + batches['Quantity_Expired'] + batches['Closing_Quantity']
    batch_failures = int((batch_balance != batches['Batch_Quantity_Received']).sum())
    if batch_failures:
        errors.append(f'Batch balance failures: {batch_failures}')

    # Forecast construction: future target/date alignment and target is daily total demand
    day_delta = (ml['Target_Date'] - ml['Prediction_Date']).dt.days
    date_alignment_failures = int((day_delta != 1).sum())
    inv_index = daily.set_index(['PHC_ID','Medicine_ID','Date'])['Total_Demand']
    target_mismatch = 0
    prediction_lag_mismatch = 0
    for r in ml.itertuples(index=False):
        try:
            real_target = inv_index.loc[(r.PHC_ID, r.Medicine_ID, r.Target_Date)]
            if float(real_target) != float(r.Next_Day_Demand):
                target_mismatch += 1
            real_today = inv_index.loc[(r.PHC_ID, r.Medicine_ID, r.Prediction_Date)]
            if float(real_today) != float(r.Lag_1_Demand):
                prediction_lag_mismatch += 1
        except KeyError:
            target_mismatch += 1
    if date_alignment_failures:
        errors.append(f'Forecast prediction/target date alignment failures: {date_alignment_failures}')
    if target_mismatch:
        errors.append(f'ML target vs daily inventory mismatches: {target_mismatch}')
    if prediction_lag_mismatch:
        errors.append(f'Lag_1_Demand vs prediction-date demand mismatches: {prediction_lag_mismatch}')

    # Summary by medicine
    grp = daily.groupby(['Medicine_ID','Medicine_Name','Unit_Of_Measure'], dropna=False)
    prof = grp['Total_Demand'].agg(Mean_Demand='mean', Median_Demand='median', Std_Demand='std', Min_Demand='min', Max_Demand='max').reset_index()
    prof['Coefficient_of_Variation'] = prof['Std_Demand'] / prof['Mean_Demand'].replace(0, np.nan)
    corr = daily.groupby('Medicine_ID').apply(lambda g: g['Patient_Visits'].corr(g['Total_Demand']), include_groups=False).rename('Corr_Visits_vs_Demand').reset_index()
    prof = prof.merge(corr, on='Medicine_ID', how='left')
    prof['Stockout_Row_Rate'] = grp['Stockout_Flag'].mean().values
    prof = prof.round(3)
    prof.to_csv(REPORTS / 'synthetic_demand_profile.csv', index=False)

    # Day-to-day changes, computed per PHC-medicine sequence
    daily_sorted = daily.sort_values(['PHC_ID','Medicine_ID','Date']).copy()
    daily_sorted['Previous_Day_Demand'] = daily_sorted.groupby(['PHC_ID','Medicine_ID'])['Total_Demand'].shift(1)
    daily_sorted['Absolute_Change'] = (daily_sorted['Total_Demand'] - daily_sorted['Previous_Day_Demand']).abs()
    denominator = daily_sorted['Previous_Day_Demand'].replace(0, np.nan)
    daily_sorted['Absolute_Percent_Change'] = daily_sorted['Absolute_Change'] / denominator * 100
    jumps = daily_sorted.nlargest(25, 'Absolute_Change')[['Date','PHC_ID','Medicine_ID','Medicine_Name','Previous_Day_Demand','Total_Demand','Absolute_Change','Absolute_Percent_Change','Patient_Visits','Emergency_Flag','Crisis_Flag']]
    jumps.to_csv(REPORTS / 'synthetic_largest_demand_jumps.csv', index=False)

    # Crisis comparison for PHC-DEMO-002: ratio is descriptive only
    p2 = daily[daily['PHC_ID'].eq('PHC-DEMO-002')]
    crisis = p2[p2['Crisis_Flag'].eq(1)].groupby(['Medicine_ID','Medicine_Name'])['Total_Demand'].mean().rename('Mean_Demand_During_Crisis')
    normal = p2[p2['Crisis_Flag'].eq(0)].groupby(['Medicine_ID','Medicine_Name'])['Total_Demand'].mean().rename('Mean_Demand_Outside_Crisis')
    crisis_profile = pd.concat([crisis, normal], axis=1).reset_index()
    crisis_profile['Crisis_to_NonCrisis_Demand_Ratio'] = crisis_profile['Mean_Demand_During_Crisis'] / crisis_profile['Mean_Demand_Outside_Crisis']
    crisis_profile = crisis_profile.round(3)
    crisis_profile.to_csv(REPORTS / 'synthetic_crisis_comparison.csv', index=False)

    # Order date lag: calculate from dates rather than relying on Delivery_Delay_Days field
    order_delay = (orders['Simulated_Delivery_Date'] - orders['Expected_Delivery_Date']).dt.days
    delay_field_exists = 'Delivery_Delay_Days' in orders.columns
    delay_field_missing = int(orders['Delivery_Delay_Days'].isna().sum()) if delay_field_exists else len(orders)
    delay_invalid = 0
    delay_mismatches = 0
    if delay_field_exists:
        supplied_delay = pd.to_numeric(orders['Delivery_Delay_Days'], errors='coerce')
        delay_invalid = int(supplied_delay.isna().sum() + (supplied_delay < 0).sum())
        comparable = order_delay.notna() & supplied_delay.notna()
        delay_mismatches = int((order_delay[comparable].astype(int) != supplied_delay[comparable].astype(int)).sum())
    if delay_field_missing:
        notes.append(f'Delivery_Delay_Days contains {delay_field_missing} blank values; derive from delivery dates or fix the generator export.')
    if delay_mismatches:
        errors.append(f'Delivery_Delay_Days vs date-difference mismatches: {delay_mismatches}')
    if delay_invalid:
        errors.append(f'Delivery_Delay_Days blank/invalid values: {delay_invalid}')

    stockout_rate = daily['Stockout_Flag'].mean()
    unmet_total = int(daily['Unmet_Demand'].sum())
    expired_total = int(batches['Quantity_Expired'].sum())
    crisis_start = p2.loc[p2['Crisis_Flag'].eq(1), 'Date'].min()
    crisis_end = p2.loc[p2['Crisis_Flag'].eq(1), 'Date'].max()
    median_delay = float(order_delay.median()) if len(order_delay.dropna()) else float('nan')
    delayed_orders = int((order_delay > 0).sum())
    feature_missing = ml.isna().sum()
    feature_missing = feature_missing[feature_missing > 0]

    lines = [
        'SYNTHETIC PHC DATA AUDIT',
        '========================',
        'Purpose: check internal consistency and summarise generated patterns.',
        'This does NOT establish real-world representativeness; comparison with compatible real data is required.',
        '',
        f'Daily inventory rows: {len(daily)}',
        f'PHCs: {daily.PHC_ID.nunique()} | medicines: {daily.Medicine_ID.nunique()} | dates: {daily.Date.nunique()}',
        f'ML forecast rows: {len(ml)}',
        f'Purchase orders: {len(orders)} | medicine batches: {len(batches)}',
        f'Source_Data_Type values: {source_types}',
        '',
        'INTERNAL CONSISTENCY',
        *[f'- {k}: {v}' for k,v in checks.items()],
        f'- Negative quantity cells: {negative_count}',
        f'- Duplicate PHC-medicine-date rows: {duplicate_keys}',
        f'- Opening stock vs previous closing stock continuity failures: {continuity_failures}',
        f'- Batch balance failures: {batch_failures}',
        f'- Prediction/target must be one day apart failures: {date_alignment_failures}',
        f'- ML target vs daily inventory mismatches: {target_mismatch}',
        f'- Lag_1 demand vs prediction-date demand mismatches: {prediction_lag_mismatch}',
        f'- Overall validation status: {"PASS" if not errors else "FAIL"}',
        '',
        'GENERATED PATTERN SUMMARY (SYNTHETIC ONLY)',
        f'- Stockout rows: {int(daily.Stockout_Flag.sum())} ({stockout_rate:.2%} of daily rows)',
        f'- Total unmet demand units: {unmet_total}',
        f'- Total expired units recorded in batches: {expired_total}',
        f'- Mean daily demand across medicine-specific units: {daily.Total_Demand.mean():.2f} (not comparable across units/medicines)',
        f'- Overall Pearson correlation of patient visits vs demand pooled across medicines: {daily.Patient_Visits.corr(daily.Total_Demand):.3f} (interpret cautiously due to mixed medicine scales)',
        f'- Crisis period for PHC-DEMO-002: {crisis_start.date() if pd.notna(crisis_start) else "none"} to {crisis_end.date() if pd.notna(crisis_end) else "none"}',
        f'- Orders with simulated delivery after expected date: {delayed_orders}/{len(orders)}',
        f'- Median simulated order delay in days: {median_delay:.1f}',
        f'- Delivery_Delay_Days blank/invalid values: {delay_invalid}',
        f'- Delivery_Delay_Days vs date-difference mismatches: {delay_mismatches}',
        '',
        'MISSING VALUES IN ML DATASET',
        *([f'- {k}: {int(v)}' for k,v in feature_missing.items()] if len(feature_missing) else ['- None']),
        '',
        'OUTPUT FILES',
        '- synthetic_demand_profile.csv: statistics by medicine, visit-demand correlation, stock-out rates.',
        '- synthetic_largest_demand_jumps.csv: largest day-to-day simulated changes.',
        '- synthetic_crisis_comparison.csv: crisis versus non-crisis demand for PHC-DEMO-002.',
        '',
        'WARNINGS',
        '- Patient visits, demand baselines, crisis multipliers, noise, stock levels, lead times, delays and expiry assumptions were chosen by the simulator author; they are not empirically calibrated.',
        '- A low/high stockout rate in this simulation must not be presented as the rate in real PHCs.',
        '- Demand correlations within this dataset show what the generator created, not how real patient visits affect medicine demand.',
    ]
    if notes:
        lines += ['', 'NOTES', *[f'- {n}' for n in notes]]
    if errors:
        lines += ['', 'ERRORS', *[f'- {e}' for e in errors]]
    report_path = REPORTS / 'synthetic_data_audit_report.txt'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()

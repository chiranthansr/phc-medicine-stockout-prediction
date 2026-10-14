#!/usr/bin/env python3
"""Score next-day stock-out risk for one PHC/medicine using the bundled history.

The printed model probability is not calibrated, and this demo is trained/evaluated on
synthetic data only. Do not use it for live procurement decisions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.phc_ml import read_inventory, prepare_feature_rows

DATA=ROOT/'data'; MODELS=ROOT/'models'; REPORTS=ROOT/'reports'


def build_next_day_row(inventory: pd.DataFrame, phc_id: str, medicine_id: str, declared_crisis: bool) -> tuple[pd.DataFrame, dict]:
    hist=inventory[(inventory['PHC_ID']==phc_id)&(inventory['Medicine_ID']==medicine_id)].sort_values('Date').copy()
    if hist.empty:
        raise ValueError(f'No history for {phc_id} / {medicine_id}')
    origin=pd.Timestamp(hist['Date'].max())
    target=origin+pd.Timedelta(days=1)
    last=hist.iloc[-1]
    values=hist['Total_Demand'].astype(float).to_list()
    hist_dates=hist['Date'].to_list()
    lookup={pd.Timestamp(d):float(v) for d,v in zip(hist_dates, values)}
    mean7=float(np.mean(values[-7:]))
    mean30=float(np.mean(values[-30:]))
    mean90=float(np.mean(values[-90:]))
    prev7_stockouts=int(hist.tail(7)['Stockout_Flag'].sum())
    row={
        'Prediction_Date':origin,'Target_Date':target,'PHC_ID':phc_id,'Medicine_ID':medicine_id,
        'Medicine_Name':last['Medicine_Name'],'Unit_Of_Measure':last['Unit_Of_Measure'],
        'Day_Of_Week':origin.day_name(),'Month':origin.month,
        'Patient_Visits':int(last['Patient_Visits']),'Emergency_Flag':int(last['Emergency_Flag']),
        'Crisis_Flag':int(last['Crisis_Flag']),'Demand_Deviation_Percent':float(last['Demand_Deviation_Percent']),
        'Persistent_Demand_Shift_Flag':int(last['Persistent_Demand_Shift_Flag']),
        'Lag_1_Demand':float(values[-1]), 'Lag_7_Demand':lookup.get(origin-pd.Timedelta(days=7),np.nan),
        'Avg_7_Day_Demand':mean7,'Avg_30_Day_Demand':mean30,'Avg_90_Day_Demand':mean90,
        'Consumption_Trend_Ratio_7d_to_30d':mean7/max(mean30,1e-9),
        'Previous_Stockouts_Last_7_Days':prev7_stockouts,
        'Closing_Stock':int(last['Closing_Stock']),'Minimum_Stock_Level':int(last['Minimum_Stock_Level']),
        'Reorder_Level':int(last['Reorder_Level']),'Supplier_Lead_Time_Days':int(last['Supplier_Lead_Time_Days']),
        'Days_Since_Last_Delivery':int(last['Days_Since_Last_Delivery']),
        'Days_To_Nearest_Expiry':last['Days_To_Nearest_Expiry'],
        'Next_Day_Demand':0.0,'Source_Data_Type':'SYNTHETIC',
    }
    frame=pd.DataFrame([row])
    prepared=prepare_feature_rows(frame, inventory)
    # Future crisis is an explicit user-declared scenario; it is never inferred as known by default.
    prepared['Target_Crisis_Flag']=int(declared_crisis)
    prepared['Stock_Cover_Days']=prepared['Closing_Stock']/prepared['Avg_7_Day_Demand'].replace(0,np.nan)
    prepared['Stock_to_Reorder_Ratio']=prepared['Closing_Stock']/prepared['Reorder_Level'].replace(0,np.nan)
    return prepared, {'phc_id':phc_id,'medicine_id':medicine_id,'medicine_name':str(last['Medicine_Name']),
                     'unit_of_measure':str(last['Unit_Of_Measure']),'prediction_date':str(origin.date()),
                     'target_date':str(target.date()),'current_usable_stock_at_last_record':int(last['Closing_Stock']),
                     'declared_crisis_scenario':bool(declared_crisis)}


def main() -> int:
    parser=argparse.ArgumentParser(description='Synthetic-only next-day stock-out classification demo')
    parser.add_argument('--phc-id', default='PHC-DEMO-001')
    parser.add_argument('--medicine-id', default='MED001')
    parser.add_argument('--crisis', action='store_true', help='Assume a crisis is declared for the target day')
    args=parser.parse_args()
    model_path=MODELS/'stockout_next_day_classifier.joblib'
    metadata_path=MODELS/'stockout_classifier_metadata.json'
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError('Stock-out classifier missing; run train_stockout_classifier.py first.')
    inventory=read_inventory(DATA/'phc_daily_inventory.csv')
    frame,meta=build_next_day_row(inventory,args.phc_id,args.medicine_id,args.crisis)
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    model=joblib.load(model_path)
    features=metadata['features_numeric']+metadata['features_categorical']
    score=float(model.predict_proba(frame[features])[:,1][0])
    threshold=float(metadata['threshold'])
    result={**meta,'uncalibrated_stockout_score':round(score,6),'validation_selected_threshold':threshold,
            'flag_at_threshold':bool(score>=threshold),
            'data_note':'All history and labels are synthetic. This score is not calibrated or validated for real PHCs.'}
    REPORTS.mkdir(exist_ok=True)
    name=f"demo_{args.phc_id}_{args.medicine_id}_next_day_stockout_score.json"
    (REPORTS/name).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('NEXT-DAY STOCK-OUT CLASSIFICATION DEMO')
    print(json.dumps(result,indent=2))
    print('WARNING: synthetic data only; score is uncalibrated and not validated for live inventory decisions.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())

"""Smoke tests: load model, transform known rows, predict and run inventory forecast."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from src.phc_ml import read_inventory, prepare_feature_rows, forecast_demand_horizon, project_inventory
from predict_stockout import build_next_day_row

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; MODEL=ROOT/'models'/'medicine_demand_model.joblib'

def main():
    assert MODEL.exists(), f'Model missing: {MODEL}'
    model=joblib.load(MODEL)
    inventory=read_inventory(DATA/'phc_daily_inventory.csv')
    test=pd.read_csv(DATA/'ml_validation.csv',parse_dates=['Prediction_Date','Target_Date']).head(12)
    prepared=prepare_feature_rows(test,inventory)
    pred=np.asarray(model.predict(prepared[__import__('src.phc_ml',fromlist=['FEATURES']).FEATURES]),dtype=float)
    assert len(pred)==len(test) and np.isfinite(pred).all() and (pred>=0).all()
    orders=pd.read_csv(DATA/'purchase_orders.csv'); batches=pd.read_csv(DATA/'medicine_batches.csv')
    f,meta=forecast_demand_horizon(model,inventory,'PHC-DEMO-001','MED001',horizon_days=7)
    p,s,e=project_inventory(f,inventory,orders,batches,'PHC-DEMO-001','MED001')
    assert len(f)==7 and len(p)==7
    assert np.isfinite(f['Predicted_Demand_Units']).all()
    stockout_model_path=ROOT/'models'/'stockout_next_day_classifier.joblib'
    stockout_meta_path=ROOT/'models'/'stockout_classifier_metadata.json'
    assert stockout_model_path.exists() and stockout_meta_path.exists(), 'Stock-out classifier or metadata missing'
    stockout_model=joblib.load(stockout_model_path)
    stockout_meta=json.loads(stockout_meta_path.read_text(encoding='utf-8'))
    stock_frame, stock_context=build_next_day_row(inventory, 'PHC-DEMO-001', 'MED001', False)
    stock_features=stockout_meta['features_numeric']+stockout_meta['features_categorical']
    stockout_score=float(stockout_model.predict_proba(stock_frame[stock_features])[:,1][0])
    assert np.isfinite(stockout_score) and 0.0 <= stockout_score <= 1.0
    results={
      'demand_model_loads':True,'stockout_classifier_loads':True,'stockout_classifier_next_day_score_smoke_test':True,
      'stockout_demo_prediction_date':stock_context['prediction_date'],'stockout_demo_target_date':stock_context['target_date'],
      'stockout_uncalibrated_score':round(stockout_score,6),'validation_row_prediction_smoke_test':True,'smoke_test_rows':len(test),
      'nonnegative_finite_predictions':True,'recursive_7_day_forecast':True,'inventory_projection':True,
      'forecast_start_date':meta['forecast_start_date'],'forecast_rows':len(f),'inventory_rows':len(p),
      'note':'Smoke tests confirm package mechanics only; all data and evaluation results are synthetic, and stock-out scores are not calibrated for real PHCs.'
    }
    (ROOT/'reports'/'package_smoke_test.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('PACKAGE SMOKE TEST PASSED')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()

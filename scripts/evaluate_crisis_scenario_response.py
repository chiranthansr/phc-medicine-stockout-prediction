"""Counterfactual inference smoke test for the declared crisis input.

This compares model forecasts with target-day crisis off/on for the same synthetic history.
It measures model response only, not real crisis accuracy.
"""
from pathlib import Path
import sys
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phc_ml import read_inventory, forecast_demand_horizon
DATA=ROOT/'data'; REPORTS=ROOT/'reports'; MODEL=ROOT/'models'/'medicine_demand_model.joblib'

def main():
    inventory=read_inventory(DATA/'phc_daily_inventory.csv')
    model=joblib.load(MODEL)
    medicines=inventory[['Medicine_ID','Medicine_Name','Unit_Of_Measure']].drop_duplicates()
    rows=[]
    for r in medicines.itertuples(index=False):
        normal,_=forecast_demand_horizon(model,inventory,'PHC-DEMO-002',r.Medicine_ID,14,target_crisis_flags=False)
        crisis,_=forecast_demand_horizon(model,inventory,'PHC-DEMO-002',r.Medicine_ID,14,target_crisis_flags=True)
        a=float(normal['Predicted_Demand_Units'].mean()); b=float(crisis['Predicted_Demand_Units'].mean())
        rows.append({'PHC_ID':'PHC-DEMO-002','Medicine_ID':r.Medicine_ID,'Medicine_Name':r.Medicine_Name,'Horizon_Days':14,
          'Mean_Forecast_Crisis_Off':round(a,3),'Mean_Forecast_Crisis_On':round(b,3),
          'Mean_Change_Units':round(b-a,3),'Relative_Change_Percent':round(100*(b-a)/max(a,1e-9),2),
          'Interpretation':'Model scenario sensitivity only; not real-world crisis response accuracy.'})
    out=pd.DataFrame(rows).sort_values('Relative_Change_Percent',ascending=False)
    REPORTS.mkdir(exist_ok=True)
    out.to_csv(REPORTS/'crisis_scenario_forecast_sensitivity.csv',index=False)
    lines=['MODEL CRISIS-SCENARIO FORECAST SENSITIVITY','===========================================',
      'This is a model-response diagnostic for toggling the declared Target_Crisis_Flag on the same synthetic history.',
      'It does not validate whether crisis multipliers are realistic or whether real PHC demand will increase.', '',out.to_string(index=False), '']
    (REPORTS/'crisis_scenario_forecast_sensitivity.txt').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__': main()

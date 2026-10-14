"""CLI for next-day recursive demand forecasting and inventory planning.

Example (from the package root):
python forecast_inventory.py --phc-id PHC-DEMO-001 --medicine-id MED001 --horizon 14
python forecast_inventory.py --phc-id PHC-DEMO-002 --medicine-id MED001 --horizon 14 --crisis
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import joblib

from src.phc_ml import read_inventory, forecast_demand_horizon, project_inventory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL = ROOT / "models" / "medicine_demand_model.joblib"
REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic PHC medicine demand forecasting and stock-planning demo")
    parser.add_argument("--phc-id", default="PHC-DEMO-001", help="For example PHC-DEMO-001 or PHC-DEMO-002")
    parser.add_argument("--medicine-id", default="MED001", help="For example MED001 (Paracetamol) or MED006 (Metformin)")
    parser.add_argument("--horizon", type=int, default=14, help="Requested forecast horizon in days (1-365)")
    parser.add_argument("--crisis", action="store_true", help="Assume a declared crisis condition for every forecast target day")
    parser.add_argument("--current-stock", type=int, default=None, help="Optional current usable stock override; batch-level expiry detail becomes unavailable")
    parser.add_argument("--output-prefix", default="demo", help="Prefix used for report filenames")
    args = parser.parse_args()
    if not MODEL.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL}. Run train_and_evaluate.py first.")
    if args.horizon < 1 or args.horizon > 365:
        parser.error("--horizon must be between 1 and 365")

    inventory = read_inventory(DATA / "phc_daily_inventory.csv")
    orders = pd.read_csv(DATA / "purchase_orders.csv")
    batches = pd.read_csv(DATA / "medicine_batches.csv")
    model = joblib.load(MODEL)
    # Forecast far enough to make the reorder calculation, even when the requested horizon is short.
    planning_horizon = max(args.horizon, 25, 10)
    planning_forecast, meta = forecast_demand_horizon(
        model=model,
        inventory=inventory,
        phc_id=args.phc_id,
        medicine_id=args.medicine_id,
        horizon_days=planning_horizon,
        target_crisis_flags=args.crisis,
        current_stock=args.current_stock,
    )
    projection_full, summary, expiry_risk = project_inventory(
        forecast=planning_forecast,
        inventory=inventory,
        purchase_orders=orders,
        batches=batches,
        phc_id=args.phc_id,
        medicine_id=args.medicine_id,
        current_stock=args.current_stock,
        target_cover_days=25,
    )
    forecast_requested = planning_forecast.head(args.horizon).copy()
    projection_requested = projection_full.head(args.horizon).copy()
    requested_stockouts = projection_requested[projection_requested["Projected_Stockout_Flag"] == 1]
    summary["User_Requested_Horizon_Days"] = int(args.horizon)
    summary["Planning_Horizon_Used_For_Reorder"] = int(planning_horizon)
    summary["Requested_Horizon_Forecast_Demand_Total"] = round(float(projection_requested["Predicted_Demand_Units"].sum()), 2)
    summary["Requested_Horizon_Projected_Stockout_Days"] = int(projection_requested["Projected_Stockout_Flag"].sum())
    summary["Requested_Horizon_First_Stockout_Date"] = str(requested_stockouts.iloc[0]["Target_Date"]) if not requested_stockouts.empty else ""
    summary["Crisis_Scenario_Assumed"] = bool(args.crisis)
    summary["Current_Stock_Override_Used"] = args.current_stock is not None

    REPORTS.mkdir(exist_ok=True)
    prefix = f"{args.output_prefix}_{args.phc_id}_{args.medicine_id}_{args.horizon}d"
    forecast_path = REPORTS / f"{prefix}_demand_forecast.csv"
    inventory_path = REPORTS / f"{prefix}_inventory_projection.csv"
    summary_path = REPORTS / f"{prefix}_planning_summary.json"
    expiry_path = REPORTS / f"{prefix}_expiry_risk.csv"
    forecast_requested.to_csv(forecast_path, index=False)
    projection_requested.to_csv(inventory_path, index=False)
    expiry_risk.to_csv(expiry_path, index=False)
    summary["Data_Note"] = "Demo uses synthetic historical data ending 2025-03-31; dates in this forecast are illustrative, not a current real PHC forecast."
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("SYNTHETIC PHC FORECAST / INVENTORY DEMO COMPLETE")
    print(f"PHC / medicine: {args.phc_id} / {args.medicine_id} ({summary['Medicine_Name']})")
    print(f"Historical data ends: {meta['origin_date']}; forecast starts: {meta['forecast_start_date']}")
    print(f"Requested horizon: {args.horizon} days; planning horizon used: {planning_horizon} days")
    print(f"Total predicted demand in requested horizon: {summary['Requested_Horizon_Forecast_Demand_Total']} {summary['Unit_Of_Measure']}")
    print(f"Projected stock-out days in requested horizon: {summary['Requested_Horizon_Projected_Stockout_Days']}")
    print(f"Reorder recommended now: {summary['Reorder_Recommended_Now']}")
    print(f"Recommended order quantity: {summary['Recommended_Order_Quantity']} {summary['Unit_Of_Measure']}")
    print(f"Projected expired units from known batches over planning horizon: {summary['Projected_Expired_Units_From_Known_Batches']}")
    print(f"Saved: {forecast_path.name}, {inventory_path.name}, {summary_path.name}, {expiry_path.name}")
    print("WARNING: synthetic-data demonstration only; stock-out flags are scenario calculations, not calibrated probabilities.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

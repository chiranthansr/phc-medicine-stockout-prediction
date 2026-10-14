"""FastAPI bridge between a web UI and the PHC ML models.

Run from the package root:
    python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000

All bundled facility-level records are synthetic. Do not use this demo for live
clinical or procurement decisions without real-data validation and safeguards.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DEMAND_MODEL_PATH = MODELS_DIR / "medicine_demand_model.joblib"
STOCKOUT_MODEL_PATH = MODELS_DIR / "stockout_next_day_classifier.joblib"
STOCKOUT_METADATA_PATH = MODELS_DIR / "stockout_classifier_metadata.json"

from src.phc_ml import read_inventory, forecast_demand_horizon, project_inventory
from predict_stockout import build_next_day_row


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "PHC_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="PHC Medicine ML API",
    version="1.0.0",
    description=(
        "Backend API for synthetic PHC medicine demand forecasting, inventory planning, "
        "and next-day stock-out scoring. All bundled facility-level data is synthetic."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    phc_id: str = Field(min_length=1, examples=["PHC-DEMO-001"])
    medicine_id: str = Field(min_length=1, examples=["MED001"])
    horizon_days: int = Field(default=14, ge=1, le=365)
    crisis: bool = Field(default=False, description="Assume a declared crisis for each forecast target day.")
    current_stock: int | None = Field(default=None, ge=0, description="Optional override for currently usable stock.")


class StockoutRequest(BaseModel):
    phc_id: str = Field(min_length=1, examples=["PHC-DEMO-001"])
    medicine_id: str = Field(min_length=1, examples=["MED001"])
    crisis: bool = Field(default=False, description="Assume a declared crisis for the next target day.")


# Load once when the API process starts, not once per request.
try:
    if not DEMAND_MODEL_PATH.exists():
        raise FileNotFoundError(f"Demand model missing: {DEMAND_MODEL_PATH}")
    if not STOCKOUT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Stock-out model missing: {STOCKOUT_MODEL_PATH}")
    if not STOCKOUT_METADATA_PATH.exists():
        raise FileNotFoundError(f"Stock-out metadata missing: {STOCKOUT_METADATA_PATH}")

    DEMAND_MODEL = joblib.load(DEMAND_MODEL_PATH)
    STOCKOUT_MODEL = joblib.load(STOCKOUT_MODEL_PATH)
    STOCKOUT_METADATA = json.loads(STOCKOUT_METADATA_PATH.read_text(encoding="utf-8"))
    INVENTORY = read_inventory(DATA_DIR / "phc_daily_inventory.csv")
    PURCHASE_ORDERS = pd.read_csv(DATA_DIR / "purchase_orders.csv")
    MEDICINE_BATCHES = pd.read_csv(DATA_DIR / "medicine_batches.csv")
except Exception as exc:  # surface clear startup error in the server log
    raise RuntimeError(f"Could not initialise PHC ML API resources: {exc}") from exc


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _clean_value(v) for k, v in row.items()} for row in frame.to_dict(orient="records")]


def _validate_pair(phc_id: str, medicine_id: str) -> None:
    exists = ((INVENTORY["PHC_ID"] == phc_id) & (INVENTORY["Medicine_ID"] == medicine_id)).any()
    if not bool(exists):
        raise HTTPException(status_code=404, detail=f"No bundled history for PHC {phc_id!r} and medicine {medicine_id!r}.")


@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    """Basic health check for the frontend or hosting platform."""
    return {
        "status": "ok",
        "demand_model_loaded": True,
        "stockout_model_loaded": True,
        "source_data_type": "SYNTHETIC",
        "history_end_date": str(pd.to_datetime(INVENTORY["Date"]).max().date()),
        "warning": "Synthetic demo only; not validated for live PHC decisions.",
    }


@app.get("/catalog", tags=["catalog"])
def catalog() -> dict[str, Any]:
    """Return PHCs, medicine combinations, and latest bundled-history date."""
    combos = INVENTORY[["PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure"]].drop_duplicates()
    return {
        "phcs": sorted(INVENTORY["PHC_ID"].dropna().astype(str).unique().tolist()),
        "medicines": _records(combos.sort_values(["PHC_ID", "Medicine_ID"])),
        "history_start_date": str(pd.to_datetime(INVENTORY["Date"]).min().date()),
        "history_end_date": str(pd.to_datetime(INVENTORY["Date"]).max().date()),
        "source_data_type": "SYNTHETIC",
    }


@app.get("/medicines", tags=["catalog"])
def medicines(phc_id: str | None = Query(default=None)) -> dict[str, Any]:
    """Return available medicines, optionally filtered to a PHC."""
    frame = INVENTORY[["PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure"]].drop_duplicates()
    if phc_id:
        frame = frame[frame["PHC_ID"] == phc_id]
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"Unknown PHC: {phc_id}")
    return {"medicines": _records(frame.sort_values(["PHC_ID", "Medicine_ID"]))}


@app.post("/forecast", tags=["forecasting"])
def forecast(request: ForecastRequest) -> dict[str, Any]:
    """Forecast demand and return inventory, reorder, stock-out and expiry scenarios."""
    _validate_pair(request.phc_id, request.medicine_id)
    # Plan at least 25 days to support the package's order-up-to-cover policy.
    planning_horizon = max(request.horizon_days, 25)
    try:
        demand_frame, meta = forecast_demand_horizon(
            model=DEMAND_MODEL,
            inventory=INVENTORY,
            phc_id=request.phc_id,
            medicine_id=request.medicine_id,
            horizon_days=planning_horizon,
            target_crisis_flags=request.crisis,
            current_stock=request.current_stock,
        )
        inventory_frame, plan_summary, expiry_frame = project_inventory(
            forecast=demand_frame,
            inventory=INVENTORY,
            purchase_orders=PURCHASE_ORDERS,
            batches=MEDICINE_BATCHES,
            phc_id=request.phc_id,
            medicine_id=request.medicine_id,
            current_stock=request.current_stock,
            target_cover_days=25,
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    requested_forecast = demand_frame.head(request.horizon_days)
    requested_projection = inventory_frame.head(request.horizon_days)
    first_stockout = requested_projection.loc[requested_projection["Projected_Stockout_Flag"] == 1, "Target_Date"]
    risk_dates = set(requested_projection["Target_Date"].astype(str))
    requested_expiry = expiry_frame[expiry_frame["Target_Date"].astype(str).isin(risk_dates)] if not expiry_frame.empty else expiry_frame

    summary = dict(plan_summary)
    summary.update({
        "User_Requested_Horizon_Days": int(request.horizon_days),
        "Planning_Horizon_Used_For_Reorder": int(planning_horizon),
        "Requested_Horizon_Forecast_Demand_Total": round(float(requested_projection["Predicted_Demand_Units"].sum()), 2),
        "Requested_Horizon_Projected_Stockout_Days": int(requested_projection["Projected_Stockout_Flag"].sum()),
        "Requested_Horizon_First_Stockout_Date": str(first_stockout.iloc[0]) if not first_stockout.empty else None,
        "Crisis_Scenario_Assumed": bool(request.crisis),
        "Current_Stock_Override_Used": request.current_stock is not None,
        "Forecast_Origin_Date": meta["origin_date"],
        "Forecast_Start_Date": meta["forecast_start_date"],
    })

    return {
        "summary": {k: _clean_value(v) for k, v in summary.items()},
        "forecast": _records(requested_forecast),
        "inventory_projection": _records(requested_projection),
        "expiry_risk": _records(requested_expiry),
        "data_note": "Historical facility-level data is synthetic; forecasts and inventory decisions are illustrative scenarios, not live PHC recommendations.",
    }


@app.post("/stockout-risk", tags=["classification"])
def stockout_risk(request: StockoutRequest) -> dict[str, Any]:
    """Return the next-day classifier score and thresholded class for a PHC/medicine."""
    _validate_pair(request.phc_id, request.medicine_id)
    try:
        frame, context = build_next_day_row(INVENTORY, request.phc_id, request.medicine_id, request.crisis)
        features = STOCKOUT_METADATA["features_numeric"] + STOCKOUT_METADATA["features_categorical"]
        score = float(STOCKOUT_MODEL.predict_proba(frame[features])[:, 1][0])
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    threshold = float(STOCKOUT_METADATA["threshold"])
    return {
        **context,
        "uncalibrated_stockout_score": round(score, 6),
        "validation_selected_threshold": threshold,
        "flag_at_threshold": bool(score >= threshold),
        "data_note": "Synthetic-only classifier; score is not probability-calibrated and is not validated for real PHCs.",
    }


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "message": "PHC Medicine ML API is running.",
        "docs": "/docs",
        "health": "/health",
        "warning": "Synthetic demo only.",
    }

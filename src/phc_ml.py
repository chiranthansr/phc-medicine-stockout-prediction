"""Feature preparation and forecasting helpers for the PHC medicine-demand prototype.

All facility-level data included with this prototype are synthetic. The multi-day
forecast is recursive and is a planning scenario, not a calibrated prediction interval.
"""
from __future__ import annotations

from datetime import date, timedelta
from math import ceil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "Patient_Visits",
    "Emergency_Flag",
    "Crisis_Flag",
    "Target_Crisis_Flag",
    "Expected_Target_Patient_Visits",
    "Demand_Deviation_Percent",
    "Persistent_Demand_Shift_Flag",
    "Lag_1_Demand",
    "Lag_7_Demand",
    "Avg_7_Day_Demand",
    "Avg_30_Day_Demand",
    "Avg_90_Day_Demand",
    "Consumption_Trend_Ratio_7d_to_30d",
    "Patient_Visits_Avg_7d",
    "Patient_Visits_Lag_1",
    "Patient_Visits_Change_1d",
    "Demand_Rolling_7d_Std",
    "Lag_14_Demand",
]
CATEGORICAL_FEATURES = ["PHC_ID", "Medicine_ID", "Target_Day_Of_Week", "Target_Month"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Next_Day_Demand"


def read_inventory(path: str | Path) -> pd.DataFrame:
    inv = pd.read_csv(path, parse_dates=["Date"])
    required = {"Date", "PHC_ID", "Medicine_ID", "Total_Demand", "Patient_Visits", "Crisis_Flag", "Emergency_Flag", "Closing_Stock"}
    missing = sorted(required - set(inv.columns))
    if missing:
        raise ValueError(f"Inventory CSV missing required columns: {missing}")
    if not inv["Source_Data_Type"].eq("SYNTHETIC").all():
        raise ValueError("This packaged training pipeline expects the clearly labelled synthetic prototype dataset.")
    return inv.sort_values(["PHC_ID", "Medicine_ID", "Date"]).reset_index(drop=True)


def _context_maps(inventory: pd.DataFrame):
    """Build past-only date context maps from the inventory ledger."""
    inv = inventory.copy()
    inv["Date"] = pd.to_datetime(inv["Date"])

    visits = inv[["Date", "PHC_ID", "Patient_Visits"]].drop_duplicates(["Date", "PHC_ID"]).copy()
    visits = visits.sort_values(["PHC_ID", "Date"]).reset_index(drop=True)
    visits["Weekday"] = visits["Date"].dt.day_name()
    visits["Patient_Visits_Avg_7d"] = visits.groupby("PHC_ID")["Patient_Visits"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    visits["Patient_Visits_Lag_1"] = visits.groupby("PHC_ID")["Patient_Visits"].shift(1)
    visits["Patient_Visits_Change_1d"] = visits["Patient_Visits"] - visits["Patient_Visits_Lag_1"]

    visit_history = {
        phc: g.sort_values("Date").reset_index(drop=True)
        for phc, g in visits.groupby("PHC_ID")
    }
    visit_features = {
        (r.PHC_ID, r.Date): {
            "Patient_Visits_Avg_7d": float(r.Patient_Visits_Avg_7d),
            "Patient_Visits_Lag_1": float(r.Patient_Visits_Lag_1) if pd.notna(r.Patient_Visits_Lag_1) else np.nan,
            "Patient_Visits_Change_1d": float(r.Patient_Visits_Change_1d) if pd.notna(r.Patient_Visits_Change_1d) else np.nan,
            "Patient_Visits": int(r.Patient_Visits),
        }
        for r in visits.itertuples(index=False)
    }

    daily_flags = inv[["Date", "PHC_ID", "Crisis_Flag", "Emergency_Flag"]].drop_duplicates(["Date", "PHC_ID"])
    flags = {
        (r.PHC_ID, r.Date): {"Crisis_Flag": int(r.Crisis_Flag), "Emergency_Flag": int(r.Emergency_Flag)}
        for r in daily_flags.itertuples(index=False)
    }

    demand_features = {}
    demand_history = {}
    for (phc, med), g in inv.groupby(["PHC_ID", "Medicine_ID"], sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        demand = g["Total_Demand"].astype(float)
        rolling_std = demand.rolling(7, min_periods=2).std().fillna(0.0)
        lag14 = demand.shift(14)
        for i, r in g.iterrows():
            demand_features[(phc, med, r["Date"])] = {
                "Demand_Rolling_7d_Std": float(rolling_std.iloc[i]),
                "Lag_14_Demand": float(lag14.iloc[i]) if pd.notna(lag14.iloc[i]) else np.nan,
            }
        demand_history[(phc, med)] = g[["Date", "Total_Demand", "Demand_Deviation_Percent", "Persistent_Demand_Shift_Flag"]].copy()
    return visit_history, visit_features, flags, demand_features, demand_history


def prepare_feature_rows(frame: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """Add derived features without using actual target-day demand or future observations.

    Exception by design: Target_Crisis_Flag is a known/declared scenario input for the
    target date. In real use the user must supply this condition; it is not an inferred fact.
    Expected_Target_Patient_Visits is calculated from same-weekday visits at or before
    Prediction_Date, never from the actual target-day visits.
    """
    df = frame.copy()
    df["Prediction_Date"] = pd.to_datetime(df["Prediction_Date"])
    df["Target_Date"] = pd.to_datetime(df["Target_Date"])
    if not ((df["Target_Date"] - df["Prediction_Date"]).dt.days == 1).all():
        raise ValueError("Every target date must be exactly one day after its prediction date.")

    visit_history, visit_features, flags, demand_features, _ = _context_maps(inventory)
    expected_visits = []
    target_crisis = []
    target_day = []
    target_month = []
    for r in df.itertuples(index=False):
        phc = r.PHC_ID
        pred_date = pd.Timestamp(r.Prediction_Date)
        target_date = pd.Timestamp(r.Target_Date)
        hist = visit_history.get(phc)
        if hist is None:
            expected = np.nan
        else:
            hist = hist[hist["Date"] <= pred_date]
            same_weekday = hist[hist["Weekday"] == target_date.day_name()]
            if not same_weekday.empty:
                expected = float(same_weekday["Patient_Visits"].mean())
            elif not hist.empty:
                expected = float(hist["Patient_Visits"].tail(30).mean())
            else:
                expected = np.nan
        expected_visits.append(expected)
        target_crisis.append(flags.get((phc, target_date), {}).get("Crisis_Flag", 0))
        target_day.append(target_date.day_name())
        target_month.append(str(target_date.month))

    df["Expected_Target_Patient_Visits"] = expected_visits
    df["Target_Crisis_Flag"] = target_crisis
    df["Target_Day_Of_Week"] = target_day
    df["Target_Month"] = target_month

    added = []
    for r in df.itertuples(index=False):
        key = (r.PHC_ID, r.Prediction_Date)
        visit = visit_features.get(key, {})
        dfeat = demand_features.get((r.PHC_ID, r.Medicine_ID, r.Prediction_Date), {})
        added.append({**visit, **dfeat})
    added_df = pd.DataFrame(added, index=df.index)
    # Input Patient_Visits is already part of the source row. Keep the engineered value only for lag/rolling statistics.
    for c in ["Patient_Visits_Avg_7d", "Patient_Visits_Lag_1", "Patient_Visits_Change_1d", "Demand_Rolling_7d_Std", "Lag_14_Demand"]:
        df[c] = added_df[c]
    missing = sorted(set(FEATURES + [TARGET]) - set(df.columns))
    if missing:
        raise ValueError(f"Feature preparation failed; columns missing: {missing}")
    # Do not silently convert targets or features based on test-set statistics.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
    return df


def _persistent_flag(demands: Sequence[float], base_demand: float) -> tuple[float, int]:
    """Calculate today's deviation % and the five-consecutive-day shift heuristic."""
    flags = []
    deviations = []
    vals = list(map(float, demands))
    for i, value in enumerate(vals):
        prev = vals[max(0, i - 7):i]
        prev_avg = float(np.mean(prev)) if prev else max(float(base_demand), 1.0)
        deviation = (value - prev_avg) / max(prev_avg, 1.0)
        deviations.append(deviation)
        flags.append(abs(deviation) >= 0.25)
    percent = deviations[-1] * 100 if deviations else 0.0
    persistent = int(len(flags) >= 5 and all(flags[-5:]))
    return percent, persistent


def forecast_demand_horizon(
    model,
    inventory: pd.DataFrame,
    phc_id: str,
    medicine_id: str,
    horizon_days: int = 14,
    target_crisis_flags: Sequence[int] | None = None,
    current_stock: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Recursively forecast medicine demand for an arbitrary horizon.

    The origin is the final date in the supplied synthetic inventory history. Future
    patient visits are estimated from historical same-weekday visits; future emergencies
    are unknown and set to zero. Crisis flags are explicit scenario inputs (default off).
    These are scenario forecasts, not calibrated prediction intervals.
    """
    if horizon_days < 1 or horizon_days > 365:
        raise ValueError("horizon_days must be between 1 and 365.")
    inv = inventory.copy()
    inv["Date"] = pd.to_datetime(inv["Date"])
    subset = inv[(inv["PHC_ID"] == phc_id) & (inv["Medicine_ID"] == medicine_id)].sort_values("Date").copy()
    if subset.empty:
        raise ValueError(f"No inventory history for PHC_ID={phc_id!r}, Medicine_ID={medicine_id!r}.")
    origin = pd.Timestamp(subset["Date"].max())
    latest = subset[subset["Date"] == origin].iloc[-1]
    observed_demand = subset["Total_Demand"].astype(float).tolist()
    if current_stock is None:
        starting_stock = int(latest["Closing_Stock"])
    else:
        starting_stock = max(0, int(current_stock))

    phc_history = inv[(inv["PHC_ID"] == phc_id) & (inv["Date"] <= origin)][["Date", "Patient_Visits", "Emergency_Flag", "Crisis_Flag"]].drop_duplicates(["Date"]).sort_values("Date")
    observed_visits = phc_history[["Date", "Patient_Visits"]].copy()
    expected_visits_by_weekday = {}
    for weekday, grp in observed_visits.assign(Weekday=observed_visits["Date"].dt.day_name()).groupby("Weekday"):
        expected_visits_by_weekday[weekday] = float(grp["Patient_Visits"].mean())
    fallback_visits = float(observed_visits["Patient_Visits"].tail(30).mean())

    # Use observed data for the current prediction date; future current-day visits are estimated.
    visit_series = observed_visits["Patient_Visits"].astype(float).tolist()
    emergency_series = phc_history["Emergency_Flag"].astype(int).tolist()
    crisis_series = phc_history["Crisis_Flag"].astype(int).tolist()
    last_visit_date = origin
    med_baseline = float(subset["Total_Demand"].tail(30).mean())
    # `base_demand` only applies if a historical deviation window is empty.
    base_demand = max(med_baseline, 1.0)

    if target_crisis_flags is None:
        crisis_schedule = [0] * horizon_days
    elif isinstance(target_crisis_flags, (bool, int, np.integer)):
        crisis_schedule = [int(bool(target_crisis_flags))] * horizon_days
    else:
        crisis_schedule = [int(bool(x)) for x in target_crisis_flags]
        if len(crisis_schedule) != horizon_days:
            raise ValueError("target_crisis_flags must have exactly horizon_days entries.")

    forecast_rows = []
    current_emergency = int(latest["Emergency_Flag"])
    current_crisis = int(latest["Crisis_Flag"])
    for step in range(horizon_days):
        target_date = origin + pd.Timedelta(days=step + 1)
        prediction_date = target_date - pd.Timedelta(days=1)
        target_crisis = crisis_schedule[step]
        # Current day's visits/flags are observed only at the origin; farther ahead they are estimates/unknowns.
        if step == 0:
            current_visits = float(latest["Patient_Visits"])
            current_emergency = int(latest["Emergency_Flag"])
            current_crisis = int(latest["Crisis_Flag"])
        else:
            current_visits = float(visit_series[-1])
            current_emergency = 0
            current_crisis = int(crisis_schedule[step - 1])
        expected_target_visits = expected_visits_by_weekday.get(target_date.day_name(), fallback_visits)
        # The synthetic generator's demand deviation compares current demand against up to 7 prior days.
        deviation_pct, persistent_flag = _persistent_flag(observed_demand, base_demand)
        last7 = observed_demand[-7:]
        last30 = observed_demand[-30:]
        last90 = observed_demand[-90:]
        avg7 = float(np.mean(last7)) if last7 else med_baseline
        avg30 = float(np.mean(last30)) if last30 else med_baseline
        avg90 = float(np.mean(last90)) if last90 else med_baseline
        std7 = float(np.std(last7, ddof=1)) if len(last7) > 1 else 0.0
        pv7 = visit_series[-7:]
        pv_avg7 = float(np.mean(pv7)) if pv7 else fallback_visits
        pv_lag1 = float(visit_series[-2]) if len(visit_series) >= 2 else np.nan
        pv_change = current_visits - pv_lag1 if np.isfinite(pv_lag1) else np.nan
        lag1 = float(observed_demand[-1]) if observed_demand else med_baseline
        lag7 = float(observed_demand[-8]) if len(observed_demand) >= 8 else np.nan
        lag14 = float(observed_demand[-15]) if len(observed_demand) >= 15 else np.nan
        row = {
            "Patient_Visits": current_visits,
            "Emergency_Flag": current_emergency,
            "Crisis_Flag": current_crisis,
            "Target_Crisis_Flag": target_crisis,
            "Expected_Target_Patient_Visits": expected_target_visits,
            "Demand_Deviation_Percent": deviation_pct,
            "Persistent_Demand_Shift_Flag": persistent_flag,
            "Lag_1_Demand": lag1,
            "Lag_7_Demand": lag7,
            "Avg_7_Day_Demand": avg7,
            "Avg_30_Day_Demand": avg30,
            "Avg_90_Day_Demand": avg90,
            "Consumption_Trend_Ratio_7d_to_30d": avg7 / avg30 if avg30 else 1.0,
            "Patient_Visits_Avg_7d": pv_avg7,
            "Patient_Visits_Lag_1": pv_lag1,
            "Patient_Visits_Change_1d": pv_change,
            "Demand_Rolling_7d_Std": std7,
            "Lag_14_Demand": lag14,
            "PHC_ID": phc_id,
            "Medicine_ID": medicine_id,
            "Target_Day_Of_Week": target_date.day_name(),
            "Target_Month": str(target_date.month),
        }
        x = pd.DataFrame([row], columns=FEATURES)
        predicted = max(0.0, float(model.predict(x)[0]))
        predicted = min(predicted, max(0.0, avg90 * 5.0))  # conservative guard against unsupported extreme recursive drift
        forecast_rows.append({
            "Prediction_Date": prediction_date.date().isoformat(),
            "Target_Date": target_date.date().isoformat(),
            "PHC_ID": phc_id,
            "Medicine_ID": medicine_id,
            "Medicine_Name": str(latest["Medicine_Name"]),
            "Unit_Of_Measure": str(latest["Unit_Of_Measure"]),
            "Predicted_Demand_Units": round(predicted, 2),
            "Planning_Demand_Units_Ceil": int(ceil(predicted)),
            "Expected_Target_Patient_Visits": round(expected_target_visits, 2),
            "Target_Crisis_Flag": target_crisis,
            "Forecast_Method": "Recursive one-day model; future visits estimated from historical weekday means; future emergency events assumed absent",
            "Source_Data_Type": "SYNTHETIC_HISTORY_WITH_MODEL_FORECAST",
        })
        observed_demand.append(predicted)
        visit_series.append(expected_target_visits)
        emergency_series.append(0)
        crisis_series.append(target_crisis)

    meta = {
        "phc_id": phc_id,
        "medicine_id": medicine_id,
        "origin_date": origin.date().isoformat(),
        "forecast_start_date": (origin + pd.Timedelta(days=1)).date().isoformat(),
        "horizon_days": int(horizon_days),
        "starting_stock": int(starting_stock),
        "unit_of_measure": str(latest["Unit_Of_Measure"]),
        "medicine_name": str(latest["Medicine_Name"]),
        "lead_time_days": int(latest["Supplier_Lead_Time_Days"]),
        "reorder_level": int(latest["Reorder_Level"]),
        "minimum_stock_level": int(latest["Minimum_Stock_Level"]),
        "target_crisis_flags_used": crisis_schedule,
        "synthetic_data_warning": "The historical PHC-level data is synthetic. Forecast performance is not validated on real PHC records.",
    }
    return pd.DataFrame(forecast_rows), meta


def project_inventory(
    forecast: pd.DataFrame,
    inventory: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    batches: pd.DataFrame,
    phc_id: str,
    medicine_id: str,
    current_stock: int | None = None,
    target_cover_days: int = 25,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Apply FEFO stock simulation, pending-delivery and reorder rules to demand forecasts.

    Pending deliveries are synthetic scheduled receipts from the supplied purchase-order file.
    Their expiry dates are unknown, so they are excluded from expiry-risk estimates.
    """
    inv = inventory.copy()
    inv["Date"] = pd.to_datetime(inv["Date"])
    batches = batches.copy()
    batches["Receipt_Date"] = pd.to_datetime(batches["Receipt_Date"], errors="coerce")
    batches["Expiry_Date"] = pd.to_datetime(batches["Expiry_Date"], errors="coerce")
    orders = purchase_orders.copy()
    for col in ["Order_Date", "Expected_Delivery_Date", "Simulated_Delivery_Date", "Actual_Delivery_Date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    f = forecast.copy()
    f["Target_Date"] = pd.to_datetime(f["Target_Date"])
    subset = inv[(inv["PHC_ID"] == phc_id) & (inv["Medicine_ID"] == medicine_id)].sort_values("Date")
    if subset.empty:
        raise ValueError("No matching PHC/medicine in inventory dataset.")
    latest = subset.iloc[-1]
    origin = pd.Timestamp(latest["Date"])
    start_stock = int(latest["Closing_Stock"] if current_stock is None else max(0, current_stock))

    known_batches = batches[(batches["PHC_ID"] == phc_id) & (batches["Medicine_ID"] == medicine_id) & (pd.to_numeric(batches["Closing_Quantity"], errors="coerce").fillna(0) > 0)].copy()
    batch_total = int(pd.to_numeric(known_batches["Closing_Quantity"], errors="coerce").fillna(0).sum())
    lots = []
    if current_stock is None and batch_total == start_stock and start_stock > 0:
        for r in known_batches.itertuples(index=False):
            qty = int(r.Closing_Quantity)
            if qty > 0:
                expiry = pd.Timestamp(r.Expiry_Date) if pd.notna(r.Expiry_Date) else pd.NaT
                lots.append({"Batch_ID": r.Batch_ID, "Expiry_Date": expiry, "Remaining_Quantity": qty, "Lot_Source": "Existing batch; expiry known"})
    else:
        # If the caller overrides current stock, batch composition is unknown. Avoid inventing expiry dates.
        lots = [{"Batch_ID": "CURRENT_STOCK_UNALLOCATED", "Expiry_Date": pd.NaT, "Remaining_Quantity": start_stock, "Lot_Source": "Current stock override/unknown batch composition"}] if start_stock > 0 else []
    if sum(x["Remaining_Quantity"] for x in lots) != start_stock:
        lots = [{"Batch_ID": "CURRENT_STOCK_RECONCILED", "Expiry_Date": pd.NaT, "Remaining_Quantity": start_stock, "Lot_Source": "Stock total reconciled; detailed expiry unavailable"}] if start_stock > 0 else []

    order_filter = orders[(orders["PHC_ID"] == phc_id) & (orders["Medicine_ID"] == medicine_id)].copy()
    pending = order_filter[order_filter["Actual_Delivery_Date"].isna() & order_filter["Simulated_Delivery_Date"].notna() & (order_filter["Simulated_Delivery_Date"] > origin)]
    deliveries_by_date: dict[pd.Timestamp, list[dict]] = {}
    for r in pending.itertuples(index=False):
        d = pd.Timestamp(r.Simulated_Delivery_Date)
        deliveries_by_date.setdefault(d, []).append({"order_id": r.Order_ID, "qty": int(r.Order_Quantity)})

    projection = []
    batch_expiry_events = []
    stock = start_stock
    for r in f.sort_values("Target_Date").itertuples(index=False):
        d = pd.Timestamp(r.Target_Date)
        expired_today = 0
        # Conservative assumption: a batch is unavailable on and after its expiry date.
        for lot in lots:
            expiry = lot["Expiry_Date"]
            if lot["Remaining_Quantity"] > 0 and pd.notna(expiry) and expiry <= d:
                expired_today += lot["Remaining_Quantity"]
                batch_expiry_events.append({
                    "Target_Date": d.date().isoformat(), "Batch_ID": lot["Batch_ID"],
                    "Expiry_Date": expiry.date().isoformat(), "Quantity_Expired": lot["Remaining_Quantity"],
                    "Expiry_Reason": "Remaining known batch quantity was unusable on/after expiry date",
                })
                lot["Remaining_Quantity"] = 0
        for delivery in deliveries_by_date.get(d, []):
            if delivery["qty"] > 0:
                lots.append({"Batch_ID": f"PENDING_{delivery['order_id']}", "Expiry_Date": pd.NaT, "Remaining_Quantity": int(delivery["qty"]), "Lot_Source": "Pending scheduled delivery; expiry unknown"})
        stock_before_demand = int(sum(x["Remaining_Quantity"] for x in lots))
        demand_units = int(r.Planning_Demand_Units_Ceil)
        shortage = max(0, demand_units - stock_before_demand)
        dispensed = min(demand_units, stock_before_demand)
        # FEFO: earliest known expiries first; unknown expiries last.
        lots.sort(key=lambda x: (pd.Timestamp.max if pd.isna(x["Expiry_Date"]) else x["Expiry_Date"], x["Batch_ID"]))
        need = dispensed
        for lot in lots:
            if need <= 0:
                break
            used = min(lot["Remaining_Quantity"], need)
            lot["Remaining_Quantity"] -= used
            need -= used
        stock_after = int(sum(x["Remaining_Quantity"] for x in lots))
        projection.append({
            "Target_Date": d.date().isoformat(),
            "PHC_ID": phc_id,
            "Medicine_ID": medicine_id,
            "Medicine_Name": str(r.Medicine_Name),
            "Unit_Of_Measure": str(r.Unit_Of_Measure),
            "Predicted_Demand_Units": float(r.Predicted_Demand_Units),
            "Planning_Demand_Units_Ceil": demand_units,
            "Opening_Usable_Stock": stock,
            "Expired_Before_Demand": int(expired_today),
            "Pending_Delivery_Received": sum(x["qty"] for x in deliveries_by_date.get(d, [])),
            "Usable_Stock_Before_Demand": stock_before_demand,
            "Projected_Dispensed_Units": dispensed,
            "Projected_Unmet_Demand_Units": shortage,
            "Projected_Stockout_Flag": int(shortage > 0),
            "Projected_Closing_Stock": stock_after,
        })
        stock = stock_after

    proj = pd.DataFrame(projection)
    expiry_df = pd.DataFrame(batch_expiry_events, columns=["Target_Date", "Batch_ID", "Expiry_Date", "Quantity_Expired", "Expiry_Reason"])
    lead_time = int(latest["Supplier_Lead_Time_Days"])
    reorder_level = int(latest["Reorder_Level"])
    projected_stockout_dates = proj.loc[proj["Projected_Stockout_Flag"] == 1, "Target_Date"].tolist()
    within_lead = proj.head(max(1, lead_time))
    stockout_within_lead = bool((within_lead["Projected_Stockout_Flag"] == 1).any())
    stock_at_lead_end = int(within_lead["Projected_Closing_Stock"].iloc[-1]) if not within_lead.empty else stock
    order_now = bool(stockout_within_lead or stock_at_lead_end <= reorder_level or start_stock <= reorder_level)
    planning_window = proj.head(min(target_cover_days, len(proj)))
    avg_daily_demand = float(planning_window["Predicted_Demand_Units"].mean()) if not planning_window.empty else 0.0
    target_stock = int(ceil(max(0.0, avg_daily_demand) * target_cover_days))
    incoming_within_cover = sum(
        dqty for d, items in deliveries_by_date.items()
        if d <= origin + pd.Timedelta(days=target_cover_days)
        for dqty in [sum(x["qty"] for x in items)]
    )
    inventory_position = start_stock + int(incoming_within_cover)
    recommended_qty = max(0, target_stock - inventory_position) if order_now else 0
    summary = {
        "PHC_ID": phc_id,
        "Medicine_ID": medicine_id,
        "Medicine_Name": str(latest["Medicine_Name"]),
        "Unit_Of_Measure": str(latest["Unit_Of_Measure"]),
        "Data_Origin_Date": origin.date().isoformat(),
        "Forecast_Horizon_Days": int(len(f)),
        "Starting_Usable_Stock": start_stock,
        "Reorder_Level": reorder_level,
        "Supplier_Lead_Time_Days": lead_time,
        "Forecast_Demand_Total": round(float(proj["Predicted_Demand_Units"].sum()), 2),
        "Forecast_Planning_Units_Total": int(proj["Planning_Demand_Units_Ceil"].sum()),
        "Projected_Stockout_Flag_Days": int(proj["Projected_Stockout_Flag"].sum()),
        "First_Project_Stockout_Date": projected_stockout_dates[0] if projected_stockout_dates else "",
        "Projected_Expired_Units_From_Known_Batches": int(expiry_df["Quantity_Expired"].sum()) if not expiry_df.empty else 0,
        "Pending_Order_Units_Scheduled_Within_Cover_Window": int(incoming_within_cover),
        "Inventory_Position_Including_Pending_Receipts": int(inventory_position),
        "Reorder_Recommended_Now": order_now,
        "Recommended_Order_Quantity": int(recommended_qty),
        "Order_Up_To_Cover_Days": int(target_cover_days),
        "Warning": "Synthetic-data planning estimate only. Pending receipts use simulated scheduled dates. Unknown expiry of incoming deliveries is not estimated.",
    }
    return proj, summary, expiry_df

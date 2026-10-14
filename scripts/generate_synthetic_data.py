#!/usr/bin/env python3
"""Generate a reproducible synthetic PHC medicine inventory dataset.

This is a teaching/prototype simulator, not real clinical or facility data.
Uses only the Python standard library.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20261012
START_DATE = date(2025, 1, 1)
DAYS = 90
END_DATE = START_DATE + timedelta(days=DAYS - 1)
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
OUT_DIR = PACKAGE_DIR / "data"

PHCS = [
    {"PHC_ID": "PHC-DEMO-001", "State": "Karnataka", "Baseline_Patient_Visits": 72},
    {"PHC_ID": "PHC-DEMO-002", "State": "Tamil Nadu", "Baseline_Patient_Visits": 61},
]

# The names/forms below are selected from the PHC section of the user's uploaded
# National Health Mission Essential Medicine List for SHC & PHC Level (March 2020).
MEDICINES = [
    {"Medicine_ID": "MED001", "Medicine_Name": "Paracetamol Tablet 500 mg", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 30, "Crisis_Multiplier": 1.70},
    {"Medicine_ID": "MED002", "Medicine_Name": "Amoxicillin Capsule 500 mg", "Unit_Of_Measure": "capsules", "Base_Daily_Demand": 6, "Crisis_Multiplier": 1.45},
    {"Medicine_ID": "MED003", "Medicine_Name": "Oral rehydration salts (ORS)", "Unit_Of_Measure": "sachets", "Base_Daily_Demand": 8, "Crisis_Multiplier": 1.35},
    {"Medicine_ID": "MED004", "Medicine_Name": "Zinc sulphate Dispersible Tablet 20 mg", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 9, "Crisis_Multiplier": 1.30},
    {"Medicine_ID": "MED005", "Medicine_Name": "Amlodipine Tablet 5 mg", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 11, "Crisis_Multiplier": 1.10},
    {"Medicine_ID": "MED006", "Medicine_Name": "Metformin Tablet 500 mg", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 17, "Crisis_Multiplier": 1.12},
    {"Medicine_ID": "MED007", "Medicine_Name": "Albendazole Tablet 400 mg", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 3, "Crisis_Multiplier": 1.18},
    {"Medicine_ID": "MED008", "Medicine_Name": "Ferrous salt 60 mg + Folic acid 500 mcg Tablet", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 15, "Crisis_Multiplier": 1.12},
    {"Medicine_ID": "MED009", "Medicine_Name": "Omeprazole Capsule 20 mg", "Unit_Of_Measure": "capsules", "Base_Daily_Demand": 9, "Crisis_Multiplier": 1.10},
    {"Medicine_ID": "MED010", "Medicine_Name": "Salbutamol Tablet 2 mg", "Unit_Of_Measure": "tablets", "Base_Daily_Demand": 5, "Crisis_Multiplier": 1.65},
]

DAILY_FIELDS = [
    "Date", "PHC_ID", "State", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure",
    "Opening_Stock", "Expired_Quantity", "Quantity_Received", "Available_Stock",
    "Total_Demand", "Dispensed_Quantity", "Unmet_Demand", "Closing_Stock", "Stockout_Flag",
    "Patient_Visits", "Emergency_Flag", "Crisis_Flag", "Demand_Deviation_Percent",
    "Persistent_Demand_Shift_Flag", "Minimum_Stock_Level", "Reorder_Level",
    "Supplier_Lead_Time_Days", "Days_Since_Last_Delivery", "Nearest_Expiry_Date",
    "Days_To_Nearest_Expiry", "Source_Data_Type"
]
ORDER_FIELDS = [
    "Order_ID", "PHC_ID", "Medicine_ID", "Medicine_Name", "Order_Date", "Order_Quantity",
    "Supplier_Lead_Time_Days", "Delivery_Delay_Days", "Expected_Delivery_Date", "Simulated_Delivery_Date",
    "Actual_Delivery_Date", "Quantity_Received", "Delivery_Status", "Source_Data_Type"
]
BATCH_FIELDS = [
    "Batch_ID", "PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure",
    "Batch_Quantity_Received", "Receipt_Date", "Expiry_Date", "Source_Order_ID",
    "Quantity_Dispensed", "Quantity_Expired", "Closing_Quantity", "Batch_Status", "Source_Data_Type"
]
ML_FIELDS = [
    "Prediction_Date", "Target_Date", "PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure",
    "Day_Of_Week", "Month", "Patient_Visits", "Emergency_Flag", "Crisis_Flag",
    "Demand_Deviation_Percent", "Persistent_Demand_Shift_Flag", "Lag_1_Demand", "Lag_7_Demand",
    "Avg_7_Day_Demand", "Avg_30_Day_Demand", "Avg_90_Day_Demand",
    "Consumption_Trend_Ratio_7d_to_30d", "Previous_Stockouts_Last_7_Days",
    "Closing_Stock", "Minimum_Stock_Level", "Reorder_Level", "Supplier_Lead_Time_Days",
    "Days_Since_Last_Delivery", "Days_To_Nearest_Expiry", "Next_Day_Demand", "Source_Data_Type"
]
DEMAND_AUDIT_FIELDS = [
    "Date", "PHC_ID", "State", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure",
    "Configured_Base_Daily_Demand", "PHC_Medicine_Base_Demand", "Baseline_Patient_Visits",
    "Patient_Visits", "Patient_Visit_Factor", "Season_Factor", "Trend_Factor",
    "Emergency_Flag", "Emergency_Factor", "Crisis_Flag", "Configured_Crisis_Multiplier",
    "Applied_Crisis_Factor", "Random_Noise_Factor", "Random_Noise_Raw_Gaussian_Draw",
    "Random_Noise_Standard_Normal_Z", "Expected_Demand_No_Noise_Before_Crisis_Unrounded",
    "Expected_Demand_No_Noise_With_Crisis_Unrounded", "Expected_Demand_Before_Crisis_Unrounded",
    "Expected_Demand_With_Crisis_Unrounded", "Counterfactual_Demand_Without_Crisis",
    "Generated_Total_Demand", "Rounding_Adjustment_With_Crisis", "Source_Data_Type"
]


def iso(d):
    return d.isoformat() if d else ""


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]
    date_idx = {d: i for i, d in enumerate(dates)}

    medicine_by_id = {m["Medicine_ID"]: m for m in MEDICINES}
    phc_by_id = {p["PHC_ID"]: p for p in PHCS}
    pair_settings = {}
    facility_day = {}

    # Patient visits and exogenous scenarios are generated once per PHC and day.
    for phc in PHCS:
        for i, d in enumerate(dates):
            weekday = d.weekday()
            weekday_factor = 0.52 if weekday == 6 else (0.78 if weekday == 5 else 1.0)
            seasonal = 1.0 + 0.055 * math.sin(2 * math.pi * i / 30.0)
            visits = max(8, int(round(rng.gauss(phc["Baseline_Patient_Visits"] * weekday_factor * seasonal, 7))))
            crisis_flag = int(phc["PHC_ID"] == "PHC-DEMO-002" and 45 <= i <= 57)
            emergency_flag = int(rng.random() < 0.035)
            facility_day[(phc["PHC_ID"], d)] = {
                "Patient_Visits": visits,
                "Emergency_Flag": emergency_flag,
                "Crisis_Flag": crisis_flag,
                "Weekday_Factor": weekday_factor,
            }

    # Per PHC-medicine settings and synthetic daily demand sequence.
    demand_history = {}
    demand_features = {}
    crisis_effect_rows = []
    demand_audit_rows = []
    for phc in PHCS:
        for med in MEDICINES:
            lead_time = rng.randint(3, 8)
            base = med["Base_Daily_Demand"] * rng.uniform(0.82, 1.18)
            trend_per_day = rng.choice([-0.0020, -0.0010, 0.0, 0.0, 0.0010, 0.0020])
            minimum_stock = max(1, math.ceil(base * 7))
            reorder_level = max(minimum_stock, math.ceil(base * (lead_time + 3)))
            target_stock = max(reorder_level + 1, math.ceil(base * 25))
            pair_settings[(phc["PHC_ID"], med["Medicine_ID"])] = {
                "lead_time": lead_time,
                "base": base,
                "trend_per_day": trend_per_day,
                "minimum_stock": minimum_stock,
                "reorder_level": reorder_level,
                "target_stock": target_stock,
            }
            series = []
            for i, d in enumerate(dates):
                fd = facility_day[(phc["PHC_ID"], d)]
                baseline_visits = phc["Baseline_Patient_Visits"]
                visits_factor = max(0.45, fd["Patient_Visits"] / baseline_visits)
                season_factor = 1.0 + 0.10 * math.sin((2 * math.pi * i / 31.0) + (MEDICINES.index(med) * 0.13))
                trend_factor = max(0.70, 1.0 + trend_per_day * i)
                crisis_factor = med["Crisis_Multiplier"] if fd["Crisis_Flag"] else 1.0
                emergency_factor = (1.28 if med["Medicine_ID"] in {"MED001", "MED002", "MED003", "MED004", "MED010"} else 1.12) if fd["Emergency_Flag"] else 1.0
                # Keep the raw draw and its standardized Z score for exact
                # sensitivity analysis. The same single Gaussian draw is used,
                # so the simulated demand values remain unchanged.
                raw_noise_draw = rng.gauss(1.0, 0.19)
                noise_z = (raw_noise_draw - 1.0) / 0.19
                noise = max(0.45, raw_noise_draw)
                expected_no_noise_before_crisis = base * visits_factor * season_factor * trend_factor * emergency_factor
                expected_no_noise_with_crisis = expected_no_noise_before_crisis * crisis_factor
                base_without_crisis = expected_no_noise_before_crisis * noise
                expected_with_crisis = expected_no_noise_with_crisis * noise
                demand_without_crisis = max(0, int(round(base_without_crisis)))
                demand = max(0, int(round(expected_with_crisis)))
                demand_audit_rows.append({
                    "Date": iso(d),
                    "PHC_ID": phc["PHC_ID"],
                    "State": phc["State"],
                    "Medicine_ID": med["Medicine_ID"],
                    "Medicine_Name": med["Medicine_Name"],
                    "Unit_Of_Measure": med["Unit_Of_Measure"],
                    "Configured_Base_Daily_Demand": med["Base_Daily_Demand"],
                    "PHC_Medicine_Base_Demand": base,
                    "Baseline_Patient_Visits": baseline_visits,
                    "Patient_Visits": fd["Patient_Visits"],
                    "Patient_Visit_Factor": visits_factor,
                    "Season_Factor": season_factor,
                    "Trend_Factor": trend_factor,
                    "Emergency_Flag": fd["Emergency_Flag"],
                    "Emergency_Factor": emergency_factor,
                    "Crisis_Flag": fd["Crisis_Flag"],
                    "Configured_Crisis_Multiplier": med["Crisis_Multiplier"],
                    "Applied_Crisis_Factor": crisis_factor,
                    "Random_Noise_Factor": noise,
                    "Random_Noise_Raw_Gaussian_Draw": raw_noise_draw,
                    "Random_Noise_Standard_Normal_Z": noise_z,
                    "Expected_Demand_No_Noise_Before_Crisis_Unrounded": expected_no_noise_before_crisis,
                    "Expected_Demand_No_Noise_With_Crisis_Unrounded": expected_no_noise_with_crisis,
                    "Expected_Demand_Before_Crisis_Unrounded": base_without_crisis,
                    "Expected_Demand_With_Crisis_Unrounded": expected_with_crisis,
                    "Counterfactual_Demand_Without_Crisis": demand_without_crisis,
                    "Generated_Total_Demand": demand,
                    "Rounding_Adjustment_With_Crisis": demand - expected_with_crisis,
                    "Source_Data_Type": "SYNTHETIC",
                })
                if fd["Crisis_Flag"]:
                    crisis_effect_rows.append({
                        "Date": iso(d),
                        "PHC_ID": phc["PHC_ID"],
                        "Medicine_ID": med["Medicine_ID"],
                        "Medicine_Name": med["Medicine_Name"],
                        "Configured_Crisis_Multiplier": med["Crisis_Multiplier"],
                        "Patient_Visits": fd["Patient_Visits"],
                        "Emergency_Flag": fd["Emergency_Flag"],
                        "Demand_With_Crisis": demand,
                        "Counterfactual_Demand_Without_Crisis": demand_without_crisis,
                        "Incremental_Demand_From_Crisis": demand - demand_without_crisis,
                    })
                series.append(demand)
            demand_history[(phc["PHC_ID"], med["Medicine_ID"])] = series

            # Deviation/persistence indicators use only current and prior demand.
            deviations = []
            high_low_flags = []
            for i, demand in enumerate(series):
                prev = series[max(0, i - 7):i]
                prev_avg = (sum(prev) / len(prev)) if prev else max(base, 1)
                deviation = (demand - prev_avg) / max(prev_avg, 1)
                deviations.append(deviation)
                high_low_flags.append(1 if (deviation >= 0.25 or deviation <= -0.25) else 0)
            persistent = []
            for i in range(DAYS):
                window = high_low_flags[max(0, i - 4):i + 1]
                persistent.append(int(len(window) == 5 and all(window)))
            demand_features[(phc["PHC_ID"], med["Medicine_ID"])] = {
                "deviations": deviations,
                "persistent": persistent,
            }

    # Save a paired counterfactual report for the crisis window. Each row compares
    # demand with and without the crisis multiplier while holding visits, season,
    # trend, emergency status, and the sampled noise value constant.
    crisis_summary_by_medicine = []
    crisis_groups = defaultdict(list)
    for r in crisis_effect_rows:
        crisis_groups[(r["Medicine_ID"], r["Medicine_Name"])].append(r)
    for (medicine_id, medicine_name), group in sorted(crisis_groups.items()):
        with_crisis = [int(r["Demand_With_Crisis"]) for r in group]
        without_crisis = [int(r["Counterfactual_Demand_Without_Crisis"]) for r in group]
        increments = [int(r["Incremental_Demand_From_Crisis"]) for r in group]
        mean_with = sum(with_crisis) / len(with_crisis)
        mean_without = sum(without_crisis) / len(without_crisis)
        crisis_summary_by_medicine.append({
            "Medicine_ID": medicine_id,
            "Medicine_Name": medicine_name,
            "Crisis_Days": len(group),
            "Configured_Crisis_Multiplier": group[0]["Configured_Crisis_Multiplier"],
            "Mean_Demand_With_Crisis": round(mean_with, 3),
            "Mean_Counterfactual_Demand_Without_Crisis": round(mean_without, 3),
            "Mean_Incremental_Demand": round(sum(increments) / len(increments), 3),
            "Observed_Days_Demand_At_Least_Counterfactual": sum(a >= b for a, b in zip(with_crisis, without_crisis)),
            "Observed_Days_With_Increase": sum(a > b for a, b in zip(with_crisis, without_crisis)),
        })
    crisis_summary_fields = [
        "Medicine_ID", "Medicine_Name", "Crisis_Days", "Configured_Crisis_Multiplier",
        "Mean_Demand_With_Crisis", "Mean_Counterfactual_Demand_Without_Crisis",
        "Mean_Incremental_Demand", "Observed_Days_Demand_At_Least_Counterfactual",
        "Observed_Days_With_Increase",
    ]
    write_csv(OUT_DIR / "crisis_counterfactual_comparison.csv", crisis_summary_fields, crisis_summary_by_medicine)

    # Inventory simulation. Each batch is tracked separately and issued by FEFO.
    daily_rows = []
    orders = []
    batches_all = []
    order_events = defaultdict(list)
    batch_counter = defaultdict(int)
    order_counter = 0
    pair_daily_lookup = {}

    def add_batch(phc_id, med, qty, receipt_date, expiry_date, source_order_id):
        key = (phc_id, med["Medicine_ID"])
        batch_counter[key] += 1
        short_phc = phc_id[-3:]
        batch_id = f"B-{short_phc}-{med['Medicine_ID']}-{batch_counter[key]:03d}"
        row = {
            "Batch_ID": batch_id,
            "PHC_ID": phc_id,
            "Medicine_ID": med["Medicine_ID"],
            "Medicine_Name": med["Medicine_Name"],
            "Unit_Of_Measure": med["Unit_Of_Measure"],
            "Batch_Quantity_Received": int(qty),
            "Receipt_Date": receipt_date,
            "Expiry_Date": expiry_date,
            "Source_Order_ID": source_order_id,
            "Quantity_Dispensed": 0,
            "Quantity_Expired": 0,
            "remaining": int(qty),
            "Source_Data_Type": "SYNTHETIC",
        }
        batches_all.append(row)
        return row

    for phc in PHCS:
        phc_id = phc["PHC_ID"]
        for med in MEDICINES:
            med_id = med["Medicine_ID"]
            key = (phc_id, med_id)
            settings = pair_settings[key]
            base = settings["base"]
            starting_stock_days = rng.uniform(5, 9) if rng.random() < 0.20 else rng.uniform(13, 21)
            starting_stock = max(4, int(round(base * starting_stock_days)))
            # Some initial quantities deliberately have near expiry so expiry-risk logic is testable.
            if rng.random() < 0.25:
                first_fraction = rng.uniform(0.42, 0.58)
                first_expiry = START_DATE + timedelta(days=rng.randint(3, 9))
            else:
                first_fraction = rng.uniform(0.25, 0.40)
                first_expiry = START_DATE + timedelta(days=rng.randint(12, 35))
            first_qty = max(1, int(round(starting_stock * first_fraction)))
            second_qty = starting_stock - first_qty
            initial_receipt_date = START_DATE - timedelta(days=rng.randint(10, 35))
            batches = [
                add_batch(phc_id, med, first_qty, initial_receipt_date, first_expiry, "INITIAL-STOCK"),
                add_batch(phc_id, med, second_qty, initial_receipt_date, START_DATE + timedelta(days=rng.randint(180, 730)), "INITIAL-STOCK"),
            ]
            open_orders = []
            last_delivery_date = START_DATE - timedelta(days=rng.randint(10, 35))
            series = demand_history[key]
            dfeat = demand_features[key]

            for i, d in enumerate(dates):
                opening_stock = sum(b["remaining"] for b in batches)
                expired_qty = 0
                for b in batches:
                    if b["remaining"] > 0 and b["Expiry_Date"] < d:
                        expired_qty += b["remaining"]
                        b["Quantity_Expired"] += b["remaining"]
                        b["remaining"] = 0

                received_qty = 0
                # Apply simulated deliveries due today; arriving stock becomes a new batch.
                for order in order_events.get(d, []):
                    if order["PHC_ID"] == phc_id and order["Medicine_ID"] == med_id and not order["_delivered"]:
                        received_qty += int(order["Order_Quantity"])
                        shelf_life = rng.randint(365, 900) if rng.random() < 0.88 else rng.randint(90, 180)
                        batches.append(add_batch(phc_id, med, order["Order_Quantity"], d, d + timedelta(days=shelf_life), order["Order_ID"]))
                        order["_delivered"] = True
                        order["Actual_Delivery_Date"] = iso(d)
                        order["Quantity_Received"] = int(order["Order_Quantity"])
                        order["Delivery_Status"] = "Delivered"
                        last_delivery_date = d

                available_stock = sum(b["remaining"] for b in batches)
                demand = int(series[i])
                dispensed = min(demand, available_stock)
                unmet = demand - dispensed
                remaining_to_dispense = dispensed
                # FEFO: use the batch with the nearest expiry date first.
                for b in sorted((x for x in batches if x["remaining"] > 0), key=lambda x: (x["Expiry_Date"], x["Receipt_Date"], x["Batch_ID"])):
                    if remaining_to_dispense <= 0:
                        break
                    use_qty = min(b["remaining"], remaining_to_dispense)
                    b["remaining"] -= use_qty
                    b["Quantity_Dispensed"] += use_qty
                    remaining_to_dispense -= use_qty
                closing_stock = sum(b["remaining"] for b in batches)
                days_since_delivery = max(0, (d - last_delivery_date).days)
                active_batches = [b for b in batches if b["remaining"] > 0]
                nearest_expiry = min((b["Expiry_Date"] for b in active_batches), default=None)
                days_to_expiry = (nearest_expiry - d).days if nearest_expiry else ""
                deviation = dfeat["deviations"][i]
                fd = facility_day[(phc_id, d)]

                row = {
                    "Date": iso(d), "PHC_ID": phc_id, "State": phc["State"],
                    "Medicine_ID": med_id, "Medicine_Name": med["Medicine_Name"], "Unit_Of_Measure": med["Unit_Of_Measure"],
                    "Opening_Stock": opening_stock, "Expired_Quantity": expired_qty, "Quantity_Received": received_qty,
                    "Available_Stock": available_stock, "Total_Demand": demand, "Dispensed_Quantity": dispensed,
                    "Unmet_Demand": unmet, "Closing_Stock": closing_stock, "Stockout_Flag": int(unmet > 0),
                    "Patient_Visits": fd["Patient_Visits"], "Emergency_Flag": fd["Emergency_Flag"], "Crisis_Flag": fd["Crisis_Flag"],
                    "Demand_Deviation_Percent": round(deviation * 100, 2),
                    "Persistent_Demand_Shift_Flag": dfeat["persistent"][i],
                    "Minimum_Stock_Level": settings["minimum_stock"], "Reorder_Level": settings["reorder_level"],
                    "Supplier_Lead_Time_Days": settings["lead_time"], "Days_Since_Last_Delivery": days_since_delivery,
                    "Nearest_Expiry_Date": iso(nearest_expiry), "Days_To_Nearest_Expiry": days_to_expiry,
                    "Source_Data_Type": "SYNTHETIC",
                }
                daily_rows.append(row)
                pair_daily_lookup[(phc_id, med_id, d)] = row

                # Update order status relative to the simulation horizon.
                inventory_position = closing_stock + sum(o["Order_Quantity"] for o in open_orders if not o["_delivered"])
                pending = any(not o["_delivered"] for o in open_orders)
                if inventory_position <= settings["reorder_level"] and not pending:
                    order_qty = max(1, settings["target_stock"] - inventory_position)
                    order_counter += 1
                    order_id = f"ORD-{order_counter:05d}"
                    expected_delivery = d + timedelta(days=settings["lead_time"])
                    # Most orders arrive close to plan; a minority have a synthetic disruption.
                    delay_days = rng.randint(5, 12) if rng.random() < 0.18 else rng.choice([0, 0, 0, 1, 2, 3])
                    simulated_delivery = expected_delivery + timedelta(days=delay_days)
                    order = {
                        "Order_ID": order_id, "PHC_ID": phc_id, "Medicine_ID": med_id,
                        "Medicine_Name": med["Medicine_Name"], "Order_Date": iso(d),
                        "Order_Quantity": int(order_qty), "Supplier_Lead_Time_Days": settings["lead_time"],
                        "Delivery_Delay_Days": delay_days,
                        "Expected_Delivery_Date": iso(expected_delivery), "Simulated_Delivery_Date": iso(simulated_delivery),
                        "Actual_Delivery_Date": "", "Quantity_Received": 0, "Delivery_Status": "Pending beyond simulation" if simulated_delivery > END_DATE else "Awaiting delivery",
                        "Source_Data_Type": "SYNTHETIC", "_delivered": False,
                        "_simulated_delivery_date": simulated_delivery,
                    }
                    orders.append(order)
                    open_orders.append(order)
                    if simulated_delivery <= END_DATE:
                        order_events[simulated_delivery].append(order)

    # Export the demand-generation internals separately for auditing.
    # This file includes random factors used to generate targets and must not be used as ML input.
    write_csv(OUT_DIR / "demand_generation_audit.csv", DEMAND_AUDIT_FIELDS, demand_audit_rows)

    # Export inventory and orders without internal simulator fields.
    write_csv(OUT_DIR / "phc_daily_inventory.csv", DAILY_FIELDS, daily_rows)
    public_orders = []
    for o in orders:
        public_orders.append({k: o.get(k, "") for k in ORDER_FIELDS})
    write_csv(OUT_DIR / "purchase_orders.csv", ORDER_FIELDS, public_orders)

    batch_rows = []
    for b in batches_all:
        remaining = int(b["remaining"])
        if remaining > 0:
            status = "In stock"
        elif b["Quantity_Expired"] > 0:
            status = "Expired/depleted"
        else:
            status = "Depleted"
        batch_rows.append({
            "Batch_ID": b["Batch_ID"], "PHC_ID": b["PHC_ID"], "Medicine_ID": b["Medicine_ID"],
            "Medicine_Name": b["Medicine_Name"], "Unit_Of_Measure": b["Unit_Of_Measure"],
            "Batch_Quantity_Received": b["Batch_Quantity_Received"], "Receipt_Date": iso(b["Receipt_Date"]),
            "Expiry_Date": iso(b["Expiry_Date"]), "Source_Order_ID": b["Source_Order_ID"],
            "Quantity_Dispensed": int(b["Quantity_Dispensed"]), "Quantity_Expired": int(b["Quantity_Expired"]),
            "Closing_Quantity": remaining, "Batch_Status": status, "Source_Data_Type": "SYNTHETIC",
        })
    write_csv(OUT_DIR / "medicine_batches.csv", BATCH_FIELDS, batch_rows)

    # Prepare next-day forecasting rows; every feature is available on Prediction_Date.
    ml_rows = []
    for phc in PHCS:
        phc_id = phc["PHC_ID"]
        for med in MEDICINES:
            med_id = med["Medicine_ID"]
            key = (phc_id, med_id)
            series = demand_history[key]
            dfeat = demand_features[key]
            for i in range(DAYS - 1):
                d = dates[i]
                target_d = dates[i + 1]
                demand_window_7 = series[max(0, i - 6):i + 1]
                demand_window_30 = series[max(0, i - 29):i + 1]
                demand_window_90 = series[max(0, i - 89):i + 1]
                avg7 = sum(demand_window_7) / len(demand_window_7)
                avg30 = sum(demand_window_30) / len(demand_window_30)
                avg90 = sum(demand_window_90) / len(demand_window_90)
                row = pair_daily_lookup[(phc_id, med_id, d)]
                prev_stockouts = sum(
                    pair_daily_lookup[(phc_id, med_id, dates[j])]["Stockout_Flag"]
                    for j in range(max(0, i - 6), i + 1)
                )
                ml_rows.append({
                    "Prediction_Date": iso(d), "Target_Date": iso(target_d), "PHC_ID": phc_id,
                    "Medicine_ID": med_id, "Medicine_Name": med["Medicine_Name"], "Unit_Of_Measure": med["Unit_Of_Measure"],
                    "Day_Of_Week": d.strftime("%A"), "Month": d.month,
                    "Patient_Visits": row["Patient_Visits"], "Emergency_Flag": row["Emergency_Flag"], "Crisis_Flag": row["Crisis_Flag"],
                    "Demand_Deviation_Percent": row["Demand_Deviation_Percent"],
                    "Persistent_Demand_Shift_Flag": row["Persistent_Demand_Shift_Flag"],
                    "Lag_1_Demand": series[i], "Lag_7_Demand": series[i - 7] if i >= 7 else "",
                    "Avg_7_Day_Demand": round(avg7, 3), "Avg_30_Day_Demand": round(avg30, 3),
                    "Avg_90_Day_Demand": round(avg90, 3),
                    "Consumption_Trend_Ratio_7d_to_30d": round(avg7 / avg30, 4) if avg30 else 1.0,
                    "Previous_Stockouts_Last_7_Days": prev_stockouts,
                    "Closing_Stock": row["Closing_Stock"], "Minimum_Stock_Level": row["Minimum_Stock_Level"],
                    "Reorder_Level": row["Reorder_Level"], "Supplier_Lead_Time_Days": row["Supplier_Lead_Time_Days"],
                    "Days_Since_Last_Delivery": row["Days_Since_Last_Delivery"],
                    "Days_To_Nearest_Expiry": row["Days_To_Nearest_Expiry"],
                    "Next_Day_Demand": series[i + 1], "Source_Data_Type": "SYNTHETIC",
                })
    ml_rows.sort(key=lambda r: (r["Target_Date"], r["PHC_ID"], r["Medicine_ID"]))
    write_csv(OUT_DIR / "ml_training_full.csv", ML_FIELDS, ml_rows)

    # Date-ordered train/validation/test sets. Never shuffle a forecasting split.
    unique_target_dates = sorted({r["Target_Date"] for r in ml_rows})
    train_cut = int(len(unique_target_dates) * 0.70)
    val_cut = int(len(unique_target_dates) * 0.85)
    train_dates = set(unique_target_dates[:train_cut])
    val_dates = set(unique_target_dates[train_cut:val_cut])
    test_dates = set(unique_target_dates[val_cut:])
    write_csv(OUT_DIR / "ml_train.csv", ML_FIELDS, [r for r in ml_rows if r["Target_Date"] in train_dates])
    write_csv(OUT_DIR / "ml_validation.csv", ML_FIELDS, [r for r in ml_rows if r["Target_Date"] in val_dates])
    write_csv(OUT_DIR / "ml_test.csv", ML_FIELDS, [r for r in ml_rows if r["Target_Date"] in test_dates])

    # Validation: stock equation, stock/demand non-negativity, key counts and uniqueness.
    errors = []
    for r in daily_rows:
        expected_close = int(r["Opening_Stock"]) - int(r["Expired_Quantity"]) + int(r["Quantity_Received"]) - int(r["Dispensed_Quantity"])
        if expected_close != int(r["Closing_Stock"]):
            errors.append(f"Inventory equation failed: {r['PHC_ID']} {r['Medicine_ID']} {r['Date']}")
        for field in ("Opening_Stock", "Expired_Quantity", "Quantity_Received", "Available_Stock", "Total_Demand", "Dispensed_Quantity", "Unmet_Demand", "Closing_Stock"):
            if int(r[field]) < 0:
                errors.append(f"Negative value: {field} {r['PHC_ID']} {r['Medicine_ID']} {r['Date']}")
        if int(r["Dispensed_Quantity"]) + int(r["Unmet_Demand"]) != int(r["Total_Demand"]):
            errors.append(f"Demand balance failed: {r['PHC_ID']} {r['Medicine_ID']} {r['Date']}")
        if int(r["Available_Stock"]) != int(r["Opening_Stock"]) - int(r["Expired_Quantity"]) + int(r["Quantity_Received"]):
            errors.append(f"Availability balance failed: {r['PHC_ID']} {r['Medicine_ID']} {r['Date']}")

    expected_daily = len(PHCS) * len(MEDICINES) * DAYS
    if len(daily_rows) != expected_daily:
        errors.append(f"Expected {expected_daily} inventory records, found {len(daily_rows)}")
    keys = [(r["PHC_ID"], r["Medicine_ID"], r["Date"]) for r in daily_rows]
    if len(keys) != len(set(keys)):
        errors.append("Duplicate PHC-medicine-date inventory keys found")
    for b in batch_rows:
        if int(b["Batch_Quantity_Received"]) != int(b["Quantity_Dispensed"]) + int(b["Quantity_Expired"]) + int(b["Closing_Quantity"]):
            errors.append(f"Batch balance failed: {b['Batch_ID']}")
    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors[:30]))

    # Save a clear data dictionary and run summary with generated deliverables.
    readme = f"""# PHC Medicine Stock-Out Prediction — Synthetic Prototype Data\n\n\
**IMPORTANT: All facility-level dates, demand, inventory, patients, orders, delays, batches, expiry events, stock-outs and crisis events in this package are synthetic demo data. They are not actual records from real PHCs.**\n\n\
## Configuration\n- PHCs: {len(PHCS)} demo facilities\n- Medicines: {len(MEDICINES)} selected medicine/form entries\n- Simulation period: {START_DATE.isoformat()} through {END_DATE.isoformat()} ({DAYS} days)\n- Daily inventory rows: {len(daily_rows):,}\n- Next-day ML rows: {len(ml_rows):,}\n- Seed: {SEED} (repeatable generation)\n- Crisis scenario: PHC-DEMO-002, {dates[45].isoformat()} through {dates[57].isoformat()}\n- Runtime: Python 3.9+; standard library only\n\n\
## Files\n- `phc_daily_inventory.csv`: one row per PHC, medicine and date; includes stock balance, received quantities, total demand, quantity dispensed, unmet demand, stock-outs, patient visits, emergency/crisis flags and expiry indicators.\n- `purchase_orders.csv`: simulated replenishment orders, supplier lead times, expected/simulated delivery dates, quantities received and pending status.\n- `medicine_batches.csv`: batch-level quantities received, receipt/expiry dates, dispensing, expired quantity and remaining quantity. FEFO is used for issuing stock.\n- `crisis_counterfactual_comparison.csv`: paired synthetic comparison of demand with and without the crisis multiplier while holding other sampled factors constant.\n- `demand_generation_audit.csv`: logs the factors used to generate each synthetic demand value, including random noise. **Diagnostic only; do not use as model inputs because it documents how the target was generated.**\n- `ml_training_full.csv`: next-day demand forecasting dataset with features known on `Prediction_Date` and target `Next_Day_Demand`.\n- `ml_train.csv`, `ml_validation.csv`, `ml_test.csv`: chronological splits grouped by target date; records are not randomly shuffled.\n- `generate_phc_synthetic_data.py`: reproducible generator.\n\n\
## Medicine catalogue\nThe ten medicine/form entries are selected from the PHC section of the uploaded NHM *Essential Medicine List for SHC & PHC Level* (updated March 2020): Paracetamol 500 mg tablet; Amoxicillin 500 mg capsule; ORS; Zinc sulphate 20 mg dispersible tablet; Amlodipine 5 mg tablet; Metformin 500 mg tablet; Albendazole 400 mg tablet; Ferrous salt 60 mg + folic acid 500 mcg tablet; Omeprazole 20 mg capsule; Salbutamol 2 mg tablet.\n\n\
## Key field definitions\n- `Total_Demand`: simulated required quantity, including quantity that could not be dispensed. This avoids teaching the model that low dispensing during a stock-out means low demand.\n- `Dispensed_Quantity`: quantity issued from usable on-hand stock.\n- `Unmet_Demand`: `Total_Demand - Dispensed_Quantity`.\n- `Opening_Stock`: previous closing stock before today's expired stock is removed.\n- `Expired_Quantity`: quantity removed from stock because the batch's expiry date passed. A batch remains usable through its expiry date in this simulator and is removed on the following day.\n- `Available_Stock`: `Opening_Stock - Expired_Quantity + Quantity_Received`.\n- `Closing_Stock`: `Available_Stock - Dispensed_Quantity`.\n- `Stockout_Flag`: 1 when demand exceeds what can be dispensed.\n- `Persistent_Demand_Shift_Flag`: a simple simulation indicator when demand deviates at least 25% from the prior seven-day average for five consecutive days. It is a prototype heuristic, not a validated clinical rule.\n- `Next_Day_Demand`: target for forecasting; each feature row uses information available no later than `Prediction_Date`.\n- `Source_Data_Type`: all rows are marked `SYNTHETIC`.\n\n\
## Important limitations\n1. This is a small 90-day logic prototype, not enough on its own to claim reliable real-world forecasting accuracy. Validate the rules, then scale the simulation and seek real facility data if available.\n2. The PHC IDs and patient-visit figures are invented. Public state-level OPD/IPD totals were not converted directly into daily PHC patient counts because the reporting granularity/periods do not match.\n3. Demand patterns, crisis multipliers, starting stock, lead times, delivery delays and shelf lives are configurable simulation assumptions, not measured ground truth.\n4. The ML CSV contains a synthetic demand target. Real-world training requires trustworthy historical demand/dispensing data and an explicit method for estimating unmet demand during stock-outs.\n5. Keep these synthetic files separate from all public source datasets and clearly label them as synthetic in your report.\n\n\
## Validation performed\n- Inventory balance: Opening Stock - Expired Quantity + Quantity Received - Dispensed Quantity = Closing Stock.\n- Demand balance: Dispensed Quantity + Unmet Demand = Total Demand.\n- Batch balance: Batch Quantity Received = Quantity Dispensed + Quantity Expired + Closing Quantity.\n- Checked for non-negative quantities, duplicate PHC-medicine-date keys and expected record counts.\n"""
    (OUT_DIR / "DATA_GENERATION_README.md").write_text(readme, encoding="utf-8")

    print("SYNTHETIC PHC DATA GENERATION COMPLETE")
    print(f"Output directory: {OUT_DIR}")
    print(f"Simulation range: {START_DATE} to {END_DATE}")
    print(f"Daily inventory rows: {len(daily_rows):,} (expected {expected_daily:,})")
    print(f"Purchase orders: {len(public_orders):,}")
    print(f"Medicine batches: {len(batch_rows):,}")
    print(f"Full ML rows: {len(ml_rows):,}")
    print(f"Train/validation/test rows: {sum(1 for r in ml_rows if r['Target_Date'] in train_dates):,} / {sum(1 for r in ml_rows if r['Target_Date'] in val_dates):,} / {sum(1 for r in ml_rows if r['Target_Date'] in test_dates):,}")
    print(f"Stock-out rows: {sum(int(r['Stockout_Flag']) for r in daily_rows):,}")
    print(f"Expired units: {sum(int(r['Quantity_Expired']) for r in batch_rows):,}")
    print(f"Crisis counterfactual rows: {len(crisis_summary_by_medicine):,} medicines summarised")
    print(f"Demand-generation audit rows: {len(demand_audit_rows):,} (diagnostic only; not an ML feature file)")
    print(f"Validation errors: {len(errors)}")
    print(f"Data directory: {OUT_DIR}")


if __name__ == "__main__":
    main()

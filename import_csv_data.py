
import csv
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Load database settings from the project root.
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"


def read_csv(filename):
    path = DATA / filename
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def value(row, column):
    """Convert an empty CSV field to SQL NULL."""
    result = row.get(column)
    if result is None or result.strip() == "":
        return None
    return result.strip()


def integer(row, column):
    result = value(row, column)
    return int(float(result)) if result is not None else None


def number(row, column):
    result = value(row, column)
    return float(result) if result is not None else None


def boolean(row, column):
    result = value(row, column)
    if result is None:
        return None
    return result.lower() in ("1", "true", "t", "yes")


def insert_rows(cur, table, columns, rows, conflict_columns=None):
    if not rows:
        print(f"{table}: no records to insert")
        return

    column_sql = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))
    conflict_sql = ""

    if conflict_columns:
        conflict_sql = (
            " ON CONFLICT (" + ", ".join(conflict_columns)
            + ") DO NOTHING"
        )

    sql = (
        f"INSERT INTO {table} ({column_sql}) "
        f"VALUES ({placeholders}){conflict_sql}"
    )

    cur.executemany(
        sql,
        [tuple(row.get(col) for col in columns) for row in rows],
    )
    print(f"{table}: processed {len(rows)} records")


def main():
    inventory = read_csv("phc_daily_inventory.csv")
    catalogue = read_csv("medicine_catalogue.csv")
    orders = read_csv("purchase_orders.csv")
    batches = read_csv("medicine_batches.csv")

    with psycopg.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    ) as conn:
        with conn.cursor() as cur:

            # 1. PHC master table
            phc_map = {}
            for row in inventory:
                phc_id = value(row, "PHC_ID")
                state = value(row, "State")
                if phc_id and phc_id not in phc_map:
                    phc_map[phc_id] = {
                        "phc_id": phc_id,
                        "state": state,
                    }

            insert_rows(
                cur, "phcs", ["phc_id", "state"],
                list(phc_map.values()), ["phc_id"]
            )

            # 2. Medicine master table
            medicine_rows = []
            for row in catalogue:
                medicine_rows.append({
                    "medicine_id": value(row, "Medicine_ID"),
                    "medicine_name": value(row, "Medicine_Name"),
                    "unit_of_measure": value(row, "Unit_Of_Measure"),
                    "configured_base_daily_demand_assumption":
                        number(row, "Configured_Base_Daily_Demand_Assumption"),
                    "configured_crisis_multiplier_assumption":
                        number(row, "Configured_Crisis_Multiplier_Assumption"),
                    "medicine_list_source": value(row, "Medicine_List_Source"),
                    "source_data_type": value(row, "Source_Data_Type"),
                })

            insert_rows(
                cur, "medicines",
                [
                    "medicine_id", "medicine_name", "unit_of_measure",
                    "configured_base_daily_demand_assumption",
                    "configured_crisis_multiplier_assumption",
                    "medicine_list_source", "source_data_type",
                ],
                medicine_rows, ["medicine_id"]
            )

            # 3. Purchase orders
            order_rows = []
            for row in orders:
                order_rows.append({
                    "order_id": value(row, "Order_ID"),
                    "phc_id": value(row, "PHC_ID"),
                    "medicine_id": value(row, "Medicine_ID"),
                    "order_date": value(row, "Order_Date"),
                    "order_quantity": integer(row, "Order_Quantity"),
                    "supplier_lead_time_days":
                        integer(row, "Supplier_Lead_Time_Days"),
                    "delivery_delay_days":
                        integer(row, "Delivery_Delay_Days"),
                    "expected_delivery_date":
                        value(row, "Expected_Delivery_Date"),
                    "simulated_delivery_date":
                        value(row, "Simulated_Delivery_Date"),
                    "actual_delivery_date":
                        value(row, "Actual_Delivery_Date"),
                    "quantity_received": integer(row, "Quantity_Received"),
                    "delivery_status": value(row, "Delivery_Status"),
                    "source_data_type": value(row, "Source_Data_Type"),
                })

            insert_rows(
                cur, "purchase_orders",
                [
                    "order_id", "phc_id", "medicine_id", "order_date",
                    "order_quantity", "supplier_lead_time_days",
                    "delivery_delay_days", "expected_delivery_date",
                    "simulated_delivery_date", "actual_delivery_date",
                    "quantity_received", "delivery_status", "source_data_type",
                ],
                order_rows, ["order_id"]
            )

            # 4. Medicine batches
            batch_rows = []
            for row in batches:
                batch_rows.append({
                    "batch_id": value(row, "Batch_ID"),
                    "phc_id": value(row, "PHC_ID"),
                    "medicine_id": value(row, "Medicine_ID"),
                    "batch_quantity_received":
                        integer(row, "Batch_Quantity_Received"),
                    "receipt_date": value(row, "Receipt_Date"),
                    "expiry_date": value(row, "Expiry_Date"),
                    "source_order_id": value(row, "Source_Order_ID"),
                    "quantity_dispensed": integer(row, "Quantity_Dispensed"),
                    "quantity_expired": integer(row, "Quantity_Expired"),
                    "closing_quantity": integer(row, "Closing_Quantity"),
                    "batch_status": value(row, "Batch_Status"),
                    "source_data_type": value(row, "Source_Data_Type"),
                })

            insert_rows(
                cur, "medicine_batches",
                [
                    "batch_id", "phc_id", "medicine_id",
                    "batch_quantity_received", "receipt_date", "expiry_date",
                    "source_order_id", "quantity_dispensed", "quantity_expired",
                    "closing_quantity", "batch_status", "source_data_type",
                ],
                batch_rows, ["batch_id"]
            )

            # 5. Daily inventory
            inventory_rows = []
            for row in inventory:
                inventory_rows.append({
                    "inventory_date": value(row, "Date"),
                    "phc_id": value(row, "PHC_ID"),
                    "medicine_id": value(row, "Medicine_ID"),
                    "opening_stock": integer(row, "Opening_Stock"),
                    "expired_quantity": integer(row, "Expired_Quantity"),
                    "quantity_received": integer(row, "Quantity_Received"),
                    "available_stock": integer(row, "Available_Stock"),
                    "total_demand": integer(row, "Total_Demand"),
                    "dispensed_quantity": integer(row, "Dispensed_Quantity"),
                    "unmet_demand": integer(row, "Unmet_Demand"),
                    "closing_stock": integer(row, "Closing_Stock"),
                    "stockout_flag": boolean(row, "Stockout_Flag"),
                    "patient_visits": integer(row, "Patient_Visits"),
                    "emergency_flag": boolean(row, "Emergency_Flag"),
                    "crisis_flag": boolean(row, "Crisis_Flag"),
                    "demand_deviation_percent":
                        number(row, "Demand_Deviation_Percent"),
                    "persistent_demand_shift_flag":
                        boolean(row, "Persistent_Demand_Shift_Flag"),
                    "minimum_stock_level":
                        integer(row, "Minimum_Stock_Level"),
                    "reorder_level": integer(row, "Reorder_Level"),
                    "supplier_lead_time_days":
                        integer(row, "Supplier_Lead_Time_Days"),
                    "days_since_last_delivery":
                        integer(row, "Days_Since_Last_Delivery"),
                    "nearest_expiry_date":
                        value(row, "Nearest_Expiry_Date"),
                    "days_to_nearest_expiry":
                        number(row, "Days_To_Nearest_Expiry"),
                    "source_data_type": value(row, "Source_Data_Type"),
                })

            insert_rows(
                cur, "daily_inventory",
                [
                    "inventory_date", "phc_id", "medicine_id",
                    "opening_stock", "expired_quantity", "quantity_received",
                    "available_stock", "total_demand", "dispensed_quantity",
                    "unmet_demand", "closing_stock", "stockout_flag",
                    "patient_visits", "emergency_flag", "crisis_flag",
                    "demand_deviation_percent", "persistent_demand_shift_flag",
                    "minimum_stock_level", "reorder_level",
                    "supplier_lead_time_days", "days_since_last_delivery",
                    "nearest_expiry_date", "days_to_nearest_expiry",
                    "source_data_type",
                ],
                inventory_rows,
                ["inventory_date", "phc_id", "medicine_id"]
            )

    print("\nImport completed. Checking table counts next.")


if __name__ == "__main__":
    main()

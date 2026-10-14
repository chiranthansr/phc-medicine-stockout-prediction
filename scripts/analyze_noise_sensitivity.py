"""Compare hypothetical demand-noise scales using the generator's recorded Z scores.

Demand-only sensitivity analysis. It does not alter source files, regenerate inventory,
retrain ML models, or establish real-world representativeness.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
INPUT = DATA / "demand_generation_audit.csv"
SUMMARY_OUT = REPORTS / "noise_sensitivity_summary.csv"
MEDICINE_OUT = REPORTS / "noise_sensitivity_by_medicine.csv"
REPORT_OUT = REPORTS / "noise_sensitivity_report.txt"
NOISE_FLOOR = 0.45
SCENARIOS = [
    ("No random noise", 0.00),
    ("Lower noise", 0.10),
    ("Current noise", 0.19),
    ("Higher-noise stress scenario", 0.25),
]


def main() -> int:
    if not INPUT.exists():
        print(f"ERROR: Input file not found: {INPUT}")
        return 1
    REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    required = [
        "Date", "PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure",
        "Random_Noise_Factor", "Random_Noise_Standard_Normal_Z",
        "Expected_Demand_No_Noise_With_Crisis_Unrounded", "Generated_Total_Demand"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        print("Replace the generator and regenerate first; the raw Z score is needed for an exact paired comparison.")
        return 1
    for col in ["Random_Noise_Factor", "Random_Noise_Standard_Normal_Z", "Expected_Demand_No_Noise_With_Crisis_Unrounded", "Generated_Total_Demand"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Random_Noise_Factor", "Random_Noise_Standard_Normal_Z", "Expected_Demand_No_Noise_With_Crisis_Unrounded", "Generated_Total_Demand"]).copy()
    if df.empty:
        print("ERROR: No valid rows to analyse.")
        return 1

    summary_rows = []
    medicine_frames = []
    for scenario, sigma in SCENARIOS:
        noise = (1.0 + df["Random_Noise_Standard_Normal_Z"] * sigma).clip(lower=NOISE_FLOOR)
        expected = df["Expected_Demand_No_Noise_With_Crisis_Unrounded"] * noise
        rounded = expected.round().clip(lower=0)
        abs_impact = (expected - df["Expected_Demand_No_Noise_With_Crisis_Unrounded"]).abs()
        pct_impact = 100 * abs_impact / df["Expected_Demand_No_Noise_With_Crisis_Unrounded"].abs().clip(lower=1e-9)
        diff_current = (rounded - df["Generated_Total_Demand"]).abs()
        summary_rows.append({
            "Scenario": scenario,
            "Noise_Std_Dev_Assumption": sigma,
            "Rows": len(df),
            "Mean_Noise_Factor": noise.mean(),
            "Mean_Expected_Demand": expected.mean(),
            "Mean_Absolute_Impact_Units_vs_No_Noise": abs_impact.mean(),
            "Mean_Absolute_Impact_Percent_vs_No_Noise": pct_impact.mean(),
            "Rows_with_Impact_at_least_20pct": int((pct_impact >= 20).sum()),
            "Percent_Rows_with_Impact_at_least_20pct": 100 * (pct_impact >= 20).mean(),
            "Mean_Absolute_Difference_vs_Current_Generated_Demand": diff_current.mean(),
            "Exact_Reproduction_of_Current_Demand": bool((diff_current == 0).all()) if sigma == 0.19 else "not applicable",
            "Note": "Synthetic demand-only sensitivity; not calibrated to real PHC demand."
        })
        temp = df[["Medicine_ID", "Medicine_Name", "Unit_Of_Measure"]].copy()
        temp["Scenario"] = scenario
        temp["Noise_Std_Dev_Assumption"] = sigma
        temp["Mean_Expected_Demand_Row"] = expected
        temp["Absolute_Impact_Units"] = abs_impact
        temp["Absolute_Impact_Percent"] = pct_impact
        temp["Rounded_Scenario_Demand"] = rounded
        temp["Absolute_Difference_vs_Current_Demand"] = diff_current
        grouped = temp.groupby(["Scenario", "Noise_Std_Dev_Assumption", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure"], as_index=False).agg(
            Rows=("Rounded_Scenario_Demand", "size"),
            Mean_Expected_Demand=("Mean_Expected_Demand_Row", "mean"),
            Mean_Absolute_Impact_Units_vs_No_Noise=("Absolute_Impact_Units", "mean"),
            Mean_Absolute_Impact_Percent_vs_No_Noise=("Absolute_Impact_Percent", "mean"),
            Rows_with_Impact_at_least_20pct=("Absolute_Impact_Percent", lambda x: int((x >= 20).sum())),
            Mean_Absolute_Difference_vs_Current_Demand=("Absolute_Difference_vs_Current_Demand", "mean"),
        )
        grouped["Percent_Rows_with_Impact_at_least_20pct"] = 100 * grouped["Rows_with_Impact_at_least_20pct"] / grouped["Rows"]
        medicine_frames.append(grouped)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUT, index=False, float_format="%.4f")
    pd.concat(medicine_frames, ignore_index=True).to_csv(MEDICINE_OUT, index=False, float_format="%.4f")
    with REPORT_OUT.open("w", encoding="utf-8") as f:
        f.write("SYNTHETIC DEMAND NOISE SENSITIVITY ANALYSIS\n")
        f.write("===========================================\n\n")
        f.write("Paired counterfactual comparison using saved standardized Gaussian Z scores.\n")
        f.write("The same Z scores are used across scenarios; the current sigma=0.19 scenario should reproduce generated demand.\n")
        f.write("The 0.45 lower noise floor is retained in each scenario. This is demand-only and does not recalculate inventory or stock-outs.\n")
        f.write("These are illustrative assumptions, not empirical calibration to real PHC demand.\n\n")
        f.write(summary.to_string(index=False))
        f.write(f"\n\nSaved: {SUMMARY_OUT.name}, {MEDICINE_OUT.name}\n")
    print("DEMAND NOISE SENSITIVITY ANALYSIS COMPLETE")
    print(f"Rows analysed: {len(df)}")
    print(summary[["Scenario", "Noise_Std_Dev_Assumption", "Mean_Absolute_Impact_Percent_vs_No_Noise", "Percent_Rows_with_Impact_at_least_20pct", "Mean_Absolute_Difference_vs_Current_Generated_Demand", "Exact_Reproduction_of_Current_Demand"]].to_string(index=False))
    print(f"Saved: {SUMMARY_OUT.name}, {MEDICINE_OUT.name}, {REPORT_OUT.name}")
    print("NOTE: synthetic demand-only sensitivity; not evidence of real-world PHC variability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

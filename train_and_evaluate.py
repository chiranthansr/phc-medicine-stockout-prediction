"""Train, select, refit and evaluate the PHC demand model.

- Selects among candidate regressors on the validation period only.
- Refits the chosen configuration on train+validation.
- Uses the later chronological test split once for final evaluation.
- Labels all facility-level data and results as synthetic.
"""
from __future__ import annotations

import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.phc_ml import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, read_inventory, prepare_feature_rows

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODEL_PATH = MODELS / "medicine_demand_model.joblib"
SEED = 42


def load_split(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required data file: {path}")
    df = pd.read_csv(path, parse_dates=["Prediction_Date", "Target_Date"])
    if "Source_Data_Type" not in df or not df["Source_Data_Type"].eq("SYNTHETIC").all():
        raise ValueError(f"{name} contains non-synthetic or unlabelled data. Check data provenance before training.")
    return df


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def candidate_models() -> dict:
    return {
        "ExtraTrees_400_leaf5": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=5, max_features=0.8,
            random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees_400_leaf3": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=0.8,
            random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees_600_leaf8": ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=8, max_features=0.8,
            random_state=SEED, n_jobs=-1,
        ),
        "RandomForest_400_leaf3": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=0.8,
            random_state=SEED, n_jobs=-1,
        ),
        "RandomForest_500_leaf5": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, max_features=0.8,
            random_state=SEED, n_jobs=-1,
        ),
        "GradientBoosting_Huber": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.04, max_depth=2,
            min_samples_leaf=5, loss="huber", random_state=SEED,
        ),
    }


def macro_normalized_mae(y_true, pred, medicine_ids) -> float:
    frame = pd.DataFrame({"actual": np.asarray(y_true, dtype=float), "pred": np.asarray(pred, dtype=float), "medicine": np.asarray(medicine_ids)})
    scores = []
    for _, g in frame.groupby("medicine"):
        denom = max(float(g["actual"].mean()), 1.0)
        scores.append(float(mean_absolute_error(g["actual"], g["pred"]) / denom))
    return float(np.mean(scores)) if scores else float("nan")


def metric_row(name, y_true, pred, medicine_ids) -> dict:
    p = np.maximum(np.asarray(pred, dtype=float), 0.0)
    y = np.asarray(y_true, dtype=float)
    return {
        "Model": name,
        "Rows": int(len(y)),
        "MAE_Pooled_Different_Units_Caution": float(mean_absolute_error(y, p)),
        "RMSE_Pooled_Different_Units_Caution": float(mean_squared_error(y, p) ** 0.5),
        "R2_Pooled_Different_Units_Caution": float(r2_score(y, p)),
        "Macro_Mean_Per_Medicine_Normalized_MAE": macro_normalized_mae(y, p, medicine_ids),
        "Interpretation_Note": "Pooled MAE/RMSE/R2 combine tablets, capsules and sachets; prioritize per-medicine metrics and macro normalized MAE.",
    }


def per_medicine_metrics(y_true, pred, medicines) -> pd.DataFrame:
    df = pd.DataFrame({"Actual": np.asarray(y_true, dtype=float), "Predicted": np.maximum(np.asarray(pred, dtype=float), 0), "Medicine_ID": np.asarray(medicines)})
    rows = []
    for medicine, g in df.groupby("Medicine_ID"):
        err = g["Predicted"] - g["Actual"]
        abs_err = err.abs()
        actual_mean = float(g["Actual"].mean())
        rows.append({
            "Medicine_ID": medicine,
            "Rows": int(len(g)),
            "Actual_Mean": actual_mean,
            "Predicted_Mean": float(g["Predicted"].mean()),
            "MAE_Units_For_This_Medicine": float(abs_err.mean()),
            "RMSE_Units_For_This_Medicine": float(np.sqrt(np.mean(err ** 2))),
            "MAE_As_Percent_Of_Medicine_Mean": float(100 * abs_err.mean() / max(actual_mean, 1.0)),
            "Mean_Signed_Error_Predicted_Minus_Actual": float(err.mean()),
            "Overprediction_Rows": int((err > 0).sum()),
            "Underprediction_Rows": int((err < 0).sum()),
            "Note": "Units are medicine-specific; do not compare raw MAE across dosage forms as if units were interchangeable.",
        })
    return pd.DataFrame(rows).sort_values("MAE_As_Percent_Of_Medicine_Mean", ascending=False)


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    inventory = read_inventory(DATA / "phc_daily_inventory.csv")
    train_raw = load_split("ml_train.csv")
    val_raw = load_split("ml_validation.csv")
    test_raw = load_split("ml_test.csv")

    # Temporal isolation checks. All rows sharing a target date remain in the same split.
    train_max = train_raw["Target_Date"].max()
    val_min, val_max = val_raw["Target_Date"].min(), val_raw["Target_Date"].max()
    test_min = test_raw["Target_Date"].min()
    if not (train_max < val_min <= val_max < test_min):
        raise ValueError(f"Non-chronological or overlapping split dates: train max={train_max}, validation={val_min}..{val_max}, test min={test_min}")
    if set(train_raw["Target_Date"]).intersection(val_raw["Target_Date"]) or set(val_raw["Target_Date"]).intersection(test_raw["Target_Date"]):
        raise ValueError("Target dates overlap between splits; aborting to prevent temporal leakage.")

    train = prepare_feature_rows(train_raw, inventory)
    valid = prepare_feature_rows(val_raw, inventory)
    test = prepare_feature_rows(test_raw, inventory)
    expected_counts = (1240, 260, 280)
    got_counts = (len(train), len(valid), len(test))
    if got_counts != expected_counts:
        print(f"NOTE: current split rows {got_counts} differ from original expected split {expected_counts}; continuing with actual files.")
    if not set(FEATURES).issubset(train.columns):
        raise ValueError("Missing final feature columns.")

    X_train, y_train = train[FEATURES], train[TARGET].astype(float)
    X_val, y_val = valid[FEATURES], valid[TARGET].astype(float)
    val_results = []
    fitted_candidates = {}
    candidate_predictions = {}
    print("VALIDATION-ONLY MODEL SELECTION", flush=True)
    for name, estimator in candidate_models().items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = np.maximum(pipe.predict(X_val), 0.0)
        met = metric_row(name, y_val, pred, valid["Medicine_ID"])
        val_results.append(met)
        fitted_candidates[name] = pipe
        candidate_predictions[name] = pred
        print(f"{name}: macro normalized MAE={met['Macro_Mean_Per_Medicine_Normalized_MAE']:.4f}, MAE={met['MAE_Pooled_Different_Units_Caution']:.4f}", flush=True)

    # Transparent simple baselines, not candidate ML estimators.
    baseline_specs = {
        "Baseline_Previous_Day": valid["Lag_1_Demand"].fillna(train["Lag_1_Demand"].median()).to_numpy(),
        "Baseline_7_Day_Average": valid["Avg_7_Day_Demand"].to_numpy(),
        "Baseline_30_Day_Average": valid["Avg_30_Day_Demand"].to_numpy(),
    }
    for name, pred in baseline_specs.items():
        val_results.append(metric_row(name, y_val, pred, valid["Medicine_ID"]))
    val_comparison = pd.DataFrame(val_results).sort_values("Macro_Mean_Per_Medicine_Normalized_MAE")
    val_comparison.to_csv(REPORTS / "validation_model_comparison.csv", index=False, float_format="%.6f")

    ml_rows = val_comparison[~val_comparison["Model"].str.startswith("Baseline_")]
    selected_name = str(ml_rows.sort_values("Macro_Mean_Per_Medicine_Normalized_MAE").iloc[0]["Model"])
    selected_estimator = candidate_models()[selected_name]
    selected_validation_pred = candidate_predictions[selected_name]
    validation_detail = valid[["Prediction_Date", "Target_Date", "PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure", TARGET]].copy()
    validation_detail["Predicted_Demand"] = selected_validation_pred.round(3)
    validation_detail["Signed_Error_Predicted_Minus_Actual"] = (validation_detail["Predicted_Demand"] - validation_detail[TARGET]).round(3)
    validation_detail["Absolute_Error"] = validation_detail["Signed_Error_Predicted_Minus_Actual"].abs().round(3)
    validation_detail.to_csv(REPORTS / "validation_predictions_selected_model.csv", index=False)
    per_medicine_metrics(y_val, selected_validation_pred, valid["Medicine_ID"]).to_csv(REPORTS / "validation_metrics_by_medicine.csv", index=False, float_format="%.5f")

    # Final fit uses train+validation after model family/hyperparameters are selected.
    trainval = pd.concat([train, valid], ignore_index=True).sort_values(["Target_Date", "PHC_ID", "Medicine_ID"])
    final_model = Pipeline([("preprocessor", make_preprocessor()), ("model", selected_estimator)])
    final_model.fit(trainval[FEATURES], trainval[TARGET].astype(float))
    test_pred = np.maximum(final_model.predict(test[FEATURES]), 0.0)

    # One final evaluation on the reserved, later test period.
    selected_test_metrics = metric_row(selected_name, test[TARGET], test_pred, test["Medicine_ID"])
    test_baselines = {
        "Baseline_Previous_Day": test["Lag_1_Demand"].fillna(trainval["Lag_1_Demand"].median()).to_numpy(),
        "Baseline_7_Day_Average": test["Avg_7_Day_Demand"].to_numpy(),
        "Baseline_30_Day_Average": test["Avg_30_Day_Demand"].to_numpy(),
    }
    test_metric_rows = [selected_test_metrics]
    for name, pred in test_baselines.items():
        test_metric_rows.append(metric_row(name, test[TARGET], pred, test["Medicine_ID"]))
    test_metric_df = pd.DataFrame(test_metric_rows)
    test_metric_df.to_csv(REPORTS / "test_metrics_comparison.csv", index=False, float_format="%.6f")

    test_detail = test[["Prediction_Date", "Target_Date", "PHC_ID", "Medicine_ID", "Medicine_Name", "Unit_Of_Measure", "Target_Crisis_Flag" if "Target_Crisis_Flag" in test else "Target_Date", TARGET]].copy()
    test_detail["Predicted_Demand"] = test_pred.round(3)
    test_detail["Signed_Error_Predicted_Minus_Actual"] = (test_detail["Predicted_Demand"] - test_detail[TARGET]).round(3)
    test_detail["Absolute_Error"] = test_detail["Signed_Error_Predicted_Minus_Actual"].abs().round(3)
    test_detail.to_csv(REPORTS / "test_predictions.csv", index=False)
    per_medicine_metrics(test[TARGET], test_pred, test["Medicine_ID"]).to_csv(REPORTS / "test_metrics_by_medicine.csv", index=False, float_format="%.5f")

    # Feature importance is descriptive for tree ensembles, not causal explanation.
    try:
        pre = final_model.named_steps["preprocessor"]
        names = pre.get_feature_names_out()
        imps = final_model.named_steps["model"].feature_importances_
        importance = pd.DataFrame({"Feature": names, "Importance": imps}).sort_values("Importance", ascending=False)
        importance.to_csv(REPORTS / "feature_importance.csv", index=False, float_format="%.8f")
    except Exception as exc:
        (REPORTS / "feature_importance_unavailable.txt").write_text(f"Feature names could not be exported: {exc}\n", encoding="utf-8")

    joblib.dump(final_model, MODEL_PATH, compress=3)
    metadata = {
        "project": "AI-Based Medicine Stock-Out Prediction and Inventory Management System for PHCs",
        "model_type": selected_name,
        "estimator": selected_estimator.__class__.__name__,
        "model_path": "models/medicine_demand_model.joblib",
        "target": TARGET,
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "selected_using": "Lowest macro mean per-medicine normalized MAE on chronological validation split",
        "fit_rows_final": int(len(trainval)),
        "test_rows": int(len(test)),
        "train_date_max": str(train_raw["Target_Date"].max().date()),
        "validation_dates": [str(val_raw["Target_Date"].min().date()), str(val_raw["Target_Date"].max().date())],
        "test_dates": [str(test_raw["Target_Date"].min().date()), str(test_raw["Target_Date"].max().date())],
        "source_data": "SYNTHETIC ONLY; all PHC-level demand, visits, inventory, deliveries, batches and crisis values are simulated.",
        "future_known_feature": "Target_Crisis_Flag is a declared target-day scenario input; in real deployment it must be supplied only if known/declared.",
        "unvalidated_real_world_accuracy": True,
        "random_seed": SEED,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics = {
        "model_selection": "Validation-only; selected model refit on train+validation before single final test evaluation.",
        "selected_model": selected_name,
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "test_target_date_range": [str(test_raw["Target_Date"].min().date()), str(test_raw["Target_Date"].max().date())],
        "selected_model_test_metrics": selected_test_metrics,
        "validation_selected_model_metrics": val_comparison[val_comparison["Model"] == selected_name].iloc[0].to_dict(),
        "important_caution": "Pooled metrics combine tablets, capsules and sachets. Inspect per-medicine metrics. All reported scores are on synthetic records only and do not establish real PHC accuracy.",
        "why_no_precision_recall": "This demand model is regression, so MAE/RMSE/R2 are appropriate. A separate stock-out classifier and its precision/recall metrics are produced by train_stockout_classifier.py.",
    }
    (REPORTS / "model_evaluation_summary.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    report = []
    report.append("PHC MEDICINE DEMAND MODEL — FINAL EVALUATION\n")
    report.append("All demand and facility records in this package are synthetic. Scores do not establish real-world PHC forecasting accuracy.\n\n")
    report.append(f"Selected model: {selected_name}\n")
    report.append(f"Selection metric: macro mean per-medicine normalized MAE (lower is better); evaluated only on chronological validation data.\n")
    report.append(f"Final fitting rows (train + validation): {len(trainval)}\n")
    report.append(f"Final test rows: {len(test)}; target dates {test_raw['Target_Date'].min().date()} through {test_raw['Target_Date'].max().date()}\n\n")
    report.append("Validation model comparison (model selection):\n")
    report.append(val_comparison.to_string(index=False))
    report.append("\n\nFinal held-out test comparison (test used once after model selection):\n")
    report.append(test_metric_df.to_string(index=False))
    report.append("\n\nPer-medicine test metrics are in test_metrics_by_medicine.csv.\n")
    report.append("Pooled MAE/RMSE/R2 mix units (tablets/capsules/sachets) and must be interpreted cautiously. Macro normalized MAE computes MAE/actual-mean per medicine and averages these percentages, giving each medicine equal weight.\n")
    report.append("The target-day crisis flag is a scenario input. Crisis generalization cannot be validated from the final test split because its target dates contain no crisis period.\n")
    (REPORTS / "model_evaluation_summary.txt").write_text("".join(report), encoding="utf-8")

    version_text = f"Python={platform.python_version()}\npandas={pd.__version__}\nscikit-learn={sklearn.__version__}\njoblib={joblib.__version__}\nnumpy={np.__version__}\n"
    (REPORTS / "training_environment.txt").write_text(version_text, encoding="utf-8")

    print("\nFINAL ML TRAINING COMPLETE")
    print(f"Selected model: {selected_name}")
    print(f"Final fit rows: {len(trainval)}; final test rows: {len(test)}")
    print("Test comparison:")
    print(test_metric_df[["Model", "MAE_Pooled_Different_Units_Caution", "RMSE_Pooled_Different_Units_Caution", "R2_Pooled_Different_Units_Caution", "Macro_Mean_Per_Medicine_Normalized_MAE"]].to_string(index=False))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Reports: {REPORTS}")
    print("WARNING: all reported results are on synthetic data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

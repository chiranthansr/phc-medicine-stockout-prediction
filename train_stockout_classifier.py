#!/usr/bin/env python3
"""Train/evaluate a next-day stock-out classifier on the bundled synthetic data.

Labels are joined from the target date's synthetic inventory Stockout_Flag. Features
are known at prediction time plus an explicitly declared target-day crisis scenario.
All results are synthetic-only and are not evidence of real PHC performance.
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.phc_ml import prepare_feature_rows, read_inventory

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
LABEL = "Stockout_Next_Day"

NUMERIC = [
    "Patient_Visits", "Emergency_Flag", "Crisis_Flag", "Target_Crisis_Flag",
    "Expected_Target_Patient_Visits", "Demand_Deviation_Percent",
    "Persistent_Demand_Shift_Flag", "Lag_1_Demand", "Lag_7_Demand",
    "Avg_7_Day_Demand", "Avg_30_Day_Demand", "Avg_90_Day_Demand",
    "Consumption_Trend_Ratio_7d_to_30d", "Patient_Visits_Avg_7d",
    "Patient_Visits_Lag_1", "Patient_Visits_Change_1d",
    "Demand_Rolling_7d_Std", "Lag_14_Demand", "Closing_Stock",
    "Minimum_Stock_Level", "Reorder_Level", "Supplier_Lead_Time_Days",
    "Days_Since_Last_Delivery", "Days_To_Nearest_Expiry",
    "Previous_Stockouts_Last_7_Days", "Stock_Cover_Days", "Stock_to_Reorder_Ratio",
]
CATEGORICAL = ["PHC_ID", "Medicine_ID", "Target_Day_Of_Week", "Target_Month"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def label_split(filename: str, inventory: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(DATA / filename, parse_dates=["Prediction_Date", "Target_Date"])
    label_map = inventory[["Date", "PHC_ID", "Medicine_ID", "Stockout_Flag"]].rename(
        columns={"Date": "Target_Date", "Stockout_Flag": LABEL}
    )
    out = frame.merge(label_map, on=["Target_Date", "PHC_ID", "Medicine_ID"], how="left", validate="one_to_one")
    if out[LABEL].isna().any():
        raise ValueError(f"Could not derive all next-day stock-out labels for {filename}.")
    out[LABEL] = out[LABEL].astype(int)
    prepared = prepare_feature_rows(out, inventory)
    prepared["Stock_Cover_Days"] = prepared["Closing_Stock"] / prepared["Avg_7_Day_Demand"].replace(0, np.nan)
    prepared["Stock_to_Reorder_Ratio"] = prepared["Closing_Stock"] / prepared["Reorder_Level"].replace(0, np.nan)
    missing = sorted(set(FEATURES + [LABEL]) - set(prepared.columns))
    if missing:
        raise ValueError(f"Feature construction failed for {filename}: {missing}")
    if "Next_Day_Demand" in FEATURES:
        raise AssertionError("Leakage: target demand is present in classifier features.")
    return prepared.sort_values(["Target_Date", "PHC_ID", "Medicine_ID"]).reset_index(drop=True)


def candidates() -> dict[str, tuple[object, bool]]:
    return {
        "ExtraTreesClassifier_400_leaf3_balanced": (
            ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.8,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1), False),
        "RandomForestClassifier_400_leaf3_balanced": (
            RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.8,
                                   class_weight="balanced_subsample", random_state=SEED, n_jobs=-1), False),
        "LogisticRegression_balanced": (
            LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5, random_state=SEED), True),
    }


def probabilities(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipe.predict_proba(X)[:, 1], dtype=float)


def metric_record(name: str, y_true, scores, threshold: float, split: str) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
        ap = average_precision_score(y_true, scores) if y_true.sum() else float("nan")
    return {
        "Split": split, "Model": name, "Rows": int(len(y_true)),
        "Positive_Labels": int(y_true.sum()), "Positive_Rate": float(y_true.mean()),
        "Threshold": float(threshold), "Accuracy": float(accuracy_score(y_true, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Average_Precision_PR_AUC": float(ap), "ROC_AUC": float(roc),
        "True_Negatives": int(tn), "False_Positives": int(fp),
        "False_Negatives": int(fn), "True_Positives": int(tp),
    }


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    # Threshold is tuned on validation only. The coarse grid helps avoid pretending the
    # few validation positives support highly precise threshold optimisation.
    grid = np.linspace(0.05, 0.95, 19)
    vals = [(float(t), float(f1_score(y_true, scores >= t, zero_division=0))) for t in grid]
    best_f1 = max(x[1] for x in vals)
    # On ties, select the larger threshold to avoid unnecessary alarms.
    best_threshold = max(t for t, f in vals if np.isclose(f, best_f1))
    return best_threshold, best_f1


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    inventory = read_inventory(DATA / "phc_daily_inventory.csv")
    train = label_split("ml_train.csv", inventory)
    valid = label_split("ml_validation.csv", inventory)
    test = label_split("ml_test.csv", inventory)

    for name, split in [("train", train), ("validation", valid), ("test", test)]:
        print(f"{name}: {len(split)} rows, positives={int(split[LABEL].sum())} ({split[LABEL].mean():.2%})")
    # Important: labels must be uncommon in this synthetic data; document support clearly.
    if train[LABEL].nunique() != 2 or valid[LABEL].nunique() != 2 or test[LABEL].nunique() != 2:
        raise ValueError("All chronological splits must contain positive and negative labels for these metrics.")

    X_train, y_train = train[FEATURES], train[LABEL].to_numpy()
    X_val, y_val = valid[FEATURES], valid[LABEL].to_numpy()
    X_test, y_test = test[FEATURES], test[LABEL].to_numpy()

    val_rows = []
    pipes = {}
    val_scores_map = {}
    for name, (estimator, scale) in candidates().items():
        pipe = Pipeline([("preprocessor", make_preprocessor(scale)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        scores = probabilities(pipe, X_val)
        val_scores_map[name] = scores
        pipes[name] = pipe
        rec = metric_record(name, y_val, scores, 0.5, "validation_threshold_default")
        val_rows.append(rec)
        print(f"{name}: validation AP={rec['Average_Precision_PR_AUC']:.4f}, F1@0.50={rec['F1']:.4f}")

    # Select model by Average Precision on validation (threshold-free ranking metric).
    validation_scores = pd.DataFrame(val_rows).sort_values(
        ["Average_Precision_PR_AUC", "F1"], ascending=False
    )
    selected_name = str(validation_scores.iloc[0]["Model"])
    threshold, val_best_f1 = select_f1_threshold(y_val, val_scores_map[selected_name])

    # Refit only after selection; test is not touched to choose model or threshold.
    trainval = pd.concat([train, valid], ignore_index=True).sort_values(["Target_Date", "PHC_ID", "Medicine_ID"])
    estimator, scale = candidates()[selected_name]
    final_pipe = Pipeline([("preprocessor", make_preprocessor(scale)), ("model", estimator)])
    final_pipe.fit(trainval[FEATURES], trainval[LABEL].to_numpy())
    test_scores = probabilities(final_pipe, X_test)

    # Majority-class baseline has zero recall/precision for the rare positive class. This
    # is reported alongside accuracy to show why accuracy alone is not sufficient.
    baseline_scores = np.zeros(len(y_test), dtype=float)
    baseline = metric_record("Baseline_Predict_No_Stockout", y_test, baseline_scores, 0.5, "test")
    test_model = metric_record(selected_name, y_test, test_scores, threshold, "test")
    pd.DataFrame([test_model, baseline]).to_csv(REPORTS / "stockout_classifier_test_metrics.csv", index=False, float_format="%.6f")
    validation_scores.to_csv(REPORTS / "stockout_classifier_validation_model_comparison.csv", index=False, float_format="%.6f")

    detail = test[["Prediction_Date", "Target_Date", "PHC_ID", "Medicine_ID", "Medicine_Name", LABEL]].copy()
    detail["Predicted_Stockout_Probability"] = test_scores.round(6)
    detail["Predicted_Stockout_Flag"] = (test_scores >= threshold).astype(int)
    detail["Threshold_Used"] = threshold
    detail.to_csv(REPORTS / "stockout_classifier_test_predictions.csv", index=False)

    per_medicine = []
    for med_id in sorted(test["Medicine_ID"].unique()):
        idx = np.flatnonzero(test["Medicine_ID"].to_numpy() == med_id)
        if len(idx) == 0:
            continue
        y_m = y_test[idx]
        s_m = test_scores[idx]
        rec = metric_record(selected_name, y_m, s_m, threshold, "test")
        rec["Medicine_ID"] = med_id
        rec["Medicine_Name"] = str(test.iloc[idx[0]]["Medicine_Name"])
        per_medicine.append(rec)
    pd.DataFrame(per_medicine).to_csv(REPORTS / "stockout_classifier_test_by_medicine.csv", index=False, float_format="%.6f")

    joblib.dump(final_pipe, MODELS / "stockout_next_day_classifier.joblib", compress=3)
    metadata = {
        "project": "AI-Based Medicine Stock-Out Prediction and Inventory Management System for PHCs",
        "model_type": selected_name, "task": "next-day Stockout_Flag classification",
        "target": LABEL,
        "features_numeric": NUMERIC, "features_categorical": CATEGORICAL,
        "label_definition": "Stockout_Flag in synthetic inventory on Target_Date, joined by PHC_ID and Medicine_ID.",
        "class_imbalance": {"train_positive": int(y_train.sum()), "validation_positive": int(y_val.sum()), "test_positive": int(y_test.sum())},
        "selected_on": "Average precision (PR AUC) on chronological validation split only",
        "threshold": threshold, "threshold_selection": "Grid search maximizing validation F1; ties select higher threshold; test not used for threshold selection.",
        "source_data": "SYNTHETIC ONLY; not validated on real PHC records.",
        "real_world_accuracy_validated": False,
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
    }
    (MODELS / "stockout_classifier_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "NEXT-DAY STOCK-OUT CLASSIFIER — SYNTHETIC DATA ONLY",
        "===================================================",
        "Target: whether simulated Stockout_Flag is 1 on Target_Date.",
        "Features are from Prediction_Date or known scenario fields; target-day actual stock/demand columns are not input features.",
        f"Train: {len(train)} rows, {int(y_train.sum())} positives ({y_train.mean():.2%}).",
        f"Validation: {len(valid)} rows, {int(y_val.sum())} positives ({y_val.mean():.2%}).",
        f"Test: {len(test)} rows, {int(y_test.sum())} positives ({y_test.mean():.2%}).",
        f"Selected model (validation AP): {selected_name}.",
        f"Threshold selected on validation by F1: {threshold:.2f} (validation best F1={val_best_f1:.4f}).",
        "",
        "Final held-out test metrics:",
        pd.DataFrame([test_model, baseline]).to_string(index=False),
        "",
        "Warning: Only 14 positives exist in the test set; precision/recall/F1 are unstable with this small support.",
        "Accuracy alone is misleading for this imbalanced label (most days have no stock-out).",
        "All labels derive from a synthetic inventory simulator. These metrics do not establish real-world stock-out prediction quality.",
        "Baseline predicts no stock-out and is included to expose class-imbalance tradeoffs.",
    ]
    (REPORTS / "stockout_classifier_evaluation.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nSTOCK-OUT CLASSIFIER FINAL TEST METRICS")
    print(pd.DataFrame([test_model, baseline])[["Model", "Rows", "Positive_Labels", "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "F1", "Average_Precision_PR_AUC", "ROC_AUC", "False_Positives", "False_Negatives", "True_Positives"]].to_string(index=False))
    print(f"Selected classifier: {selected_name}; validation-selected threshold={threshold:.2f}")
    print("All classifier results are synthetic-only and should be interpreted cautiously.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

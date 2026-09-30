import os
import gc
import glob
import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from feature_decode import convert_features
print("ICU MORTALITY - XGBOOST FINAL TEST EVALUATION")
TEST_DIR = "/home/mahith/BDA_PROJECT/Model/xgb_clean/test_data/test_features"
MODEL_PATH = "/home/mahith/BDA_PROJECT/Model/models/icu_mortality_xgboost_clean.json"
THRESHOLD_PATH = "/home/mahith/BDA_PROJECT/Model/xgb_clean/results/optimal_threshold.json"
RESULTS_PATH = "/home/mahith/BDA_PROJECT/Model/xgb_clean/results/xgboost_test_results.txt"
EXPECTED_FEATURES = 137
BATCH_SIZE = 100000
print("\nCHECKING SOFTWARE")
print(f"XGBoost : {xgb.__version__}")
print("\nCHECKING MODEL")
if not os.path.exists(MODEL_PATH): raise RuntimeError(f"Model not found: {MODEL_PATH}")
booster = xgb.Booster()
booster.load_model(MODEL_PATH)
print("MODEL LOADED SUCCESSFULLY")
print(f"Model boosting rounds : {booster.num_boosted_rounds()}")
print(f"Model features       : {booster.num_features()}")
if booster.num_features() != EXPECTED_FEATURES: raise RuntimeError(f"Feature count mismatch: {booster.num_features()} != {EXPECTED_FEATURES}")
print("\nCHECKING EARLY STOPPING STATE")
best_iteration = getattr(booster, "best_iteration", None)
print(f"Best iteration : {best_iteration}")
if best_iteration is None or best_iteration < 0: raise RuntimeError("booster.best_iteration is missing or invalid.")
prediction_iteration_range = (0, best_iteration + 1)
print(f"Using iteration_range : {prediction_iteration_range}")
print("\nCHECKING LOCKED THRESHOLD")
if not os.path.exists(THRESHOLD_PATH): raise RuntimeError(f"Threshold file not found: {THRESHOLD_PATH}")
with open(THRESHOLD_PATH, "r") as f: threshold_data = json.load(f)
best_threshold = float(threshold_data["recommended_threshold"])
locked_best_iteration = int(threshold_data["best_iteration"])
print(f"Locked threshold : {best_threshold:.2f}")
print(f"Locked best iteration : {locked_best_iteration}")
print("\nVERIFYING MODEL CONSISTENCY")
if locked_best_iteration != best_iteration: raise RuntimeError(f"Model/threshold mismatch: threshold JSON has best_iteration={locked_best_iteration}, loaded model has best_iteration={best_iteration}")
print("CONSISTENCY CHECK PASSED")
print("\nCHECKING TEST PARQUET FILES")
test_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.parquet")))
print(f"Test partitions : {len(test_files)}")
if len(test_files) == 0: raise RuntimeError("No test Parquet files found.")
for i, path in enumerate(test_files):
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f"{i + 1:02d}. {os.path.basename(path)} : {size_mb:.1f} MB")
print("\nGENERATING TEST PREDICTIONS")
all_predictions = []
all_labels = []
prediction_start = time.time()
for i, path in enumerate(test_files):
    print(f"\nTEST PARTITION {i + 1}/{len(test_files)}")
    start = time.time()
    df = pd.read_parquet(path, columns=["features", "label"])
    print(f"Rows : {len(df):,}")
    for start_idx in range(0, len(df), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        feature_values = df["features"].iloc[start_idx:end_idx].tolist()
        y = df["label"].iloc[start_idx:end_idx].to_numpy(dtype=np.float32)
        X = np.vstack([convert_features(value) for value in feature_values]).astype(np.float32, copy=False)
        if X.shape[1] != booster.num_features(): raise RuntimeError(f"Feature count mismatch: {X.shape[1]} != {booster.num_features()}")
        predictions = booster.predict(xgb.DMatrix(X), iteration_range=prediction_iteration_range)
        all_predictions.append(predictions)
        all_labels.append(y)
        print(f"Batch {start_idx:,}-{end_idx:,} : {len(predictions):,} predictions")
        del feature_values, X, y, predictions
        gc.collect()
    print(f"Partition time : {(time.time() - start) / 60:.2f} minutes")
    del df
    gc.collect()
prediction_time = time.time() - prediction_start
print("\nTEST PREDICTIONS COMPLETED")
print(f"Prediction time : {prediction_time / 60:.2f} minutes")
print("\nCOMBINING TEST RESULTS")
y_true = np.concatenate(all_labels)
y_prob = np.concatenate(all_predictions)
del all_labels, all_predictions
gc.collect()
print(f"Test rows   : {len(y_true):,}")
print(f"Predictions : {len(y_prob):,}")
if len(y_true) != len(y_prob): raise RuntimeError("Number of labels and predictions do not match.")
print("\nCALCULATING TEST METRICS")
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
y_pred = (y_prob >= best_threshold).astype(np.int8)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print("\nFINAL TEST RESULTS")
print(f"Threshold   : {best_threshold:.2f}")
print(f"ROC-AUC     : {roc_auc:.4f}")
print(f"PR-AUC      : {pr_auc:.4f}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1-score    : {f1:.4f}")
print("\nCONFUSION MATRIX")
print(np.array([[tn, fp], [fn, tp]]))
print("\nCONFUSION MATRIX VALUES")
print(f"True Negatives  : {tn:,}")
print(f"False Positives : {fp:,}")
print(f"False Negatives : {fn:,}")
print(f"True Positives  : {tp:,}")
print("\nSAVING FINAL TEST RESULTS")
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    f.write("ICU MORTALITY - XGBOOST FINAL TEST EVALUATION\n")
    f.write("Model variant : 137-feature clean model\n\n")
    f.write(f"Test rows : {len(y_true):,}\n")
    f.write(f"Model total built rounds : {booster.num_boosted_rounds()}\n")
    f.write(f"Model best iteration : {best_iteration}\n")
    f.write(f"Prediction iteration_range : {prediction_iteration_range}\n")
    f.write(f"Locked threshold : {best_threshold:.6f}\n\n")
    f.write("FINAL TEST METRICS\n")
    f.write(f"ROC-AUC : {roc_auc:.6f}\n")
    f.write(f"PR-AUC : {pr_auc:.6f}\n")
    f.write(f"Accuracy : {accuracy:.6f}\n")
    f.write(f"Precision : {precision:.6f}\n")
    f.write(f"Recall : {recall:.6f}\n")
    f.write(f"Specificity : {specificity:.6f}\n")
    f.write(f"F1-score : {f1:.6f}\n\n")
    f.write("CONFUSION MATRIX\n")
    f.write(str(np.array([[tn, fp], [fn, tp]])))
    f.write("\n\nCONFUSION MATRIX VALUES\n")
    f.write(f"True Negatives : {tn:,}\n")
    f.write(f"False Positives : {fp:,}\n")
    f.write(f"False Negatives : {fn:,}\n")
    f.write(f"True Positives : {tp:,}\n")
print(f"RESULTS SAVED: {RESULTS_PATH}")
print("\nTEST DATA EVALUATION COMPLETED")
print("THRESHOLD WAS SELECTED FROM VALIDATION DATA ONLY")
print("CLEAN 137-FEATURE TEST EVALUATION COMPLETED")
del y_true, y_prob, y_pred, booster
gc.collect()
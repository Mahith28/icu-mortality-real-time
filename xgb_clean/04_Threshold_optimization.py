import os
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, fbeta_score
HDFS_VALIDATION = "hdfs://localhost:9000/user/mahith/icu/clean_features/validation_features"
VALIDATION_DIR = "/home/mahith/BDA_PROJECT/Model/xgb_clean/threshold_data/validation_features"
MODEL_PATH = "/home/mahith/BDA_PROJECT/Model/models/icu_mortality_xgboost_clean.json"
RESULTS_DIR = "/home/mahith/BDA_PROJECT/Model/xgb_clean/results"
RESULTS_PATH = "/home/mahith/BDA_PROJECT/Model/xgb_clean/results/threshold_optimization_results.txt"
CSV_RESULTS_PATH = "/home/mahith/BDA_PROJECT/Model/xgb_clean/results/threshold_optimization_results.csv"
JSON_RESULTS_PATH = "/home/mahith/BDA_PROJECT/Model/xgb_clean/results/optimal_threshold.json"
EXPECTED_FEATURES = 137
BATCH_SIZE = 100000
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.01), 2)
RECALL_FLOORS = [0.70, 0.75, 0.80, 0.85, 0.90]
PRIMARY_RECALL_TARGET = 0.80
os.makedirs(RESULTS_DIR, exist_ok=True)
def vector_struct_to_dense(v, expected_len):
    if hasattr(v, "toArray"):
        result = v.toArray().astype(np.float32)
        if len(result) != expected_len:
            raise RuntimeError(f"Expected {expected_len} values but found {len(result)}")
        return result
    if not isinstance(v, dict):
        raise TypeError(f"Unexpected vector type: {type(v)}")
    vector_type = v["type"]
    vector_size = v["size"]
    if vector_size != expected_len:
        raise RuntimeError(f"Expected vector size {expected_len} but found {vector_size}")
    if vector_type == 0:
        dense = np.zeros(expected_len, dtype=np.float32)
        indices = np.asarray(v["indices"], dtype=np.int32)
        values = np.asarray(v["values"], dtype=np.float32)
        if len(indices) != len(values):
            raise RuntimeError(f"Sparse index/value length mismatch: {len(indices)} vs {len(values)}")
        if len(indices) > 0 and (indices.min() < 0 or indices.max() >= expected_len):
            raise RuntimeError(f"Sparse index out of bounds for vector size {expected_len}")
        if len(indices) > 0:
            dense[indices] = values
        return dense
    if vector_type == 1:
        values = np.asarray(v["values"], dtype=np.float32)
        if len(values) != expected_len:
            raise RuntimeError(f"Expected {expected_len} values but found {len(values)}")
        return values
    raise RuntimeError(f"Unknown Spark vector type: {vector_type}")
def load_validation_data():
    if os.path.exists(VALIDATION_DIR):
        shutil.rmtree(VALIDATION_DIR)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    subprocess.run(["hdfs", "dfs", "-get", HDFS_VALIDATION + "/*", VALIDATION_DIR], check=True)
def main():
    print("XGBOOST CLEAN MODEL - THRESHOLD OPTIMIZATION")
    try:
        print("\nLoading model")
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        feature_count = model.num_features()
        print(f"Model features: {feature_count}")
        if feature_count != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, found {feature_count}")
        best_iteration = model.best_iteration
        if best_iteration is None or best_iteration < 0:
            raise RuntimeError("Model does not contain a valid best_iteration")
        prediction_range = (0, best_iteration + 1)
        print(f"Best iteration: {best_iteration}")
        print(f"Using iteration range: {prediction_range}")
        print("\nCopying validation data from HDFS")
        load_validation_data()
        parquet_files = []
        for root, _, files in os.walk(VALIDATION_DIR):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
        parquet_files.sort()
        if not parquet_files:
            raise FileNotFoundError("No Parquet files found in validation directory")
        print(f"Validation Parquet files: {len(parquet_files)}")
        all_predictions = []
        all_labels = []
        total_rows = 0
        print("\nGenerating validation predictions")
        for file_index, parquet_file in enumerate(parquet_files, 1):
            print(f"Processing file {file_index}/{len(parquet_files)}: {os.path.basename(parquet_file)}")
            parquet_data = pd.read_parquet(parquet_file)
            if "features" not in parquet_data.columns:
                raise ValueError("features column not found")
            if "label" not in parquet_data.columns:
                raise ValueError("label column not found")
            for start in range(0, len(parquet_data), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(parquet_data))
                batch = parquet_data.iloc[start:end]
                X = np.vstack([vector_struct_to_dense(v, EXPECTED_FEATURES) for v in batch["features"]]).astype(np.float32)
                y = batch["label"].to_numpy(dtype=np.float32)
                if X.shape[1] != EXPECTED_FEATURES:
                    raise ValueError(f"Invalid feature dimension: {X.shape[1]}")
                if np.isinf(X).any():
                    raise ValueError("Infinite values found in feature matrix")
                if not np.isfinite(y).all():
                    raise ValueError("Invalid labels found")
                dmatrix = xgb.DMatrix(X, missing=np.nan)
                predictions = model.predict(dmatrix, iteration_range=prediction_range)
                if len(predictions) != len(y):
                    raise ValueError("Prediction and label lengths do not match")
                all_predictions.extend(predictions.tolist())
                all_labels.extend(y.astype(int).tolist())
                total_rows += len(y)
                del X, y, dmatrix, predictions
        y_true = np.asarray(all_labels, dtype=np.int8)
        y_prob = np.asarray(all_predictions, dtype=np.float64)
        print(f"\nValidation rows: {total_rows}")
        print(f"Predictions: {len(y_prob)}")
        print(f"Labels: {len(y_true)}")
        if len(y_true) != len(y_prob):
            raise ValueError("Final prediction and label lengths do not match")
        print("\nComputing validation metrics...")
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        threshold_results = []
        for threshold in THRESHOLDS:
            y_pred = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            threshold_results.append({
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "f2": float(f2),
                "accuracy": float(accuracy),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            })
        results_df = pd.DataFrame(threshold_results)
        best_f1_row = results_df.loc[results_df["f1"].idxmax()]
        best_f2_row = results_df.loc[results_df["f2"].idxmax()]
        recall_candidates = {}
        for recall_floor in RECALL_FLOORS:
            candidates = results_df[results_df["recall"] >= recall_floor]
            if len(candidates) == 0:
                recall_candidates[str(recall_floor)] = None
            else:
                best_precision_row = candidates.loc[candidates["precision"].idxmax()]
                recall_candidates[str(recall_floor)] = {
                    "threshold": float(best_precision_row["threshold"]),
                    "precision": float(best_precision_row["precision"]),
                    "recall": float(best_precision_row["recall"]),
                    "specificity": float(best_precision_row["specificity"]),
                    "f1": float(best_precision_row["f1"]),
                    "f2": float(best_precision_row["f2"]),
                    "tn": int(best_precision_row["tn"]),
                    "fp": int(best_precision_row["fp"]),
                    "fn": int(best_precision_row["fn"]),
                    "tp": int(best_precision_row["tp"])
                }
        primary_recall_key = str(PRIMARY_RECALL_TARGET)
        if recall_candidates[primary_recall_key] is not None:
            recommended_threshold = float(recall_candidates[primary_recall_key]["threshold"])
            recommendation_method = f"best_precision_with_recall_at_least_{int(PRIMARY_RECALL_TARGET * 100)}_percent"
        else:
            recommended_threshold = float(best_f2_row["threshold"])
            recommendation_method = "best_f2"
        print("BEST F1 THRESHOLD")
        print(f"Threshold: {best_f1_row['threshold']:.2f}")
        print(f"Precision: {best_f1_row['precision']:.4f}")
        print(f"Recall: {best_f1_row['recall']:.4f}")
        print(f"Specificity: {best_f1_row['specificity']:.4f}")
        print(f"F1: {best_f1_row['f1']:.4f}")
        print(f"F2: {best_f1_row['f2']:.4f}")
        print("BEST F2 THRESHOLD")
        print(f"Threshold: {best_f2_row['threshold']:.2f}")
        print(f"Precision: {best_f2_row['precision']:.4f}")
        print(f"Recall: {best_f2_row['recall']:.4f}")
        print(f"Specificity: {best_f2_row['specificity']:.4f}")
        print(f"F1: {best_f2_row['f1']:.4f}")
        print(f"F2: {best_f2_row['f2']:.4f}")
        print("RECALL-CONSTRAINED THRESHOLDS")
        for recall_floor in RECALL_FLOORS:
            result = recall_candidates[str(recall_floor)]
            if result is None:
                print(f"Recall >= {recall_floor:.0%}: No threshold found")
            else:
                print(f"Recall >= {recall_floor:.0%}: Threshold={result['threshold']:.2f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}, Specificity={result['specificity']:.4f}")
        print("RECOMMENDED VALIDATION THRESHOLD")
        print(f"Threshold: {recommended_threshold:.2f}")
        print(f"Method: {recommendation_method}")
        results_df.to_csv(CSV_RESULTS_PATH, index=False)
        with open(RESULTS_PATH, "w") as f:
            f.write("XGBOOST CLEAN MODEL - THRESHOLD OPTIMIZATION\n")
            f.write(f"Validation rows: {total_rows}\n")
            f.write(f"Model features: {feature_count}\n")
            f.write(f"Best iteration: {best_iteration}\n")
            f.write(f"ROC-AUC: {roc_auc:.6f}\n")
            f.write(f"PR-AUC: {pr_auc:.6f}\n\n")
            f.write("BEST F1 THRESHOLD\n")
            f.write(f"Threshold: {best_f1_row['threshold']:.2f}\n")
            f.write(f"Precision: {best_f1_row['precision']:.6f}\n")
            f.write(f"Recall: {best_f1_row['recall']:.6f}\n")
            f.write(f"Specificity: {best_f1_row['specificity']:.6f}\n")
            f.write(f"F1: {best_f1_row['f1']:.6f}\n")
            f.write(f"F2: {best_f1_row['f2']:.6f}\n\n")
            f.write("BEST F2 THRESHOLD\n")
            f.write(f"Threshold: {best_f2_row['threshold']:.2f}\n")
            f.write(f"Precision: {best_f2_row['precision']:.6f}\n")
            f.write(f"Recall: {best_f2_row['recall']:.6f}\n")
            f.write(f"Specificity: {best_f2_row['specificity']:.6f}\n")
            f.write(f"F1: {best_f2_row['f1']:.6f}\n")
            f.write(f"F2: {best_f2_row['f2']:.6f}\n\n")
            f.write("RECALL-CONSTRAINED THRESHOLDS\n")
            for recall_floor in RECALL_FLOORS:
                result = recall_candidates[str(recall_floor)]
                if result is None:
                    f.write(f"Recall >= {recall_floor:.0%}: No threshold found\n")
                else:
                    f.write(f"Recall >= {recall_floor:.0%}: Threshold={result['threshold']:.2f}, Precision={result['precision']:.6f}, Recall={result['recall']:.6f}, Specificity={result['specificity']:.6f}\n")
            f.write("\n")
            f.write("RECOMMENDED VALIDATION THRESHOLD\n")
            f.write(f"Threshold: {recommended_threshold:.2f}\n")
            f.write(f"Method: {recommendation_method}\n")
        threshold_data = {
            "model_path": MODEL_PATH,
            "validation_path": HDFS_VALIDATION,
            "validation_rows": int(total_rows),
            "feature_count": int(feature_count),
            "best_iteration": int(best_iteration),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "best_f1": {
                "threshold": float(best_f1_row["threshold"]),
                "precision": float(best_f1_row["precision"]),
                "recall": float(best_f1_row["recall"]),
                "specificity": float(best_f1_row["specificity"]),
                "f1": float(best_f1_row["f1"]),
                "f2": float(best_f1_row["f2"]),
                "accuracy": float(best_f1_row["accuracy"]),
                "tn": int(best_f1_row["tn"]),
                "fp": int(best_f1_row["fp"]),
                "fn": int(best_f1_row["fn"]),
                "tp": int(best_f1_row["tp"])
            },
            "best_f2": {
                "threshold": float(best_f2_row["threshold"]),
                "precision": float(best_f2_row["precision"]),
                "recall": float(best_f2_row["recall"]),
                "specificity": float(best_f2_row["specificity"]),
                "f1": float(best_f2_row["f1"]),
                "f2": float(best_f2_row["f2"]),
                "accuracy": float(best_f2_row["accuracy"]),
                "tn": int(best_f2_row["tn"]),
                "fp": int(best_f2_row["fp"]),
                "fn": int(best_f2_row["fn"]),
                "tp": int(best_f2_row["tp"])
            },
            "recall_constrained_results": recall_candidates,
            "recommended_threshold": float(recommended_threshold),
            "recommendation_method": recommendation_method
        }
        with open(JSON_RESULTS_PATH, "w") as f:
            json.dump(threshold_data, f, indent=2)
        print("FILES SAVED")
        print(RESULTS_PATH)
        print(CSV_RESULTS_PATH)
        print(JSON_RESULTS_PATH)
    finally:
        if os.path.exists(VALIDATION_DIR):
            shutil.rmtree(VALIDATION_DIR)
            print("\nTemporary validation data removed.")
if __name__ == "__main__":
    main()
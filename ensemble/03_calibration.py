import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
OUTPUT_DIR="/home/mahith/BDA_PROJECT/Model/ensemble/results"
INPUT_FILE=os.path.join(OUTPUT_DIR,"validation_raw_predictions.parquet")
SPLIT_FILE=os.path.join(OUTPUT_DIR,"calibration_split.parquet")
CALIBRATED_FILE=os.path.join(OUTPUT_DIR,"calibrated_validation_predictions.parquet")
CALIBRATION_MODEL_FILE=os.path.join(OUTPUT_DIR,"calibration_models.json")
REPORT_FILE=os.path.join(OUTPUT_DIR,"calibration_results.txt")
RANDOM_SEED=42
CALIBRATION_FRACTION=0.60
TUNING_FRACTION=0.20
EVALUATION_FRACTION=0.20
EXPECTED_ROWS=184926
def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)
    df=pd.read_parquet(INPUT_FILE)
    required={"stay_id","window_id","label","xgb_probability","lstm_probability"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Missing columns: {sorted(required-set(df.columns))}")
    if len(df)!=EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} aligned rows but found {len(df)}")
    if df.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate stay_id/window_id keys found")
    if not np.isfinite(df["xgb_probability"]).all():
        raise RuntimeError("XGBoost probabilities contain invalid values")
    if not np.isfinite(df["lstm_probability"]).all():
        raise RuntimeError("LSTM probabilities contain invalid values")
    if ((df["xgb_probability"]<0)|(df["xgb_probability"]>1)).any():
        raise RuntimeError("XGBoost probabilities outside [0,1]")
    if ((df["lstm_probability"]<0)|(df["lstm_probability"]>1)).any():
        raise RuntimeError("LSTM probabilities outside [0,1]")
    print(f"Loaded aligned validation rows: {len(df)}")
    print(f"Loaded stays: {df['stay_id'].nunique()}")
    return df
def create_stay_split(df):
    stays=df[["stay_id","label"]].drop_duplicates("stay_id").copy()
    label_counts=stays.groupby("stay_id")["label"].nunique()
    if (label_counts>1).any():
        raise RuntimeError("Some stays have inconsistent labels")
    train_stays,tune_eval_stays=train_test_split(
        stays,
        test_size=TUNING_FRACTION+EVALUATION_FRACTION,
        stratify=stays["label"],
        random_state=RANDOM_SEED
    )
    relative_eval_fraction=EVALUATION_FRACTION/(TUNING_FRACTION+EVALUATION_FRACTION)
    tuning_stays,evaluation_stays=train_test_split(
        tune_eval_stays,
        test_size=relative_eval_fraction,
        stratify=tune_eval_stays["label"],
        random_state=RANDOM_SEED
    )
    split_map=pd.concat([
        train_stays.assign(split="calibration_fit"),
        tuning_stays.assign(split="ensemble_tuning"),
        evaluation_stays.assign(split="calibration_evaluation")
    ],ignore_index=True)
    if split_map["stay_id"].duplicated().any():
        raise RuntimeError("A stay appears in more than one split")
    if len(split_map)!=df["stay_id"].nunique():
        raise RuntimeError("Not every stay received exactly one split")
    df=df.merge(split_map[["stay_id","split"]],on="stay_id",how="left",validate="many_to_one")
    if df["split"].isna().any():
        raise RuntimeError("Some rows did not receive a split")
    split_counts=df.groupby("split").agg(
        rows=("stay_id","size"),
        stays=("stay_id","nunique"),
        positives=("label","sum"),
        prevalence=("label","mean")
    )
    print("Split summary:")
    print(split_counts.to_string())
    return df,split_counts
def fit_calibrator(probabilities,labels):
    probabilities=np.clip(np.asarray(probabilities,dtype=np.float64),1e-6,1-1e-6)
    logits=np.log(probabilities/(1-probabilities)).reshape(-1,1)
    labels=np.asarray(labels,dtype=np.int8)
    model=LogisticRegression(C=1e6,solver="lbfgs",max_iter=1000)
    model.fit(logits,labels)
    return model
def apply_calibrator(model,probabilities):
    probabilities=np.clip(np.asarray(probabilities,dtype=np.float64),1e-6,1-1e-6)
    logits=np.log(probabilities/(1-probabilities)).reshape(-1,1)
    return model.predict_proba(logits)[:,1]
def calculate_metrics(labels,probabilities):
    return {
        "roc_auc":float(roc_auc_score(labels,probabilities)),
        "pr_auc":float(average_precision_score(labels,probabilities)),
        "brier":float(brier_score_loss(labels,probabilities)),
        "log_loss":float(log_loss(labels,probabilities,labels=[0,1]))
    }
def evaluate_split(df,split_name):
    subset=df[df["split"]==split_name].copy()
    labels=subset["label"].to_numpy(dtype=np.int8)
    results={}
    for name,column in [
        ("xgb_raw","xgb_probability"),
        ("xgb_calibrated","xgb_calibrated_probability"),
        ("lstm_raw","lstm_probability"),
        ("lstm_calibrated","lstm_calibrated_probability")
    ]:
        results[name]=calculate_metrics(
            labels,
            subset[column].to_numpy(dtype=np.float64)
        )
    return results
def check_auc_invariance(results):
    xgb_auc_difference=abs(
        results["xgb_raw"]["roc_auc"]-
        results["xgb_calibrated"]["roc_auc"]
    )
    xgb_pr_difference=abs(
        results["xgb_raw"]["pr_auc"]-
        results["xgb_calibrated"]["pr_auc"]
    )
    lstm_auc_difference=abs(
        results["lstm_raw"]["roc_auc"]-
        results["lstm_calibrated"]["roc_auc"]
    )
    lstm_pr_difference=abs(
        results["lstm_raw"]["pr_auc"]-
        results["lstm_calibrated"]["pr_auc"]
    )
    if xgb_auc_difference>1e-9:
        raise RuntimeError(f"XGBoost ROC-AUC changed after monotonic calibration: {xgb_auc_difference}")
    if xgb_pr_difference>1e-9:
        raise RuntimeError(f"XGBoost PR-AUC changed after monotonic calibration: {xgb_pr_difference}")
    if lstm_auc_difference>1e-9:
        raise RuntimeError(f"LSTM ROC-AUC changed after monotonic calibration: {lstm_auc_difference}")
    if lstm_pr_difference>1e-9:
        raise RuntimeError(f"LSTM PR-AUC changed after monotonic calibration: {lstm_pr_difference}")
    print("Calibration AUC-invariance check: PASS")
def save_models(xgb_model,lstm_model):
    data={
        "method":"Platt scaling using logistic regression on logit-transformed probabilities",
        "random_seed":RANDOM_SEED,
        "xgb_intercept":float(xgb_model.intercept_[0]),
        "xgb_coefficient":float(xgb_model.coef_[0][0]),
        "lstm_intercept":float(lstm_model.intercept_[0]),
        "lstm_coefficient":float(lstm_model.coef_[0][0])
    }
    with open(CALIBRATION_MODEL_FILE,"w") as f:
        json.dump(data,f,indent=2)
def save_report(split_counts,fit_results,evaluation_results):
    with open(REPORT_FILE,"w") as f:
        f.write("Calibration phase results\n\n")
        f.write("Split summary:\n")
        f.write(split_counts.to_string())
        f.write("\n\n")
        f.write("Calibration-fit metrics are in-sample diagnostics and are not treated as held-out performance.\n\n")
        f.write("Calibration-fit:\n")
        for model_name,metrics in fit_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.10f}\n")
            f.write(f"PR-AUC: {metrics['pr_auc']:.10f}\n")
            f.write(f"Brier: {metrics['brier']:.10f}\n")
            f.write(f"Log-loss: {metrics['log_loss']:.10f}\n")
        f.write("\n")
        f.write("Calibration-evaluation:\n")
        for model_name,metrics in evaluation_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"ROC-AUC: {metrics['roc_auc']:.10f}\n")
            f.write(f"PR-AUC: {metrics['pr_auc']:.10f}\n")
            f.write(f"Brier: {metrics['brier']:.10f}\n")
            f.write(f"Log-loss: {metrics['log_loss']:.10f}\n")
        f.write("\n")
def main():
    df=load_data()
    df,split_counts=create_stay_split(df)
    calibration_df=df[df["split"]=="calibration_fit"].copy()
    labels=calibration_df["label"].to_numpy(dtype=np.int8)
    xgb_model=fit_calibrator(calibration_df["xgb_probability"],labels)
    lstm_model=fit_calibrator(calibration_df["lstm_probability"],labels)
    df["xgb_calibrated_probability"]=apply_calibrator(xgb_model,df["xgb_probability"])
    df["lstm_calibrated_probability"]=apply_calibrator(lstm_model,df["lstm_probability"])
    if not np.isfinite(df["xgb_calibrated_probability"]).all():
        raise RuntimeError("Invalid calibrated XGBoost probabilities")
    if not np.isfinite(df["lstm_calibrated_probability"]).all():
        raise RuntimeError("Invalid calibrated LSTM probabilities")
    if ((df["xgb_calibrated_probability"]<0)|(df["xgb_calibrated_probability"]>1)).any():
        raise RuntimeError("Calibrated XGBoost probabilities outside [0,1]")
    if ((df["lstm_calibrated_probability"]<0)|(df["lstm_calibrated_probability"]>1)).any():
        raise RuntimeError("Calibrated LSTM probabilities outside [0,1]")
    output_columns=[
        "stay_id",
        "window_id",
        "label",
        "xgb_probability",
        "lstm_probability",
        "xgb_calibrated_probability",
        "lstm_calibrated_probability",
        "split"
    ]
    df[output_columns].to_parquet(CALIBRATED_FILE,index=False)
    df[["stay_id","split"]].drop_duplicates().to_parquet(SPLIT_FILE,index=False)
    fit_results=evaluate_split(df,"calibration_fit")
    evaluation_results=evaluate_split(df,"calibration_evaluation")
    check_auc_invariance(fit_results)
    check_auc_invariance(evaluation_results)
    save_models(xgb_model,lstm_model)
    save_report(split_counts,fit_results,evaluation_results)
    print(f"Calibrated predictions saved: {CALIBRATED_FILE}")
    print(f"Split assignments saved: {SPLIT_FILE}")
    print(f"Calibration models saved: {CALIBRATION_MODEL_FILE}")
    print(f"Calibration report saved: {REPORT_FILE}")
    print("Calibration completed successfully")
if __name__=="__main__":
    main()
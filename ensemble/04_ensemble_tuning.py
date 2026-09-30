import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
OUTPUT_DIR="/home/mahith/BDA_PROJECT/Model/ensemble/results"
INPUT_FILE=os.path.join(OUTPUT_DIR,"calibrated_validation_predictions.parquet")
RESULTS_FILE=os.path.join(OUTPUT_DIR,"ensemble_weight_search.csv")
REPORT_FILE=os.path.join(OUTPUT_DIR,"ensemble_tuning_results.txt")
SELECTED_FILE=os.path.join(OUTPUT_DIR,"selected_ensemble_predictions.parquet")
WEIGHT_FILE=os.path.join(OUTPUT_DIR,"selected_ensemble_weight.txt")
WEIGHT_STEP=0.01
def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)
    df=pd.read_parquet(INPUT_FILE)
    required={"stay_id","window_id","label","xgb_probability","lstm_probability","xgb_calibrated_probability","lstm_calibrated_probability","split"}
    missing=required-set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if len(df)==0:
        raise RuntimeError("Input file is empty")
    if df.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate stay_id/window_id keys found")
    expected_splits={"calibration_fit","ensemble_tuning","calibration_evaluation"}
    actual_splits=set(df["split"].unique())
    if actual_splits!=expected_splits:
        raise RuntimeError(f"Unexpected split values: {sorted(actual_splits)}")
    for column in ["xgb_calibrated_probability","lstm_calibrated_probability"]:
        values=df[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{column} contains NaN or infinite values")
        if ((values<0)|(values>1)).any():
            raise RuntimeError(f"{column} contains values outside [0,1]")
    print(f"Loaded calibrated validation rows: {len(df)}")
    print(f"Loaded stays: {df['stay_id'].nunique()}")
    return df
def verify_split_integrity(df):
    split_names=["calibration_fit","ensemble_tuning","calibration_evaluation"]
    split_stays={name:set(df.loc[df["split"]==name,"stay_id"].unique()) for name in split_names}
    for i in range(len(split_names)):
        for j in range(i+1,len(split_names)):
            first=split_names[i]
            second=split_names[j]
            overlap=split_stays[first]&split_stays[second]
            if overlap:
                raise RuntimeError(f"Stay overlap between {first} and {second}: {len(overlap)}")
    assigned=set().union(*split_stays.values())
    if len(assigned)!=df["stay_id"].nunique():
        raise RuntimeError("Not every stay belongs to exactly one split")
    tuning_rows=int((df["split"]=="ensemble_tuning").sum())
    evaluation_rows=int((df["split"]=="calibration_evaluation").sum())
    if tuning_rows==0:
        raise RuntimeError("Ensemble-tuning partition is empty")
    if evaluation_rows==0:
        raise RuntimeError("Calibration-evaluation partition is empty")
    print(f"Calibration-fit stays: {len(split_stays['calibration_fit'])}")
    print(f"Ensemble-tuning stays: {len(split_stays['ensemble_tuning'])}")
    print(f"Calibration-evaluation stays: {len(split_stays['calibration_evaluation'])}")
    print(f"Ensemble-tuning rows: {tuning_rows}")
    print(f"Calibration-evaluation rows: {evaluation_rows}")
    print("Stay-level split integrity: PASS")
def calculate_metrics(labels,probabilities):
    return {
        "roc_auc":float(roc_auc_score(labels,probabilities)),
        "pr_auc":float(average_precision_score(labels,probabilities)),
        "brier":float(brier_score_loss(labels,probabilities)),
        "log_loss":float(log_loss(labels,probabilities,labels=[0,1]))
    }
def search_weights(tuning_df):
    labels=tuning_df["label"].to_numpy(dtype=np.int8)
    xgb=tuning_df["xgb_calibrated_probability"].to_numpy(dtype=np.float64)
    lstm=tuning_df["lstm_calibrated_probability"].to_numpy(dtype=np.float64)
    records=[]
    weights=np.round(np.arange(0.0,1.0+WEIGHT_STEP/2,WEIGHT_STEP),2)
    for xgb_weight in weights:
        lstm_weight=1.0-xgb_weight
        ensemble_probability=xgb_weight*xgb+lstm_weight*lstm
        metrics=calculate_metrics(labels,ensemble_probability)
        records.append({
            "xgb_weight":float(xgb_weight),
            "lstm_weight":float(lstm_weight),
            "roc_auc":metrics["roc_auc"],
            "pr_auc":metrics["pr_auc"],
            "brier":metrics["brier"],
            "log_loss":metrics["log_loss"]
        })
    results=pd.DataFrame(records)
    results=results.sort_values(
        ["pr_auc","roc_auc","brier","log_loss"],
        ascending=[False,False,True,True]
    ).reset_index(drop=True)
    return results
def evaluate_selected_weight(df,best_xgb_weight):
    evaluation=df[df["split"]=="calibration_evaluation"].copy()
    labels=evaluation["label"].to_numpy(dtype=np.int8)
    xgb=evaluation["xgb_calibrated_probability"].to_numpy(dtype=np.float64)
    lstm=evaluation["lstm_calibrated_probability"].to_numpy(dtype=np.float64)
    best_lstm_weight=1.0-best_xgb_weight
    ensemble_probability=best_xgb_weight*xgb+best_lstm_weight*lstm
    ensemble_metrics=calculate_metrics(labels,ensemble_probability)
    xgb_metrics=calculate_metrics(labels,xgb)
    lstm_metrics=calculate_metrics(labels,lstm)
    evaluation["ensemble_probability"]=ensemble_probability
    evaluation["xgb_weight"]=best_xgb_weight
    evaluation["lstm_weight"]=best_lstm_weight
    return evaluation,xgb_metrics,lstm_metrics,ensemble_metrics
def calculate_improvements(xgb_metrics,lstm_metrics,ensemble_metrics):
    if xgb_metrics["pr_auc"]>=lstm_metrics["pr_auc"]:
        best_single_name="XGBoost"
        best_single=xgb_metrics
    else:
        best_single_name="LSTM"
        best_single=lstm_metrics
    improvements={
        "best_single_model":best_single_name,
        "auroc_absolute":ensemble_metrics["roc_auc"]-best_single["roc_auc"],
        "auroc_relative_percent":100*(ensemble_metrics["roc_auc"]-best_single["roc_auc"])/best_single["roc_auc"],
        "pr_auc_absolute":ensemble_metrics["pr_auc"]-best_single["pr_auc"],
        "pr_auc_relative_percent":100*(ensemble_metrics["pr_auc"]-best_single["pr_auc"])/best_single["pr_auc"],
        "brier_absolute":ensemble_metrics["brier"]-best_single["brier"],
        "brier_relative_percent":100*(ensemble_metrics["brier"]-best_single["brier"])/best_single["brier"],
        "log_loss_absolute":ensemble_metrics["log_loss"]-best_single["log_loss"],
        "log_loss_relative_percent":100*(ensemble_metrics["log_loss"]-best_single["log_loss"])/best_single["log_loss"]
    }
    return improvements
def analyze_weight_surface(results):
    top=results.head(10).copy()
    best_pr=float(top.iloc[0]["pr_auc"])
    top["pr_auc_difference_from_best"]=best_pr-top["pr_auc"]
    pr_range=float(top["pr_auc"].max()-top["pr_auc"].min())
    return top,pr_range
def save_outputs(results,best_xgb_weight,evaluation,xgb_metrics,lstm_metrics,ensemble_metrics,improvements,top,pr_range):
    results.to_csv(RESULTS_FILE,index=False)
    evaluation[["stay_id","window_id","label","xgb_calibrated_probability","lstm_calibrated_probability","ensemble_probability","xgb_weight","lstm_weight"]].to_parquet(SELECTED_FILE,index=False)
    best_lstm_weight=1.0-best_xgb_weight
    with open(WEIGHT_FILE,"w") as f:
        f.write(f"XGB weight: {best_xgb_weight:.2f}\n")
        f.write(f"LSTM weight: {best_lstm_weight:.2f}\n")
    with open(REPORT_FILE,"w") as f:
        f.write("Ensemble tuning results\n\n")
        f.write("Primary tuning metric: AUPRC\n")
        f.write(f"Weight search step: {WEIGHT_STEP:.2f}\n")
        f.write("Weight search population: ensemble_tuning only\n")
        f.write("Calibration-evaluation was not used for weight selection.\n")
        f.write("Calibration-evaluation was evaluated once using the selected weight.\n")
        f.write("MIMIC test was not used.\n\n")
        f.write(f"Selected XGB weight: {best_xgb_weight:.2f}\n")
        f.write(f"Selected LSTM weight: {best_lstm_weight:.2f}\n\n")
        if best_xgb_weight in (0.0,1.0):
            f.write("WARNING: selected weight is a boundary case.\n")
            f.write("The weighted ensemble therefore degenerates to a single model.\n\n")
        else:
            f.write("Selected weight is an interior blend.\n\n")
        f.write(f"Top-10 PR-AUC range: {pr_range:.10f}\n")
        if pr_range<1e-4:
            f.write("NOTE: top weight configurations are very close in PR-AUC; the optimum may be relatively flat.\n")
        else:
            f.write("Top weight configurations show measurable PR-AUC separation.\n")
        f.write("\nCalibration-evaluation metrics:\n\n")
        f.write("Calibrated XGBoost:\n")
        f.write(f"ROC-AUC: {xgb_metrics['roc_auc']:.10f}\n")
        f.write(f"PR-AUC: {xgb_metrics['pr_auc']:.10f}\n")
        f.write(f"Brier: {xgb_metrics['brier']:.10f}\n")
        f.write(f"Log-loss: {xgb_metrics['log_loss']:.10f}\n\n")
        f.write("Calibrated LSTM:\n")
        f.write(f"ROC-AUC: {lstm_metrics['roc_auc']:.10f}\n")
        f.write(f"PR-AUC: {lstm_metrics['pr_auc']:.10f}\n")
        f.write(f"Brier: {lstm_metrics['brier']:.10f}\n")
        f.write(f"Log-loss: {lstm_metrics['log_loss']:.10f}\n\n")
        f.write("Selected ensemble:\n")
        f.write(f"ROC-AUC: {ensemble_metrics['roc_auc']:.10f}\n")
        f.write(f"PR-AUC: {ensemble_metrics['pr_auc']:.10f}\n")
        f.write(f"Brier: {ensemble_metrics['brier']:.10f}\n")
        f.write(f"Log-loss: {ensemble_metrics['log_loss']:.10f}\n\n")
        f.write(f"Best single model by calibration-evaluation PR-AUC: {improvements['best_single_model']}\n\n")
        f.write("Ensemble minus best single model:\n")
        f.write(f"AUROC absolute difference: {improvements['auroc_absolute']:.10f}\n")
        f.write(f"AUROC relative difference (%): {improvements['auroc_relative_percent']:.6f}\n")
        f.write(f"PR-AUC absolute difference: {improvements['pr_auc_absolute']:.10f}\n")
        f.write(f"PR-AUC relative difference (%): {improvements['pr_auc_relative_percent']:.6f}\n")
        f.write(f"Brier absolute difference: {improvements['brier_absolute']:.10f}\n")
        f.write(f"Brier relative difference (%): {improvements['brier_relative_percent']:.6f}\n")
        f.write(f"Log-loss absolute difference: {improvements['log_loss_absolute']:.10f}\n")
        f.write(f"Log-loss relative difference (%): {improvements['log_loss_relative_percent']:.6f}\n\n")
        f.write("Top 10 ensemble configurations on ensemble-tuning:\n")
        f.write(top.to_string(index=False))
        f.write("\n")
def main():
    df=load_data()
    verify_split_integrity(df)
    tuning_df=df[df["split"]=="ensemble_tuning"].copy()
    results=search_weights(tuning_df)
    best_xgb_weight=float(results.iloc[0]["xgb_weight"])
    best_lstm_weight=1.0-best_xgb_weight
    if not 0.0<=best_xgb_weight<=1.0:
        raise RuntimeError("Invalid selected XGB weight")
    top,pr_range=analyze_weight_surface(results)
    if best_xgb_weight in (0.0,1.0):
        print("WARNING: selected weight is a boundary case")
    evaluation,xgb_metrics,lstm_metrics,ensemble_metrics=evaluate_selected_weight(df,best_xgb_weight)
    improvements=calculate_improvements(xgb_metrics,lstm_metrics,ensemble_metrics)
    save_outputs(results,best_xgb_weight,evaluation,xgb_metrics,lstm_metrics,ensemble_metrics,improvements,top,pr_range)
    print(f"Selected XGBoost weight: {best_xgb_weight:.2f}")
    print(f"Selected LSTM weight: {best_lstm_weight:.2f}")
    print(f"Top-10 PR-AUC range: {pr_range:.10f}")
    print(f"Best single model by calibration-evaluation PR-AUC: {improvements['best_single_model']}")
    print(f"Calibration-evaluation XGBoost ROC-AUC: {xgb_metrics['roc_auc']:.6f}")
    print(f"Calibration-evaluation LSTM ROC-AUC: {lstm_metrics['roc_auc']:.6f}")
    print(f"Calibration-evaluation ensemble ROC-AUC: {ensemble_metrics['roc_auc']:.6f}")
    print(f"Calibration-evaluation XGBoost PR-AUC: {xgb_metrics['pr_auc']:.6f}")
    print(f"Calibration-evaluation LSTM PR-AUC: {lstm_metrics['pr_auc']:.6f}")
    print(f"Calibration-evaluation ensemble PR-AUC: {ensemble_metrics['pr_auc']:.6f}")
    print(f"Calibration-evaluation XGBoost Brier: {xgb_metrics['brier']:.6f}")
    print(f"Calibration-evaluation LSTM Brier: {lstm_metrics['brier']:.6f}")
    print(f"Calibration-evaluation ensemble Brier: {ensemble_metrics['brier']:.6f}")
    print(f"Calibration-evaluation XGBoost log-loss: {xgb_metrics['log_loss']:.6f}")
    print(f"Calibration-evaluation LSTM log-loss: {lstm_metrics['log_loss']:.6f}")
    print(f"Calibration-evaluation ensemble log-loss: {ensemble_metrics['log_loss']:.6f}")
    print(f"Ensemble minus best single PR-AUC: {improvements['pr_auc_absolute']:.6f}")
    print(f"Ensemble minus best single AUROC: {improvements['auroc_absolute']:.6f}")
    print(f"Ensemble minus best single Brier: {improvements['brier_absolute']:.6f}")
    print(f"Ensemble minus best single log-loss: {improvements['log_loss_absolute']:.6f}")
    print(f"Weight search saved: {RESULTS_FILE}")
    print(f"Selected evaluation predictions saved: {SELECTED_FILE}")
    print(f"Selected weights saved: {WEIGHT_FILE}")
    print(f"Ensemble tuning report saved: {REPORT_FILE}")
    print("Ensemble tuning completed successfully")
if __name__=="__main__":
    main()
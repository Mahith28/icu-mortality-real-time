import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

OUTPUT_DIR="/home/mahith/BDA_PROJECT/Model/ensemble/results"
INPUT_FILE=os.path.join(OUTPUT_DIR,"selected_ensemble_predictions.parquet")
RESULTS_FILE=os.path.join(OUTPUT_DIR,"ensemble_bootstrap_results.csv")
REPORT_FILE=os.path.join(OUTPUT_DIR,"ensemble_bootstrap_report.txt")
N_BOOTSTRAPS=2000
RANDOM_SEED=42
CI_LEVEL=0.95
def load_data():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(INPUT_FILE)
    df=pd.read_parquet(INPUT_FILE)
    required={"stay_id","window_id","label","xgb_calibrated_probability","lstm_calibrated_probability","ensemble_probability","xgb_weight","lstm_weight"}
    missing=required-set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if len(df)==0:
        raise RuntimeError("Input file is empty")
    if df.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate stay_id/window_id keys found")
    for column in ["xgb_calibrated_probability","lstm_calibrated_probability","ensemble_probability"]:
        values=df[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"{column} contains NaN or infinite values")
        if ((values<0)|(values>1)).any():
            raise RuntimeError(f"{column} contains values outside [0,1]")
    labels=df["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels,[0,1]).all():
        raise RuntimeError("Labels must contain only 0 and 1")
    weights=df[["xgb_weight","lstm_weight"]].drop_duplicates()
    if len(weights)!=1:
        raise RuntimeError("Multiple ensemble weights found")
    xgb_weight=float(weights.iloc[0]["xgb_weight"])
    lstm_weight=float(weights.iloc[0]["lstm_weight"])
    if not np.isclose(xgb_weight+lstm_weight,1.0):
        raise RuntimeError("Ensemble weights do not sum to 1")
    print(f"Loaded calibration-evaluation rows: {len(df)}")
    print(f"Loaded calibration-evaluation stays: {df['stay_id'].nunique()}")
    print(f"Locked XGBoost weight: {xgb_weight:.2f}")
    print(f"Locked LSTM weight: {lstm_weight:.2f}")
    return df
def prepare_stays(df):
    stays=df["stay_id"].drop_duplicates().to_numpy()
    stay_groups={stay:group for stay,group in df.groupby("stay_id",sort=False)}
    if len(stays)<2:
        raise RuntimeError("Not enough stays for bootstrap")
    return stays,stay_groups
def calculate_metrics(df):
    labels=df["label"].to_numpy(dtype=np.int8)
    ensemble=df["ensemble_probability"].to_numpy(dtype=np.float64)
    xgb=df["xgb_calibrated_probability"].to_numpy(dtype=np.float64)
    lstm=df["lstm_calibrated_probability"].to_numpy(dtype=np.float64)
    ensemble_roc=roc_auc_score(labels,ensemble)
    xgb_roc=roc_auc_score(labels,xgb)
    lstm_roc=roc_auc_score(labels,lstm)
    ensemble_pr=average_precision_score(labels,ensemble)
    xgb_pr=average_precision_score(labels,xgb)
    lstm_pr=average_precision_score(labels,lstm)
    if xgb_pr>=lstm_pr:
        best_single_name="XGBoost"
        best_single_roc=xgb_roc
        best_single_pr=xgb_pr
    else:
        best_single_name="LSTM"
        best_single_roc=lstm_roc
        best_single_pr=lstm_pr
    return {
        "ensemble_roc_auc":float(ensemble_roc),
        "xgb_roc_auc":float(xgb_roc),
        "lstm_roc_auc":float(lstm_roc),
        "ensemble_pr_auc":float(ensemble_pr),
        "xgb_pr_auc":float(xgb_pr),
        "lstm_pr_auc":float(lstm_pr),
        "best_single_model":best_single_name,
        "best_single_roc_auc":float(best_single_roc),
        "best_single_pr_auc":float(best_single_pr),
        "roc_auc_difference":float(ensemble_roc-best_single_roc),
        "pr_auc_difference":float(ensemble_pr-best_single_pr)
    }
def bootstrap(df):
    stays,stay_groups=prepare_stays(df)
    rng=np.random.default_rng(RANDOM_SEED)
    results=[]
    for bootstrap_id in range(N_BOOTSTRAPS):
        sampled_stays=rng.choice(stays,size=len(stays),replace=True)
        sampled_parts=[stay_groups[stay] for stay in sampled_stays]
        sample=pd.concat(sampled_parts,ignore_index=True)
        labels=sample["label"].to_numpy(dtype=np.int8)
        ensemble=sample["ensemble_probability"].to_numpy(dtype=np.float64)
        xgb=sample["xgb_calibrated_probability"].to_numpy(dtype=np.float64)
        lstm=sample["lstm_calibrated_probability"].to_numpy(dtype=np.float64)
        if len(np.unique(labels))<2:
            continue
        ensemble_roc=roc_auc_score(labels,ensemble)
        xgb_roc=roc_auc_score(labels,xgb)
        lstm_roc=roc_auc_score(labels,lstm)
        ensemble_pr=average_precision_score(labels,ensemble)
        xgb_pr=average_precision_score(labels,xgb)
        lstm_pr=average_precision_score(labels,lstm)
        if xgb_pr>=lstm_pr:
            best_single_name="XGBoost"
            best_single_roc=xgb_roc
            best_single_pr=xgb_pr
        else:
            best_single_name="LSTM"
            best_single_roc=lstm_roc
            best_single_pr=lstm_pr
        results.append({
            "bootstrap_id":bootstrap_id,
            "best_single_model":best_single_name,
            "ensemble_roc_auc":ensemble_roc,
            "best_single_roc_auc":best_single_roc,
            "ensemble_pr_auc":ensemble_pr,
            "best_single_pr_auc":best_single_pr,
            "roc_auc_difference":ensemble_roc-best_single_roc,
            "pr_auc_difference":ensemble_pr-best_single_pr
        })
    results=pd.DataFrame(results)
    if len(results)<N_BOOTSTRAPS*0.99:
        raise RuntimeError(f"Too many invalid bootstrap samples: {len(results)}/{N_BOOTSTRAPS}")
    return results
def calculate_ci(values):
    alpha=1-CI_LEVEL
    lower=np.quantile(values,alpha/2)
    upper=np.quantile(values,1-alpha/2)
    return float(lower),float(upper)
def save_results(df,bootstrap_results):
    bootstrap_results.to_csv(RESULTS_FILE,index=False)
    observed=calculate_metrics(df)
    ensemble_pr_ci=calculate_ci(bootstrap_results["ensemble_pr_auc"].to_numpy())
    best_single_pr_ci=calculate_ci(bootstrap_results["best_single_pr_auc"].to_numpy())
    ensemble_roc_ci=calculate_ci(bootstrap_results["ensemble_roc_auc"].to_numpy())
    best_single_roc_ci=calculate_ci(bootstrap_results["best_single_roc_auc"].to_numpy())
    pr_difference_ci=calculate_ci(bootstrap_results["pr_auc_difference"].to_numpy())
    roc_difference_ci=calculate_ci(bootstrap_results["roc_auc_difference"].to_numpy())
    pr_positive_probability=float((bootstrap_results["pr_auc_difference"]>0).mean())
    roc_positive_probability=float((bootstrap_results["roc_auc_difference"]>0).mean())
    with open(REPORT_FILE,"w") as f:
        f.write("Ensemble bootstrap robustness analysis\n\n")
        f.write("Evaluation population: calibration-evaluation\n")
        f.write("Bootstrap unit: stay_id\n")
        f.write(f"Number of bootstrap replicates: {len(bootstrap_results)}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Confidence level: {CI_LEVEL:.2f}\n")
        f.write("Ensemble weights were fixed before bootstrap analysis.\n")
        f.write("MIMIC test data was not used.\n\n")
        f.write("Observed metrics:\n\n")
        f.write(f"Best single model by PR-AUC: {observed['best_single_model']}\n")
        f.write(f"Ensemble ROC-AUC: {observed['ensemble_roc_auc']:.6f}\n")
        f.write(f"Best single ROC-AUC: {observed['best_single_roc_auc']:.6f}\n")
        f.write(f"Ensemble PR-AUC: {observed['ensemble_pr_auc']:.6f}\n")
        f.write(f"Best single PR-AUC: {observed['best_single_pr_auc']:.6f}\n")
        f.write(f"ROC-AUC difference: {observed['roc_auc_difference']:.6f}\n")
        f.write(f"PR-AUC difference: {observed['pr_auc_difference']:.6f}\n\n")
        f.write("Individual model observed metrics:\n\n")
        f.write(f"XGBoost ROC-AUC: {observed['xgb_roc_auc']:.6f}\n")
        f.write(f"LSTM ROC-AUC: {observed['lstm_roc_auc']:.6f}\n")
        f.write(f"XGBoost PR-AUC: {observed['xgb_pr_auc']:.6f}\n")
        f.write(f"LSTM PR-AUC: {observed['lstm_pr_auc']:.6f}\n\n")
        f.write("95% bootstrap confidence intervals:\n\n")
        f.write(f"Ensemble ROC-AUC: {ensemble_roc_ci[0]:.6f} to {ensemble_roc_ci[1]:.6f}\n")
        f.write(f"Best single ROC-AUC: {best_single_roc_ci[0]:.6f} to {best_single_roc_ci[1]:.6f}\n")
        f.write(f"Ensemble PR-AUC: {ensemble_pr_ci[0]:.6f} to {ensemble_pr_ci[1]:.6f}\n")
        f.write(f"Best single PR-AUC: {best_single_pr_ci[0]:.6f} to {best_single_pr_ci[1]:.6f}\n")
        f.write(f"ROC-AUC difference: {roc_difference_ci[0]:.6f} to {roc_difference_ci[1]:.6f}\n")
        f.write(f"PR-AUC difference: {pr_difference_ci[0]:.6f} to {pr_difference_ci[1]:.6f}\n\n")
        f.write("Bootstrap probability of positive improvement:\n\n")
        f.write(f"PR-AUC: {pr_positive_probability:.6f}\n")
        f.write(f"ROC-AUC: {roc_positive_probability:.6f}\n")
    return observed,pr_difference_ci,roc_difference_ci,pr_positive_probability,roc_positive_probability
def main():
    df=load_data()
    stays,_=prepare_stays(df)
    print(f"Bootstrap stays per replicate: {len(stays)}")
    print(f"Bootstrap replicates: {N_BOOTSTRAPS}")
    bootstrap_results=bootstrap(df)
    observed,pr_difference_ci,roc_difference_ci,pr_positive_probability,roc_positive_probability=save_results(df,bootstrap_results)
    print(f"Best single model by PR-AUC: {observed['best_single_model']}")
    print(f"Observed ensemble PR-AUC: {observed['ensemble_pr_auc']:.6f}")
    print(f"Observed best single PR-AUC: {observed['best_single_pr_auc']:.6f}")
    print(f"Observed PR-AUC difference: {observed['pr_auc_difference']:.6f}")
    print(f"95% PR-AUC difference CI: {pr_difference_ci[0]:.6f} to {pr_difference_ci[1]:.6f}")
    print(f"Bootstrap probability PR-AUC improvement > 0: {pr_positive_probability:.6f}")
    print(f"Observed ensemble ROC-AUC: {observed['ensemble_roc_auc']:.6f}")
    print(f"Observed best single ROC-AUC: {observed['best_single_roc_auc']:.6f}")
    print(f"Observed ROC-AUC difference: {observed['roc_auc_difference']:.6f}")
    print(f"95% ROC-AUC difference CI: {roc_difference_ci[0]:.6f} to {roc_difference_ci[1]:.6f}")
    print(f"Bootstrap probability ROC-AUC improvement > 0: {roc_positive_probability:.6f}")
    print(f"Bootstrap results saved: {RESULTS_FILE}")
    print(f"Bootstrap report saved: {REPORT_FILE}")
    print("Ensemble bootstrap analysis completed successfully")
if __name__=="__main__":
    main()
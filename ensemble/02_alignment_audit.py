import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
OUTPUT_DIR="/home/mahith/BDA_PROJECT/Model/ensemble/results"
XGB_FILE=os.path.join(OUTPUT_DIR,"xgb_validation_predictions.parquet")
LSTM_FILE=os.path.join(OUTPUT_DIR,"lstm_validation_predictions.parquet")
ALIGNED_FILE=os.path.join(OUTPUT_DIR,"validation_raw_predictions.parquet")
AUDIT_FILE=os.path.join(OUTPUT_DIR,"alignment_audit.txt")
EXPECTED_XGB_ROWS=1610439
EXPECTED_LSTM_ROWS=184926
EXPECTED_STAYS=14402
def load_predictions():
    if not os.path.exists(XGB_FILE):
        raise FileNotFoundError(XGB_FILE)
    if not os.path.exists(LSTM_FILE):
        raise FileNotFoundError(LSTM_FILE)
    if not os.path.exists(ALIGNED_FILE):
        raise FileNotFoundError(ALIGNED_FILE)
    xgb=pd.read_parquet(XGB_FILE)
    lstm=pd.read_parquet(LSTM_FILE)
    existing=pd.read_parquet(ALIGNED_FILE)
    print(f"XGBoost prediction rows: {len(xgb)}")
    print(f"LSTM prediction rows: {len(lstm)}")
    print(f"Persisted aligned rows: {len(existing)}")
    return xgb,lstm,existing
def check_schema(xgb,lstm,existing):
    xgb_required={"stay_id","window_id","label","xgb_probability"}
    lstm_required={"stay_id","window_id","label","lstm_probability"}
    aligned_required={"stay_id","window_id","label","xgb_probability","lstm_probability"}
    if not xgb_required.issubset(xgb.columns):
        raise RuntimeError(f"XGBoost columns missing: {sorted(xgb_required-set(xgb.columns))}")
    if not lstm_required.issubset(lstm.columns):
        raise RuntimeError(f"LSTM columns missing: {sorted(lstm_required-set(lstm.columns))}")
    if not aligned_required.issubset(existing.columns):
        raise RuntimeError(f"Persisted aligned columns missing: {sorted(aligned_required-set(existing.columns))}")
    print("Schema check: PASS")
def check_row_counts(xgb,lstm,existing):
    if len(xgb)!=EXPECTED_XGB_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_XGB_ROWS} XGBoost rows but found {len(xgb)}")
    if len(lstm)!=EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LSTM_ROWS} LSTM rows but found {len(lstm)}")
    if len(existing)!=EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LSTM_ROWS} persisted aligned rows but found {len(existing)}")
    print("Row-count check: PASS")
def check_duplicates(xgb,lstm,existing):
    xgb_duplicates=int(xgb.duplicated(["stay_id","window_id"]).sum())
    lstm_duplicates=int(lstm.duplicated(["stay_id","window_id"]).sum())
    existing_duplicates=int(existing.duplicated(["stay_id","window_id"]).sum())
    print(f"XGBoost duplicate keys: {xgb_duplicates}")
    print(f"LSTM duplicate keys: {lstm_duplicates}")
    print(f"Persisted aligned duplicate keys: {existing_duplicates}")
    if xgb_duplicates!=0:
        raise RuntimeError("Duplicate XGBoost prediction keys found")
    if lstm_duplicates!=0:
        raise RuntimeError("Duplicate LSTM prediction keys found")
    if existing_duplicates!=0:
        raise RuntimeError("Duplicate keys found in persisted aligned output")
    print("Duplicate-key check: PASS")
def check_stays(xgb,lstm,existing):
    xgb_stays=set(xgb["stay_id"].unique())
    lstm_stays=set(lstm["stay_id"].unique())
    aligned_stays=set(existing["stay_id"].unique())
    print(f"XGBoost stays: {len(xgb_stays)}")
    print(f"LSTM stays: {len(lstm_stays)}")
    print(f"Persisted aligned stays: {len(aligned_stays)}")
    if xgb_stays!=lstm_stays:
        only_xgb=xgb_stays-lstm_stays
        only_lstm=lstm_stays-xgb_stays
        print(f"XGBoost-only stays: {len(only_xgb)}")
        print(f"LSTM-only stays: {len(only_lstm)}")
        raise RuntimeError("XGBoost and LSTM stay sets do not match")
    if len(xgb_stays)!=EXPECTED_STAYS:
        raise RuntimeError(f"Expected {EXPECTED_STAYS} stays but found {len(xgb_stays)}")
    if not aligned_stays.issubset(xgb_stays):
        raise RuntimeError("Persisted aligned output contains unknown stays")
    print("Stay-set check: PASS")
def check_labels(xgb,lstm,existing):
    xgb_values=set(np.unique(xgb["label"]))
    lstm_values=set(np.unique(lstm["label"]))
    aligned_values=set(np.unique(existing["label"]))
    print(f"XGBoost label values: {sorted(xgb_values)}")
    print(f"LSTM label values: {sorted(lstm_values)}")
    print(f"Aligned label values: {sorted(aligned_values)}")
    if not xgb_values.issubset({0,1}) or not lstm_values.issubset({0,1}) or not aligned_values.issubset({0,1}):
        raise RuntimeError("Labels contain values other than 0 and 1")
    print("Label-value check: PASS")
def check_probabilities(xgb,lstm,existing):
    xgb_prob=xgb["xgb_probability"].to_numpy(dtype=np.float64)
    lstm_prob=lstm["lstm_probability"].to_numpy(dtype=np.float64)
    aligned_xgb=existing["xgb_probability"].to_numpy(dtype=np.float64)
    aligned_lstm=existing["lstm_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(xgb_prob).all():
        raise RuntimeError("XGBoost probabilities contain NaN or infinite values")
    if not np.isfinite(lstm_prob).all():
        raise RuntimeError("LSTM probabilities contain NaN or infinite values")
    if not np.isfinite(aligned_xgb).all():
        raise RuntimeError("Persisted aligned XGBoost probabilities contain NaN or infinite values")
    if not np.isfinite(aligned_lstm).all():
        raise RuntimeError("Persisted aligned LSTM probabilities contain NaN or infinite values")
    if ((xgb_prob<0)|(xgb_prob>1)).any():
        raise RuntimeError("XGBoost probabilities outside [0,1]")
    if ((lstm_prob<0)|(lstm_prob>1)).any():
        raise RuntimeError("LSTM probabilities outside [0,1]")
    if ((aligned_xgb<0)|(aligned_xgb>1)).any():
        raise RuntimeError("Persisted aligned XGBoost probabilities outside [0,1]")
    if ((aligned_lstm<0)|(aligned_lstm>1)).any():
        raise RuntimeError("Persisted aligned LSTM probabilities outside [0,1]")
    print(f"XGBoost probability range: {xgb_prob.min():.6f} to {xgb_prob.max():.6f}")
    print(f"LSTM probability range: {lstm_prob.min():.6f} to {lstm_prob.max():.6f}")
    print(f"Persisted aligned XGBoost probability range: {aligned_xgb.min():.6f} to {aligned_xgb.max():.6f}")
    print(f"Persisted aligned LSTM probability range: {aligned_lstm.min():.6f} to {aligned_lstm.max():.6f}")
    print("Probability-value check: PASS")
def recompute_alignment(xgb,lstm):
    merged=xgb.merge(
        lstm,
        on=["stay_id","window_id"],
        how="outer",
        suffixes=("_xgb","_lstm"),
        indicator=True,
        validate="one_to_one"
    )
    both=int((merged["_merge"]=="both").sum())
    xgb_only=int((merged["_merge"]=="left_only").sum())
    lstm_only=int((merged["_merge"]=="right_only").sum())
    print(f"Matched prediction rows: {both}")
    print(f"XGBoost-only rows: {xgb_only}")
    print(f"LSTM-only rows: {lstm_only}")
    if both!=EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LSTM_ROWS} matched rows but found {both}")
    if xgb_only!=EXPECTED_XGB_ROWS-EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Unexpected XGBoost-only row count: {xgb_only}")
    if lstm_only!=0:
        raise RuntimeError(f"Unexpected LSTM-only row count: {lstm_only}")
    aligned=merged[merged["_merge"]=="both"].copy()
    label_mismatch=int((aligned["label_xgb"]!=aligned["label_lstm"]).sum())
    print(f"Label mismatches on aligned rows: {label_mismatch}")
    if label_mismatch!=0:
        raise RuntimeError("XGBoost and LSTM labels do not match on aligned rows")
    aligned=aligned[["stay_id","window_id","label_xgb","xgb_probability","lstm_probability"]].rename(columns={"label_xgb":"label"})
    if aligned["stay_id"].nunique()!=EXPECTED_STAYS:
        raise RuntimeError(f"Unexpected aligned stay count: {aligned['stay_id'].nunique()}")
    return aligned
def compare_with_persisted_alignment(aligned,existing):
    columns=["stay_id","window_id","label","xgb_probability","lstm_probability"]
    existing_sorted=existing[columns].sort_values(["stay_id","window_id"]).reset_index(drop=True)
    aligned_sorted=aligned[columns].sort_values(["stay_id","window_id"]).reset_index(drop=True)
    if aligned_sorted.equals(existing_sorted):
        print("Recomputed alignment matches 01's persisted output exactly: PASS")
        return
    if not np.array_equal(
        aligned_sorted[["stay_id","window_id","label"]].to_numpy(),
        existing_sorted[["stay_id","window_id","label"]].to_numpy()
    ):
        raise RuntimeError("Recomputed alignment keys or labels disagree with 01's persisted aligned file")
    if not np.allclose(
        aligned_sorted[["xgb_probability","lstm_probability"]].to_numpy(),
        existing_sorted[["xgb_probability","lstm_probability"]].to_numpy(),
        rtol=0,
        atol=1e-7
    ):
        raise RuntimeError("Recomputed alignment probabilities disagree with 01's persisted aligned file")
    print("Recomputed alignment matches 01's persisted output within tolerance: PASS")
def calculate_metrics(xgb,aligned):
    full_y=xgb["label"].to_numpy(dtype=np.int8)
    full_xgb_prob=xgb["xgb_probability"].to_numpy(dtype=np.float64)
    full_roc=roc_auc_score(full_y,full_xgb_prob)
    full_pr=average_precision_score(full_y,full_xgb_prob)
    full_prevalence=float(full_y.mean())
    y=aligned["label"].to_numpy(dtype=np.int8)
    xgb_prob=aligned["xgb_probability"].to_numpy(dtype=np.float64)
    lstm_prob=aligned["lstm_probability"].to_numpy(dtype=np.float64)
    aligned_prevalence=float(y.mean())
    aligned_xgb_roc=roc_auc_score(y,xgb_prob)
    aligned_xgb_pr=average_precision_score(y,xgb_prob)
    aligned_lstm_roc=roc_auc_score(y,lstm_prob)
    aligned_lstm_pr=average_precision_score(y,lstm_prob)
    correlation=float(np.corrcoef(xgb_prob,lstm_prob)[0,1])
    print(f"Full-population XGBoost ROC-AUC: {full_roc:.6f}")
    print(f"Full-population XGBoost PR-AUC: {full_pr:.6f}")
    print(f"Full-population positive prevalence: {full_prevalence:.6f}")
    print(f"Aligned-population XGBoost ROC-AUC: {aligned_xgb_roc:.6f}")
    print(f"Aligned-population XGBoost PR-AUC: {aligned_xgb_pr:.6f}")
    print(f"Aligned-population LSTM ROC-AUC: {aligned_lstm_roc:.6f}")
    print(f"Aligned-population LSTM PR-AUC: {aligned_lstm_pr:.6f}")
    print(f"Aligned positive prevalence: {aligned_prevalence:.6f}")
    print(f"XGBoost-LSTM probability correlation: {correlation:.6f}")
    if abs(full_roc-0.825646)>1e-5:
        raise RuntimeError(f"Full XGBoost ROC-AUC does not match saved reference: {full_roc:.6f}")
    if abs(full_pr-0.531311)>1e-5:
        raise RuntimeError(f"Full XGBoost PR-AUC does not match saved reference: {full_pr:.6f}")
    if abs(aligned_prevalence-full_prevalence)>0.01:
        print("NOTE: aligned-population prevalence differs from full-population by >1pp")
    return {
        "full_xgb_roc_auc":float(full_roc),
        "full_xgb_pr_auc":float(full_pr),
        "full_positive_prevalence":full_prevalence,
        "aligned_xgb_roc_auc":float(aligned_xgb_roc),
        "aligned_xgb_pr_auc":float(aligned_xgb_pr),
        "aligned_lstm_roc_auc":float(aligned_lstm_roc),
        "aligned_lstm_pr_auc":float(aligned_lstm_pr),
        "aligned_positive_prevalence":aligned_prevalence,
        "prediction_correlation":correlation
    }
def prediction_distribution(aligned):
    summary=aligned[["xgb_probability","lstm_probability"]].describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
    print("Prediction distribution:")
    print(summary.to_string())
    return summary
def save_audit(xgb,lstm,aligned,metrics,summary):
    with open(AUDIT_FILE,"w") as f:
        f.write(f"Observed XGBoost rows: {len(xgb)}\n")
        f.write(f"Observed LSTM rows: {len(lstm)}\n")
        f.write(f"Observed persisted aligned rows: {len(aligned)}\n")
        f.write(f"Observed XGBoost stays: {xgb['stay_id'].nunique()}\n")
        f.write(f"Observed LSTM stays: {lstm['stay_id'].nunique()}\n")
        f.write(f"Observed aligned stays: {aligned['stay_id'].nunique()}\n")
        f.write(f"Matched rows: {len(aligned)}\n")
        f.write(f"Full XGBoost ROC-AUC: {metrics['full_xgb_roc_auc']:.10f}\n")
        f.write(f"Full XGBoost PR-AUC: {metrics['full_xgb_pr_auc']:.10f}\n")
        f.write(f"Full positive prevalence: {metrics['full_positive_prevalence']:.10f}\n")
        f.write(f"Aligned XGBoost ROC-AUC: {metrics['aligned_xgb_roc_auc']:.10f}\n")
        f.write(f"Aligned XGBoost PR-AUC: {metrics['aligned_xgb_pr_auc']:.10f}\n")
        f.write(f"Aligned LSTM ROC-AUC: {metrics['aligned_lstm_roc_auc']:.10f}\n")
        f.write(f"Aligned LSTM PR-AUC: {metrics['aligned_lstm_pr_auc']:.10f}\n")
        f.write(f"Aligned positive prevalence: {metrics['aligned_positive_prevalence']:.10f}\n")
        f.write(f"XGBoost-LSTM probability correlation: {metrics['prediction_correlation']:.10f}\n")
        f.write("\nPrediction distribution:\n")
        f.write(summary.to_string())
        f.write("\n")
    print(f"Audit saved: {AUDIT_FILE}")
def main():
    xgb,lstm,existing=load_predictions()
    check_schema(xgb,lstm,existing)
    check_row_counts(xgb,lstm,existing)
    check_duplicates(xgb,lstm,existing)
    check_stays(xgb,lstm,existing)
    check_labels(xgb,lstm,existing)
    check_probabilities(xgb,lstm,existing)
    aligned=recompute_alignment(xgb,lstm)
    compare_with_persisted_alignment(aligned,existing)
    metrics=calculate_metrics(xgb,aligned)
    summary=prediction_distribution(aligned)
    save_audit(xgb,lstm,aligned,metrics,summary)
    print("Alignment audit completed successfully")
if __name__=="__main__":
    main()
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score, accuracy_score, roc_auc_score, average_precision_score, brier_score_loss, log_loss
BASE_DIR="/home/mahith/BDA_PROJECT/Model/ensemble"
RESULTS_DIR=os.path.join(BASE_DIR,"results")
INPUT_FILE=os.path.join(RESULTS_DIR,"calibrated_validation_predictions.parquet")
WEIGHT_FILE=os.path.join(RESULTS_DIR,"selected_ensemble_weight.txt")
OUTPUT_PREDICTIONS=os.path.join(RESULTS_DIR,"threshold_selection_predictions.parquet")
OUTPUT_THRESHOLD=os.path.join(RESULTS_DIR,"locked_ensemble_threshold.json")
OUTPUT_RESULTS=os.path.join(RESULTS_DIR,"ensemble_threshold_results.txt")
OUTPUT_SEARCH=os.path.join(RESULTS_DIR,"ensemble_threshold_search.csv")
OUTPUT_COMPARISON=os.path.join(RESULTS_DIR,"threshold_criteria_comparison.csv")
TARGET_SENSITIVITY=0.85
THRESHOLD_MIN=0.01
THRESHOLD_MAX=0.99
THRESHOLD_STEP=0.001
os.makedirs(RESULTS_DIR,exist_ok=True)
df=pd.read_parquet(INPUT_FILE)
required={"stay_id","window_id","label","xgb_calibrated_probability","lstm_calibrated_probability","split"}
missing=required-set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")
if df.empty:
    raise RuntimeError("Calibration dataset is empty")
if df[["stay_id","window_id"]].duplicated().any():
    raise RuntimeError("Duplicate stay_id/window_id keys found")
expected_splits={"calibration_fit","ensemble_tuning","calibration_evaluation"}
if not set(df["split"].dropna().unique()).issubset(expected_splits):
    raise RuntimeError("Unexpected split values found")
tuning=df[df["split"]=="ensemble_tuning"].copy()
if tuning.empty:
    raise RuntimeError("Ensemble-tuning dataset is empty")
for column in ["label","xgb_calibrated_probability","lstm_calibrated_probability"]:
    values=tuning[column].to_numpy()
    if not np.isfinite(values).all():
        raise RuntimeError(f"{column} contains NaN or infinite values")
if not tuning["label"].isin([0,1]).all():
    raise RuntimeError("Labels must contain only 0 and 1")
for column in ["xgb_calibrated_probability","lstm_calibrated_probability"]:
    if ((tuning[column]<0)|(tuning[column]>1)).any():
        raise RuntimeError(f"{column} contains values outside [0,1]")
weight_map={}
with open(WEIGHT_FILE,"r") as f:
    for line in f:
        line=line.strip()
        if not line or ":" not in line:
            continue
        key,_,value=line.partition(":")
        key=key.strip().lower()
        if key=="xgb weight":
            weight_map["xgb_weight"]=float(value.strip())
        elif key=="lstm weight":
            weight_map["lstm_weight"]=float(value.strip())
required_weight_keys={"xgb_weight","lstm_weight"}
missing_weight_keys=required_weight_keys-set(weight_map)
if missing_weight_keys:
    raise RuntimeError(f"Missing required weight keys: {sorted(missing_weight_keys)}")
xgb_weight=weight_map["xgb_weight"]
lstm_weight=weight_map["lstm_weight"]
if not np.isfinite(xgb_weight) or not np.isfinite(lstm_weight):
    raise RuntimeError("Ensemble weights contain NaN or infinite values")
if xgb_weight<0 or lstm_weight<0:
    raise RuntimeError("Ensemble weights cannot be negative")
if not np.isclose(xgb_weight+lstm_weight,1.0,atol=1e-12):
    raise RuntimeError("Ensemble weights do not sum to 1")
tuning["ensemble_probability"]=xgb_weight*tuning["xgb_calibrated_probability"]+lstm_weight*tuning["lstm_calibrated_probability"]
if not np.isfinite(tuning["ensemble_probability"]).all():
    raise RuntimeError("Ensemble probabilities contain NaN or infinite values")
if ((tuning["ensemble_probability"]<0)|(tuning["ensemble_probability"]>1)).any():
    raise RuntimeError("Ensemble probabilities outside [0,1]")
y=tuning["label"].astype(int).to_numpy()
p=tuning["ensemble_probability"].to_numpy()
thresholds=np.round(np.arange(THRESHOLD_MIN,THRESHOLD_MAX+THRESHOLD_STEP/2,THRESHOLD_STEP),3)
if len(thresholds)!=len(np.unique(thresholds)):
    raise RuntimeError("Threshold values are not unique")
rows=[]
for threshold in thresholds:
    prediction=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
    sensitivity=tp/(tp+fn) if (tp+fn)>0 else np.nan
    specificity=tn/(tn+fp) if (tn+fp)>0 else np.nan
    precision=precision_score(y,prediction,zero_division=0)
    npv=tn/(tn+fn) if (tn+fn)>0 else np.nan
    f1=f1_score(y,prediction,zero_division=0)
    f2=fbeta_score(y,prediction,beta=2,zero_division=0)
    accuracy=accuracy_score(y,prediction)
    alert_rate=prediction.mean()
    youden_j=sensitivity+specificity-1
    rows.append({
        "threshold":threshold,
        "sensitivity":sensitivity,
        "specificity":specificity,
        "precision":precision,
        "ppv":precision,
        "npv":npv,
        "f1":f1,
        "f2":f2,
        "youden_j":youden_j,
        "accuracy":accuracy,
        "alert_rate":alert_rate,
        "tn":tn,
        "fp":fp,
        "fn":fn,
        "tp":tp
    })
search=pd.DataFrame(rows)
if search.empty:
    raise RuntimeError("Threshold search produced no results")
if search["threshold"].duplicated().any():
    raise RuntimeError("Threshold search contains duplicate threshold values")
if not np.isfinite(search[["sensitivity","specificity"]].to_numpy()).all():
    raise RuntimeError("Threshold search contains NaN or infinite sensitivity/specificity values")
search_sorted=search.sort_values("threshold").reset_index(drop=True)
sensitivity_diffs=search_sorted["sensitivity"].diff().dropna()
specificity_diffs=search_sorted["specificity"].diff().dropna()
if (sensitivity_diffs>1e-12).any():
    raise RuntimeError("Sensitivity is not monotonically non-increasing with threshold")
if (specificity_diffs<-1e-12).any():
    raise RuntimeError("Specificity is not monotonically non-decreasing with threshold")
youden_row=search.sort_values(
    ["youden_j","threshold"],
    ascending=[False,True]
).iloc[0]
f1_row=search.sort_values(
    ["f1","threshold"],
    ascending=[False,True]
).iloc[0]
f2_row=search.sort_values(
    ["f2","threshold"],
    ascending=[False,True]
).iloc[0]
sensitivity_candidates=search[search["sensitivity"]>=TARGET_SENSITIVITY].copy()
if sensitivity_candidates.empty:
    raise RuntimeError(f"No threshold achieved sensitivity target of {TARGET_SENSITIVITY:.2f}")
sensitivity_candidates["sensitivity_distance"]=(sensitivity_candidates["sensitivity"]-TARGET_SENSITIVITY).abs()
fixed_sensitivity_row=sensitivity_candidates.sort_values(
    ["specificity","sensitivity_distance","threshold"],
    ascending=[False,True,False]
).iloc[0]
criteria_rows=[
    {
        "criterion":"Youden_J",
        "threshold":youden_row["threshold"],
        "sensitivity":youden_row["sensitivity"],
        "specificity":youden_row["specificity"],
        "precision":youden_row["precision"],
        "npv":youden_row["npv"],
        "f1":youden_row["f1"],
        "f2":youden_row["f2"],
        "youden_j":youden_row["youden_j"],
        "accuracy":youden_row["accuracy"],
        "alert_rate":youden_row["alert_rate"],
        "tn":youden_row["tn"],
        "fp":youden_row["fp"],
        "fn":youden_row["fn"],
        "tp":youden_row["tp"]
    },
    {
        "criterion":"Max_F1",
        "threshold":f1_row["threshold"],
        "sensitivity":f1_row["sensitivity"],
        "specificity":f1_row["specificity"],
        "precision":f1_row["precision"],
        "npv":f1_row["npv"],
        "f1":f1_row["f1"],
        "f2":f1_row["f2"],
        "youden_j":f1_row["youden_j"],
        "accuracy":f1_row["accuracy"],
        "alert_rate":f1_row["alert_rate"],
        "tn":f1_row["tn"],
        "fp":f1_row["fp"],
        "fn":f1_row["fn"],
        "tp":f1_row["tp"]
    },
    {
        "criterion":"Max_F2",
        "threshold":f2_row["threshold"],
        "sensitivity":f2_row["sensitivity"],
        "specificity":f2_row["specificity"],
        "precision":f2_row["precision"],
        "npv":f2_row["npv"],
        "f1":f2_row["f1"],
        "f2":f2_row["f2"],
        "youden_j":f2_row["youden_j"],
        "accuracy":f2_row["accuracy"],
        "alert_rate":f2_row["alert_rate"],
        "tn":f2_row["tn"],
        "fp":f2_row["fp"],
        "fn":f2_row["fn"],
        "tp":f2_row["tp"]
    },
    {
        "criterion":"Fixed_Sensitivity_85",
        "threshold":fixed_sensitivity_row["threshold"],
        "sensitivity":fixed_sensitivity_row["sensitivity"],
        "specificity":fixed_sensitivity_row["specificity"],
        "precision":fixed_sensitivity_row["precision"],
        "npv":fixed_sensitivity_row["npv"],
        "f1":fixed_sensitivity_row["f1"],
        "f2":fixed_sensitivity_row["f2"],
        "youden_j":fixed_sensitivity_row["youden_j"],
        "accuracy":fixed_sensitivity_row["accuracy"],
        "alert_rate":fixed_sensitivity_row["alert_rate"],
        "tn":fixed_sensitivity_row["tn"],
        "fp":fixed_sensitivity_row["fp"],
        "fn":fixed_sensitivity_row["fn"],
        "tp":fixed_sensitivity_row["tp"]
    },
]
criteria_comparison=pd.DataFrame(criteria_rows)
locked_threshold=float(fixed_sensitivity_row["threshold"])
locked_prediction=(p>=locked_threshold).astype(int)
tn,fp,fn,tp=confusion_matrix(y,locked_prediction,labels=[0,1]).ravel()
sensitivity=recall_score(y,locked_prediction,zero_division=0)
specificity=tn/(tn+fp) if (tn+fp)>0 else np.nan
precision=precision_score(y,locked_prediction,zero_division=0)
npv=tn/(tn+fn) if (tn+fn)>0 else np.nan
f1=f1_score(y,locked_prediction,zero_division=0)
f2=fbeta_score(y,locked_prediction,beta=2,zero_division=0)
accuracy=accuracy_score(y,locked_prediction)
alert_rate=locked_prediction.mean()
youden_j=sensitivity+specificity-1
roc_auc=roc_auc_score(y,p)
pr_auc=average_precision_score(y,p)
brier=brier_score_loss(y,p)
logloss=log_loss(y,p)
tuning["ensemble_prediction"]=locked_prediction
tuning.to_parquet(OUTPUT_PREDICTIONS,index=False)
search.to_csv(OUTPUT_SEARCH,index=False)
criteria_comparison.to_csv(OUTPUT_COMPARISON,index=False)
threshold_info={
    "threshold":locked_threshold,
    "target_sensitivity":TARGET_SENSITIVITY,
    "primary_selection_criterion":"fixed_sensitivity_85_percent",
    "selection_rule":"Among thresholds with sensitivity >= 0.85, maximize specificity; tie-break by sensitivity closest to 0.85, then higher threshold.",
    "alternative_criteria_evaluated":["Youden_J","Max_F1","Max_F2","Fixed_Sensitivity_85"],
    "xgb_weight":xgb_weight,
    "lstm_weight":lstm_weight,
    "selection_split":"ensemble_tuning",
    "calibration_evaluation_used":False,
    "test_used":False,
    "rows":len(tuning),
    "stays":tuning["stay_id"].nunique(),
    "sensitivity":float(sensitivity),
    "specificity":float(specificity),
    "precision":float(precision),
    "ppv":float(precision),
    "npv":float(npv),
    "f1":float(f1),
    "f2":float(f2),
    "youden_j":float(youden_j),
    "accuracy":float(accuracy),
    "alert_rate":float(alert_rate),
    "roc_auc":float(roc_auc),
    "pr_auc":float(pr_auc),
    "brier":float(brier),
    "log_loss":float(logloss),
    "tn":int(tn),
    "fp":int(fp),
    "fn":int(fn),
    "tp":int(tp)
}
with open(OUTPUT_THRESHOLD,"w") as f:
    json.dump(threshold_info,f,indent=2)
report=f"""
Ensemble threshold selection
Selection split: ensemble_tuning
Calibration-evaluation used: False
Test used: False
Ensemble weights:
XGBoost weight: {xgb_weight:.6f}
LSTM weight: {lstm_weight:.6f}
Primary threshold criterion:
Fixed sensitivity target >= {TARGET_SENSITIVITY:.2f}
Among qualifying thresholds, maximize specificity.
Additional threshold criteria evaluated:
Youden's J
Maximum F1
Maximum F2
Fixed sensitivity at 85%
Tuning rows: {len(tuning)}
Tuning stays: {tuning["stay_id"].nunique()}
Positive prevalence: {y.mean():.6f}
Threshold comparison:
{criteria_comparison.to_string(index=False)}
Primary locked threshold: {locked_threshold:.6f}
Locked-threshold metrics:
Sensitivity: {sensitivity:.6f}
Specificity: {specificity:.6f}
Precision / PPV: {precision:.6f}
NPV: {npv:.6f}
F1: {f1:.6f}
F2: {f2:.6f}
Youden's J: {youden_j:.6f}
Accuracy: {accuracy:.6f}
Alert rate: {alert_rate:.6f}
Probability metrics:
ROC-AUC: {roc_auc:.6f}
PR-AUC: {pr_auc:.6f}
Brier score: {brier:.6f}
Log-loss: {logloss:.6f}
Locked-threshold confusion matrix:
TN: {tn}
FP: {fp}
FN: {fn}
TP: {tp}
Threshold candidates: {len(thresholds)}
Thresholds meeting sensitivity target: {len(sensitivity_candidates)}
Locked threshold saved: {OUTPUT_THRESHOLD}
Threshold search saved: {OUTPUT_SEARCH}
Criteria comparison saved: {OUTPUT_COMPARISON}
Threshold-selection predictions saved: {OUTPUT_PREDICTIONS}
"""
with open(OUTPUT_RESULTS,"w") as f:
    f.write(report.strip()+"\n")
print(f"Loaded validation rows: {len(df)}")
print(f"Ensemble-tuning rows: {len(tuning)}")
print(f"Ensemble-tuning stays: {tuning['stay_id'].nunique()}")
print(f"Locked XGBoost weight: {xgb_weight:.2f}")
print(f"Locked LSTM weight: {lstm_weight:.2f}")
print(f"Primary target sensitivity: {TARGET_SENSITIVITY:.2f}")
print(f"Youden's J threshold: {youden_row['threshold']:.3f}")
print(f"Max F1 threshold: {f1_row['threshold']:.3f}")
print(f"Max F2 threshold: {f2_row['threshold']:.3f}")
print(f"Fixed 85% sensitivity threshold: {fixed_sensitivity_row['threshold']:.3f}")
print(f"Locked ensemble threshold: {locked_threshold:.3f}")
print(f"Locked sensitivity: {sensitivity:.6f}")
print(f"Locked specificity: {specificity:.6f}")
print(f"Locked precision / PPV: {precision:.6f}")
print(f"Locked NPV: {npv:.6f}")
print(f"Locked F1: {f1:.6f}")
print(f"Locked F2: {f2:.6f}")
print(f"Locked Youden's J: {youden_j:.6f}")
print(f"Locked accuracy: {accuracy:.6f}")
print(f"Locked alert rate: {alert_rate:.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")
print(f"PR-AUC: {pr_auc:.6f}")
print(f"Brier score: {brier:.6f}")
print(f"Log-loss: {logloss:.6f}")
print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Locked threshold saved: {OUTPUT_THRESHOLD}")
print(f"Threshold search saved: {OUTPUT_SEARCH}")
print(f"Threshold comparison saved: {OUTPUT_COMPARISON}")
print(f"Threshold-selection predictions saved: {OUTPUT_PREDICTIONS}")
print("Ensemble threshold analysis completed successfully")
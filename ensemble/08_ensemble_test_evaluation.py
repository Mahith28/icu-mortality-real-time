import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score, accuracy_score
BASE_DIR="/home/mahith/BDA_PROJECT/Model/ensemble"
RESULTS_DIR=os.path.join(BASE_DIR,"results")
XGB_MODEL="/home/mahith/BDA_PROJECT/Model/models/icu_mortality_xgboost_clean.json"
XGB_TEST="/home/mahith/BDA_PROJECT/Model/xgb_clean/test_data/test_features"
LSTM_MODEL="/home/mahith/BDA_PROJECT/Model/lstm_clean/results/lstm_adaptive_best.pt"
LSTM_NORM="/home/mahith/BDA_PROJECT/Model/lstm_clean/results/lstm_normalization_stats.json"
LSTM_TEST="/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences/test_sequences"
CALIBRATION_MODEL_FILE=os.path.join(RESULTS_DIR,"calibration_models.json")
WEIGHT_FILE=os.path.join(RESULTS_DIR,"selected_ensemble_weight.txt")
LOCKED_THRESHOLD_FILE=os.path.join(RESULTS_DIR,"locked_ensemble_threshold.json")
OUTPUT_RAW=os.path.join(RESULTS_DIR,"test_raw_predictions.parquet")
OUTPUT_CALIBRATED=os.path.join(RESULTS_DIR,"test_calibrated_predictions.parquet")
OUTPUT_FINAL=os.path.join(RESULTS_DIR,"final_test_predictions.parquet")
OUTPUT_RESULTS=os.path.join(RESULTS_DIR,"final_test_evaluation.txt")
OUTPUT_JSON=os.path.join(RESULTS_DIR,"final_test_evaluation.json")
OUTPUT_BOOTSTRAP=os.path.join(RESULTS_DIR,"final_test_bootstrap.csv")
EXPECTED_FEATURES=137
TEMPORAL_COUNT=54
STATIC_COUNT=83
LSTM_INPUT_SIZE=108
MAX_SEQ_LEN=120
BATCH_SIZE=1024
BOOTSTRAP_REPLICATES=2000
SEED=42
THRESHOLD_TOLERANCE=1e-9
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
XGB_DEVICE="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(RESULTS_DIR,exist_ok=True)
required_files={XGB_MODEL,XGB_TEST,LSTM_MODEL,LSTM_NORM,LSTM_TEST,CALIBRATION_MODEL_FILE,WEIGHT_FILE,LOCKED_THRESHOLD_FILE}
for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing required file or path: {file_path}")
print(f"PyTorch device: {DEVICE}")
print(f"XGBoost device: {XGB_DEVICE}")
with open(WEIGHT_FILE,"r") as f:
    weight_map={}
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
if not np.isclose(xgb_weight,0.48,atol=1e-12):
    raise RuntimeError(f"Unexpected XGBoost weight: {xgb_weight}")
if not np.isclose(lstm_weight,0.52,atol=1e-12):
    raise RuntimeError(f"Unexpected LSTM weight: {lstm_weight}")
if not np.isclose(xgb_weight+lstm_weight,1.0,atol=1e-12):
    raise RuntimeError("Ensemble weights do not sum to 1")
with open(LOCKED_THRESHOLD_FILE,"r") as f:
    threshold_info=json.load(f)
locked_threshold=float(threshold_info["threshold"])
target_sensitivity=float(threshold_info["target_sensitivity"])
if not np.isclose(locked_threshold,0.108,atol=THRESHOLD_TOLERANCE):
    raise RuntimeError(f"Unexpected locked threshold: {locked_threshold}")
if not np.isclose(target_sensitivity,0.85,atol=1e-12):
    raise RuntimeError(f"Unexpected target sensitivity: {target_sensitivity}")
if threshold_info.get("test_used",False):
    raise RuntimeError("Locked threshold indicates that test data were used")
print(f"Locked XGBoost weight: {xgb_weight:.2f}")
print(f"Locked LSTM weight: {lstm_weight:.2f}")
print(f"Locked threshold: {locked_threshold:.3f}")
with open(CALIBRATION_MODEL_FILE,"r") as f:
    calibration_models=json.load(f)
expected_method="Platt scaling using logistic regression on logit-transformed probabilities"
if calibration_models.get("method")!=expected_method:
    raise RuntimeError("Unexpected calibration method")
xgb_intercept=float(calibration_models["xgb_intercept"])
xgb_coefficient=float(calibration_models["xgb_coefficient"])
lstm_intercept=float(calibration_models["lstm_intercept"])
lstm_coefficient=float(calibration_models["lstm_coefficient"])
print(f"XGBoost calibration: intercept={xgb_intercept:.12f}, coefficient={xgb_coefficient:.12f}")
print(f"LSTM calibration: intercept={lstm_intercept:.12f}, coefficient={lstm_coefficient:.12f}")
def vector_struct_to_dense(v,expected_len):
    if hasattr(v,"as_py"):
        v=v.as_py()
    if isinstance(v,dict):
        vector_type=v.get("type")
        vector_size=v.get("size")
        if vector_size!=expected_len:
            raise RuntimeError(f"Expected {expected_len} values but found {vector_size}")
        if vector_type==0:
            dense=np.zeros(expected_len,dtype=np.float32)
            indices=v.get("indices")
            values=v.get("values")
            if indices is not None and values is not None:
                dense[np.asarray(indices,dtype=np.int32)]=np.asarray(values,dtype=np.float32)
            return dense
        if vector_type==1:
            values=v.get("values")
            if values is None:
                return np.zeros(expected_len,dtype=np.float32)
            result=np.asarray(values,dtype=np.float32)
            if len(result)!=expected_len:
                raise RuntimeError(f"Expected {expected_len} values but found {len(result)}")
            return result
    if isinstance(v,(list,tuple,np.ndarray)):
        result=np.asarray(v,dtype=np.float32).reshape(-1)
        if len(result)!=expected_len:
            raise RuntimeError(f"Expected {expected_len} values but found {len(result)}")
        return result
    raise TypeError(f"Unexpected vector type: {type(v)}")
xgb_model=xgb.Booster()
xgb_model.load_model(XGB_MODEL)
best_iteration=getattr(xgb_model,"best_iteration",None)
if best_iteration is None or best_iteration<0:
    raise RuntimeError("booster.best_iteration is missing or invalid")
prediction_iteration_range=(0,best_iteration+1)
print(f"XGBoost model features: {EXPECTED_FEATURES}")
print(f"XGBoost best iteration: {best_iteration}")
print(f"XGBoost iteration range: {prediction_iteration_range}")
xgb_dataset=pq.ParquetDataset(XGB_TEST)
xgb_fragments=xgb_dataset.fragments
xgb_count=0
xgb_prediction_chunks=[]
for fragment in xgb_fragments:
    parquet_file=pq.ParquetFile(fragment.path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE,columns=["stay_id","window_id","features","label"]):
        stay_values=batch.column("stay_id").to_pylist()
        window_values=batch.column("window_id").to_pylist()
        feature_values=batch.column("features").to_pylist()
        label_values=batch.column("label").to_pylist()
        features=np.stack([vector_struct_to_dense(v,EXPECTED_FEATURES) for v in feature_values]).astype(np.float32)
        if np.isinf(features).any():
            bad_indices=np.argwhere(np.isinf(features))
            raise RuntimeError(f"Invalid infinite XGBoost feature values at batch indices: {bad_indices[:20].tolist()}")
        probabilities=xgb_model.predict(xgb.DMatrix(features),iteration_range=prediction_iteration_range)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Invalid XGBoost probabilities")
        xgb_prediction_chunks.append(pd.DataFrame({
            "stay_id":np.asarray(stay_values,dtype=np.int64),
            "window_id":np.asarray(window_values,dtype=np.int64),
            "label":np.asarray(label_values,dtype=np.int8),
            "xgb_probability":np.asarray(probabilities,dtype=np.float32)
        }))
        xgb_count+=len(stay_values)
xgb_predictions=pd.concat(xgb_prediction_chunks,ignore_index=True)
print(f"XGBoost test rows: {xgb_count}")
if len(xgb_predictions)!=xgb_count:
    raise RuntimeError(f"XGBoost prediction count mismatch: {len(xgb_predictions)} vs {xgb_count}")
if xgb_predictions[["stay_id","window_id"]].duplicated().any():
    raise RuntimeError("Duplicate XGBoost test keys found")
if not xgb_predictions["label"].isin([0,1]).all():
    raise RuntimeError("Invalid XGBoost labels")
if not np.isfinite(xgb_predictions["xgb_probability"]).all():
    raise RuntimeError("Invalid XGBoost probabilities")
xgb_roc_auc=roc_auc_score(xgb_predictions["label"],xgb_predictions["xgb_probability"])
xgb_pr_auc=average_precision_score(xgb_predictions["label"],xgb_predictions["xgb_probability"])
print(f"XGBoost test ROC-AUC: {xgb_roc_auc:.6f}")
print(f"XGBoost test PR-AUC: {xgb_pr_auc:.6f}")
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=LSTM_INPUT_SIZE,hidden_size=128,num_layers=2,batch_first=True,dropout=0.30)
        self.static_network=nn.Sequential(nn.Linear(STATIC_COUNT,64),nn.ReLU(),nn.Dropout(0.30))
        self.fusion=nn.Sequential(nn.Linear(128+64,64),nn.ReLU(),nn.Dropout(0.30),nn.Linear(64,1))
    def forward(self,temporal,missing_mask,padding_mask,static):
        lstm_input=torch.cat([temporal,missing_mask],dim=-1)
        lengths=torch.clamp((padding_mask.round()==0).sum(dim=1).long(),min=1,max=MAX_SEQ_LEN)
        packed=nn.utils.rnn.pack_padded_sequence(lstm_input,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.lstm(packed)
        temporal_output=hidden[-1]
        static_output=self.static_network(static)
        fused=torch.cat([temporal_output,static_output],dim=1)
        return self.fusion(fused).squeeze(1)
with open(LSTM_NORM,"r") as f:
    normalization=json.load(f)
temporal_mean=np.asarray(normalization["temporal_mean"],dtype=np.float32)
temporal_std=np.asarray(normalization["temporal_std"],dtype=np.float32)
static_mean=np.asarray(normalization["static_mean"],dtype=np.float32)
static_std=np.asarray(normalization["static_std"],dtype=np.float32)
if len(temporal_mean)!=TEMPORAL_COUNT or len(temporal_std)!=TEMPORAL_COUNT:
    raise RuntimeError("Invalid temporal normalization dimensions")
if len(static_mean)!=STATIC_COUNT or len(static_std)!=STATIC_COUNT:
    raise RuntimeError("Invalid static normalization dimensions")
temporal_std=np.where(temporal_std==0,1.0,temporal_std)
static_std=np.where(static_std==0,1.0,static_std)
lstm_model=LSTMModel()
checkpoint=torch.load(LSTM_MODEL,map_location=DEVICE,weights_only=False)
if isinstance(checkpoint,dict) and "model_state_dict" in checkpoint:
    lstm_model.load_state_dict(checkpoint["model_state_dict"])
else:
    lstm_model.load_state_dict(checkpoint)
lstm_model=lstm_model.to(DEVICE)
lstm_model.eval()
print(f"LSTM model loaded on {DEVICE}")
def flatten_temporal(value):
    if hasattr(value,"as_py"):
        value=value.as_py()
    array=np.asarray(value,dtype=np.float32)
    if array.size!=MAX_SEQ_LEN*TEMPORAL_COUNT:
        raise RuntimeError(f"Expected {MAX_SEQ_LEN*TEMPORAL_COUNT} temporal values but found {array.size}")
    return array.reshape(MAX_SEQ_LEN,TEMPORAL_COUNT)
def flatten_missing(value):
    if hasattr(value,"as_py"):
        value=value.as_py()
    array=np.asarray(value,dtype=np.float32)
    if array.size!=MAX_SEQ_LEN*TEMPORAL_COUNT:
        raise RuntimeError(f"Expected {MAX_SEQ_LEN*TEMPORAL_COUNT} missing-mask values but found {array.size}")
    return array.reshape(MAX_SEQ_LEN,TEMPORAL_COUNT)
def flatten_padding(value):
    if hasattr(value,"as_py"):
        value=value.as_py()
    array=np.asarray(value,dtype=np.float32)
    if array.size!=MAX_SEQ_LEN:
        raise RuntimeError(f"Expected {MAX_SEQ_LEN} padding-mask values but found {array.size}")
    return array.reshape(MAX_SEQ_LEN)
def flatten_static(value):
    if hasattr(value,"as_py"):
        value=value.as_py()
    array=np.asarray(value,dtype=np.float32)
    if array.size!=STATIC_COUNT:
        raise RuntimeError(f"Expected {STATIC_COUNT} static values but found {array.size}")
    return array.reshape(STATIC_COUNT)
lstm_dataset=pq.ParquetDataset(LSTM_TEST)
lstm_fragments=lstm_dataset.fragments
lstm_count=0
lstm_prediction_chunks=[]
for fragment in lstm_fragments:
    parquet_file=pq.ParquetFile(fragment.path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE,columns=["stay_id","terminal_window_id","sequence_length","label","temporal","missing_mask","padding_mask","static"]):
        stay_values=batch.column("stay_id").to_pylist()
        window_values=batch.column("terminal_window_id").to_pylist()
        sequence_values=batch.column("sequence_length").to_pylist()
        label_values=batch.column("label").to_pylist()
        temporal_flat=pc.list_flatten(pc.list_flatten(batch.column("temporal"))).to_numpy(zero_copy_only=False).astype(np.float32,copy=False)
        missing_flat=pc.list_flatten(pc.list_flatten(batch.column("missing_mask"))).to_numpy(zero_copy_only=False).astype(np.float32,copy=False)
        padding_flat=pc.list_flatten(batch.column("padding_mask")).to_numpy(zero_copy_only=False).astype(np.float32,copy=False)
        static_flat=pc.list_flatten(batch.column("static")).to_numpy(zero_copy_only=False).astype(np.float32,copy=False)
        temporal_array=temporal_flat.reshape(len(stay_values),MAX_SEQ_LEN,TEMPORAL_COUNT)
        missing_array=missing_flat.reshape(len(stay_values),MAX_SEQ_LEN,TEMPORAL_COUNT)
        padding_array=padding_flat.reshape(len(stay_values),MAX_SEQ_LEN)
        static_array=static_flat.reshape(len(stay_values),STATIC_COUNT)
        sequence_values=np.asarray(batch.column("sequence_length").to_numpy(),dtype=np.int32)
        for i in range(len(stay_values)):
            valid_count=int((padding_array[i].round()==0).sum())
            if valid_count!=int(sequence_values[i]):
                raise RuntimeError(f"Sequence length mismatch for stay {stay_values[i]} window {window_values[i]}")
        if not np.isfinite(missing_array).all():
            raise RuntimeError("Invalid LSTM missing-mask values")
        if not np.isfinite(padding_array).all():
            raise RuntimeError("Invalid LSTM padding-mask values")
        if not np.isfinite(temporal_array).all():
            raise RuntimeError("Invalid LSTM temporal values before normalization")
        if not np.isfinite(static_array).all():
            raise RuntimeError("Invalid LSTM static values before normalization")
        temporal_array=((temporal_array-temporal_mean)/temporal_std).astype(np.float32,copy=False)
        static_array=((static_array-static_mean)/static_std).astype(np.float32,copy=False)
        if not np.isfinite(temporal_array).all():
            raise RuntimeError("Invalid LSTM temporal values after normalization")
        if not np.isfinite(static_array).all():
            raise RuntimeError("Invalid LSTM static values after normalization")
        temporal_tensor=torch.from_numpy(temporal_array).to(DEVICE,non_blocking=True)
        missing_tensor=torch.from_numpy(missing_array).to(DEVICE,non_blocking=True)
        padding_tensor=torch.from_numpy(padding_array).to(DEVICE,non_blocking=True)
        static_tensor=torch.from_numpy(static_array).to(DEVICE,non_blocking=True)
        with torch.inference_mode():
            logits=lstm_model(temporal_tensor,missing_tensor,padding_tensor,static_tensor)
            probabilities=torch.sigmoid(logits).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Invalid LSTM probabilities")
        lstm_prediction_chunks.append(pd.DataFrame({
            "stay_id":np.asarray(stay_values,dtype=np.int64),
            "window_id":np.asarray(window_values,dtype=np.int64),
            "label":np.asarray(label_values,dtype=np.int8),
            "lstm_probability":np.asarray(probabilities,dtype=np.float32)
        }))
        lstm_count+=len(stay_values)
        if lstm_count%100000==0:
            print(f"LSTM rows processed: {lstm_count}",flush=True)
lstm_predictions=pd.concat(lstm_prediction_chunks,ignore_index=True)
print(f"LSTM test rows: {lstm_count}")
if len(lstm_predictions)!=lstm_count:
    raise RuntimeError(f"LSTM prediction count mismatch: {len(lstm_predictions)} vs {lstm_count}")
if lstm_predictions[["stay_id","window_id"]].duplicated().any():
    raise RuntimeError("Duplicate LSTM test keys found")
if not lstm_predictions["label"].isin([0,1]).all():
    raise RuntimeError("Invalid LSTM labels")
if not np.isfinite(lstm_predictions["lstm_probability"]).all():
    raise RuntimeError("Invalid LSTM probabilities")
lstm_roc_auc=roc_auc_score(lstm_predictions["label"],lstm_predictions["lstm_probability"])
lstm_pr_auc=average_precision_score(lstm_predictions["label"],lstm_predictions["lstm_probability"])
print(f"LSTM test ROC-AUC: {lstm_roc_auc:.6f}")
print(f"LSTM test PR-AUC: {lstm_pr_auc:.6f}")
aligned=xgb_predictions.merge(lstm_predictions,on=["stay_id","window_id"],how="outer",suffixes=("_xgb","_lstm"),indicator=True)
xgb_only=int((aligned["_merge"]=="left_only").sum())
lstm_only=int((aligned["_merge"]=="right_only").sum())
print(f"XGBoost-only rows: {xgb_only}")
print(f"LSTM-only rows: {lstm_only}")
if xgb_only!=0:
    raise RuntimeError(f"XGBoost-only test rows found: {xgb_only}")
if lstm_only!=0:
    raise RuntimeError(f"LSTM-only test rows found: {lstm_only}")
aligned=aligned[aligned["_merge"]=="both"].drop(columns=["_merge"]).copy()
if not np.array_equal(aligned["label_xgb"].to_numpy(),aligned["label_lstm"].to_numpy()):
    raise RuntimeError("XGBoost and LSTM labels do not match")
aligned=aligned.rename(columns={"label_xgb":"label"}).drop(columns=["label_lstm"])
if aligned[["stay_id","window_id"]].duplicated().any():
    raise RuntimeError("Duplicate aligned test keys found")
print(f"Aligned test rows: {len(aligned)}")
print(f"Aligned test stays: {aligned['stay_id'].nunique()}")
aligned[["stay_id","window_id","label","xgb_probability","lstm_probability"]].to_parquet(OUTPUT_RAW,index=False)
def apply_platt(intercept,coefficient,probabilities):
    probabilities=np.clip(np.asarray(probabilities,dtype=np.float64),1e-6,1-1e-6)
    logits=np.log(probabilities/(1-probabilities))
    calibrated_logits=intercept+coefficient*logits
    return 1.0/(1.0+np.exp(-calibrated_logits))
aligned["xgb_calibrated_probability"]=apply_platt(xgb_intercept,xgb_coefficient,aligned["xgb_probability"])
aligned["lstm_calibrated_probability"]=apply_platt(lstm_intercept,lstm_coefficient,aligned["lstm_probability"])
if not np.isfinite(aligned[["xgb_calibrated_probability","lstm_calibrated_probability"]].to_numpy()).all():
    raise RuntimeError("Calibrated probabilities contain NaN or infinite values")
if ((aligned["xgb_calibrated_probability"]<0)|(aligned["xgb_calibrated_probability"]>1)).any():
    raise RuntimeError("XGBoost calibrated probabilities outside [0,1]")
if ((aligned["lstm_calibrated_probability"]<0)|(aligned["lstm_calibrated_probability"]>1)).any():
    raise RuntimeError("LSTM calibrated probabilities outside [0,1]")
aligned["ensemble_probability"]=xgb_weight*aligned["xgb_calibrated_probability"]+lstm_weight*aligned["lstm_calibrated_probability"]
if not np.isfinite(aligned["ensemble_probability"]).all():
    raise RuntimeError("Ensemble probabilities contain NaN or infinite values")
if ((aligned["ensemble_probability"]<0)|(aligned["ensemble_probability"]>1)).any():
    raise RuntimeError("Ensemble probabilities outside [0,1]")
aligned.to_parquet(OUTPUT_CALIBRATED,index=False)
y=aligned["label"].astype(int).to_numpy()
p=aligned["ensemble_probability"].to_numpy()
prediction=(p>=locked_threshold).astype(np.int8)
tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
sensitivity=recall_score(y,prediction,zero_division=0)
specificity=tn/(tn+fp) if (tn+fp)>0 else np.nan
precision=precision_score(y,prediction,zero_division=0)
npv=tn/(tn+fn) if (tn+fn)>0 else np.nan
f1=f1_score(y,prediction,zero_division=0)
f2=fbeta_score(y,prediction,beta=2,zero_division=0)
accuracy=accuracy_score(y,prediction)
alert_rate=prediction.mean()
roc_auc=roc_auc_score(y,p)
pr_auc=average_precision_score(y,p)
brier=brier_score_loss(y,p)
logloss=log_loss(y,p)
aligned["ensemble_prediction"]=prediction
aligned[["stay_id","window_id","label","xgb_probability","lstm_probability","xgb_calibrated_probability","lstm_calibrated_probability","ensemble_probability","ensemble_prediction"]].to_parquet(OUTPUT_FINAL,index=False)
print(f"Final test rows: {len(aligned)}")
print(f"Final test stays: {aligned['stay_id'].nunique()}")
print(f"Test positive prevalence: {y.mean():.6f}")
print(f"ROC-AUC: {roc_auc:.6f}")
print(f"PR-AUC: {pr_auc:.6f}")
print(f"Brier score: {brier:.6f}")
print(f"Log-loss: {logloss:.6f}")
print(f"Locked threshold: {locked_threshold:.3f}")
print(f"Sensitivity: {sensitivity:.6f}")
print(f"Specificity: {specificity:.6f}")
print(f"Precision / PPV: {precision:.6f}")
print(f"NPV: {npv:.6f}")
print(f"F1: {f1:.6f}")
print(f"F2: {f2:.6f}")
print(f"Accuracy: {accuracy:.6f}")
print(f"Alert rate: {alert_rate:.6f}")
print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
stay_ids=aligned["stay_id"].unique()
stay_groups=dict(tuple(aligned.groupby("stay_id",sort=False).groups.items()))
rng=np.random.default_rng(SEED)
bootstrap_rows=[]
for replicate in range(BOOTSTRAP_REPLICATES):
    sampled_stays=rng.choice(stay_ids,size=len(stay_ids),replace=True)
    sampled_indices=np.concatenate([stay_groups[stay_id] for stay_id in sampled_stays])
    sample=aligned.iloc[sampled_indices]
    y_boot=sample["label"].to_numpy(dtype=int)
    p_boot=sample["ensemble_probability"].to_numpy(dtype=float)
    if len(np.unique(y_boot))<2:
        continue
    prediction_boot=(p_boot>=locked_threshold).astype(np.int8)
    tn_b,fp_b,fn_b,tp_b=confusion_matrix(y_boot,prediction_boot,labels=[0,1]).ravel()
    sensitivity_b=tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else np.nan
    specificity_b=tn_b/(tn_b+fp_b) if (tn_b+fp_b)>0 else np.nan
    precision_b=tp_b/(tp_b+fp_b) if (tp_b+fp_b)>0 else np.nan
    npv_b=tn_b/(tn_b+fn_b) if (tn_b+fn_b)>0 else np.nan
    bootstrap_rows.append({
        "replicate":replicate,
        "roc_auc":roc_auc_score(y_boot,p_boot),
        "pr_auc":average_precision_score(y_boot,p_boot),
        "brier":brier_score_loss(y_boot,p_boot),
        "sensitivity":sensitivity_b,
        "specificity":specificity_b,
        "precision":precision_b,
        "npv":npv_b,
        "alert_rate":prediction_boot.mean()
    })
bootstrap=pd.DataFrame(bootstrap_rows)
if bootstrap.empty:
    raise RuntimeError("Bootstrap produced no valid replicates")
bootstrap.to_csv(OUTPUT_BOOTSTRAP,index=False)
def confidence_interval(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    return float(np.percentile(values,2.5)),float(np.percentile(values,97.5))
confidence_intervals={
    "roc_auc":confidence_interval(bootstrap["roc_auc"]),
    "pr_auc":confidence_interval(bootstrap["pr_auc"]),
    "brier":confidence_interval(bootstrap["brier"]),
    "sensitivity":confidence_interval(bootstrap["sensitivity"]),
    "specificity":confidence_interval(bootstrap["specificity"]),
    "precision":confidence_interval(bootstrap["precision"]),
    "npv":confidence_interval(bootstrap["npv"]),
    "alert_rate":confidence_interval(bootstrap["alert_rate"])
}
results={
    "test_data_used_for_calibration":False,
    "test_data_used_for_weight_selection":False,
    "test_data_used_for_threshold_selection":False,
    "xgb_weight":float(xgb_weight),
    "lstm_weight":float(lstm_weight),
    "locked_threshold":float(locked_threshold),
    "target_sensitivity":float(target_sensitivity),
    "rows":int(len(aligned)),
    "stays":int(aligned["stay_id"].nunique()),
    "positive_prevalence":float(y.mean()),
    "roc_auc":float(roc_auc),
    "pr_auc":float(pr_auc),
    "brier_score":float(brier),
    "log_loss":float(logloss),
    "sensitivity":float(sensitivity),
    "specificity":float(specificity),
    "precision":float(precision),
    "ppv":float(precision),
    "npv":float(npv),
    "f1":float(f1),
    "f2":float(f2),
    "accuracy":float(accuracy),
    "alert_rate":float(alert_rate),
    "tn":int(tn),
    "fp":int(fp),
    "fn":int(fn),
    "tp":int(tp),
    "bootstrap_replicates":int(len(bootstrap)),
    "bootstrap_unit":"stay_id",
    "confidence_interval_method":"percentile",
    "confidence_intervals":confidence_intervals
}
with open(OUTPUT_JSON,"w") as f:
    json.dump(results,f,indent=2)
with open(OUTPUT_RESULTS,"w") as f:
    f.write("Final MIMIC-IV test evaluation\n")
    f.write(f"XGBoost weight: {xgb_weight:.6f}\n")
    f.write(f"LSTM weight: {lstm_weight:.6f}\n")
    f.write(f"Locked threshold: {locked_threshold:.6f}\n")
    f.write(f"Target sensitivity: {target_sensitivity:.6f}\n")
    f.write(f"Test rows: {len(aligned)}\n")
    f.write(f"Test stays: {aligned['stay_id'].nunique()}\n")
    f.write(f"Positive prevalence: {y.mean():.6f}\n")
    f.write(f"ROC-AUC: {roc_auc:.6f}\n")
    f.write(f"PR-AUC: {pr_auc:.6f}\n")
    f.write(f"Brier score: {brier:.6f}\n")
    f.write(f"Log-loss: {logloss:.6f}\n")
    f.write(f"Sensitivity: {sensitivity:.6f}\n")
    f.write(f"Specificity: {specificity:.6f}\n")
    f.write(f"Precision / PPV: {precision:.6f}\n")
    f.write(f"NPV: {npv:.6f}\n")
    f.write(f"F1: {f1:.6f}\n")
    f.write(f"F2: {f2:.6f}\n")
    f.write(f"Accuracy: {accuracy:.6f}\n")
    f.write(f"Alert rate: {alert_rate:.6f}\n")
    f.write(f"TN: {tn}\n")
    f.write(f"FP: {fp}\n")
    f.write(f"FN: {fn}\n")
    f.write(f"TP: {tp}\n")
    f.write(f"Bootstrap replicates: {len(bootstrap)}\n")
    f.write("Bootstrap unit: stay_id\n")
    f.write("95% confidence intervals:\n")
    for metric,(lower,upper) in confidence_intervals.items():
        f.write(f"{metric}: {lower:.6f} to {upper:.6f}\n")
print(f"Raw test predictions saved: {OUTPUT_RAW}")
print(f"Calibrated test predictions saved: {OUTPUT_CALIBRATED}")
print(f"Final test predictions saved: {OUTPUT_FINAL}")
print(f"Final test results saved: {OUTPUT_RESULTS}")
print(f"Final test JSON saved: {OUTPUT_JSON}")
print(f"Bootstrap results saved: {OUTPUT_BOOTSTRAP}")
print("Test evaluation completed successfully")
import os
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score,average_precision_score
EXPECTED_FEATURES=137
EXPECTED_XGB_ROWS=1610439
EXPECTED_LSTM_ROWS=184926
TEMPORAL_COUNT=54
STATIC_COUNT=83
LSTM_INPUT_SIZE=108
MAX_SEQ_LEN=120
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.30
STATIC_HIDDEN=64
FUSION_HIDDEN=64
LSTM_ARROW_BATCH_SIZE=64
XGB_MODEL="/home/mahith/BDA_PROJECT/Model/models/icu_mortality_xgboost_clean.json"
XGB_VALIDATION="/home/mahith/BDA_PROJECT/Model/xgb_clean/xgb_data/validation_features"
LSTM_VALIDATION="/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences/validation_sequences"
LSTM_MODEL="/home/mahith/BDA_PROJECT/Model/lstm_clean/results/lstm_adaptive_best.pt"
LSTM_NORMALIZATION="/home/mahith/BDA_PROJECT/Model/lstm_clean/results/lstm_normalization_stats.json"
OUTPUT_DIR="/home/mahith/BDA_PROJECT/Model/ensemble/results/raw_diagnostic"
os.makedirs(OUTPUT_DIR,exist_ok=True)
def convert_features(v):
    if hasattr(v,"toArray"):
        result=v.toArray().astype(np.float32)
        if len(result)!=EXPECTED_FEATURES:
            raise RuntimeError(f"Expected {EXPECTED_FEATURES} values but found {len(result)}")
        return result
    if not isinstance(v,dict):
        raise TypeError(f"Unexpected vector type: {type(v)}")
    if "type" not in v or "size" not in v or "values" not in v:
        raise RuntimeError(f"Invalid Spark VectorUDT structure: {v}")
    vector_type=v["type"]
    vector_size=v["size"]
    if vector_size!=EXPECTED_FEATURES:
        raise RuntimeError(f"Expected vector size {EXPECTED_FEATURES} but found {vector_size}")
    values=v["values"]
    if vector_type==0:
        dense=np.zeros(EXPECTED_FEATURES,dtype=np.float32)
        indices=v.get("indices")
        if indices is not None and len(indices)>0:
            dense[np.asarray(indices,dtype=np.int32)]=np.asarray(values,dtype=np.float32)
        return dense
    if vector_type==1:
        dense=np.asarray(values,dtype=np.float32)
        if len(dense)!=EXPECTED_FEATURES:
            raise RuntimeError(f"Expected {EXPECTED_FEATURES} values but found {len(dense)}")
        return dense
    raise RuntimeError(f"Unsupported Spark vector type: {vector_type}")
def validate_xgb_files():
    if not os.path.exists(XGB_VALIDATION):
        raise FileNotFoundError(XGB_VALIDATION)
    dataset=ds.dataset(XGB_VALIDATION,format="parquet")
    required={"stay_id","window_id","features","label"}
    if not required.issubset(set(dataset.schema.names)):
        raise RuntimeError("XGBoost validation dataset is missing required columns")
    total_rows=dataset.count_rows()
    if total_rows!=EXPECTED_XGB_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_XGB_ROWS} XGBoost rows but found {total_rows}")
    print(f"XGBoost validation rows: {total_rows}")
def load_xgb_model():
    if not os.path.exists(XGB_MODEL):
        raise FileNotFoundError(XGB_MODEL)
    model=xgb.Booster()
    model.load_model(XGB_MODEL)
    feature_count=model.num_features()
    if feature_count!=EXPECTED_FEATURES:
        raise RuntimeError(f"Expected XGBoost model to use {EXPECTED_FEATURES} features but found {feature_count}")
    best_iteration=getattr(model,"best_iteration",None)
    if best_iteration is None or best_iteration<0:
        best_iteration=model.num_boosted_rounds()-1
    print(f"XGBoost model features: {feature_count}")
    print(f"XGBoost best iteration: {best_iteration}")
    return model,best_iteration
def generate_xgb_predictions(model,best_iteration):
    dataset=ds.dataset(XGB_VALIDATION,format="parquet")
    frames=[]
    for batch in dataset.to_batches(columns=["stay_id","window_id","features","label"],batch_size=100000):
        pdf=batch.to_pandas()
        if pdf["label"].isna().any():
            raise RuntimeError("XGBoost validation contains NaN labels")
        X=np.vstack(pdf["features"].map(convert_features).to_numpy())
        y=pdf["label"].to_numpy()
        if X.shape[1]!=EXPECTED_FEATURES:
            raise RuntimeError(f"Expected XGBoost input width {EXPECTED_FEATURES} but found {X.shape[1]}")
        if np.isinf(X).any():
            raise RuntimeError("XGBoost features contain infinite values")
        if not np.isfinite(y).all():
            raise RuntimeError("XGBoost labels contain NaN or infinite values")
        y=y.astype(np.int8)
        dmatrix=xgb.DMatrix(X)
        probabilities=model.predict(dmatrix,iteration_range=(0,best_iteration+1))
        if not np.isfinite(probabilities).all():
            raise RuntimeError("XGBoost predictions contain NaN or infinite values")
        if ((probabilities<0)|(probabilities>1)).any():
            raise RuntimeError("XGBoost predictions are outside [0,1]")
        frames.append(pd.DataFrame({"stay_id":pdf["stay_id"].to_numpy(dtype=np.int64),"window_id":pdf["window_id"].to_numpy(dtype=np.int64),"label":y,"xgb_probability":probabilities.astype(np.float32)}))
    result=pd.concat(frames,ignore_index=True)
    if len(result)!=EXPECTED_XGB_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_XGB_ROWS} XGBoost predictions but found {len(result)}")
    xgb_roc_auc=roc_auc_score(result["label"],result["xgb_probability"])
    xgb_pr_auc=average_precision_score(result["label"],result["xgb_probability"])
    print(f"XGBoost validation ROC-AUC: {xgb_roc_auc:.6f}")
    print(f"XGBoost validation PR-AUC: {xgb_pr_auc:.6f}")
    if result.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate XGBoost prediction keys found")
    output=os.path.join(OUTPUT_DIR,"xgb_validation_predictions.parquet")
    result.to_parquet(output,index=False)
    print(f"XGBoost predictions saved: {len(result)}")
    return result
def load_normalization_statistics():
    if not os.path.exists(LSTM_NORMALIZATION):
        raise FileNotFoundError(LSTM_NORMALIZATION)
    with open(LSTM_NORMALIZATION,"r") as f:
        stats=json.load(f)
    temporal_mean=np.asarray(stats["temporal_mean"],dtype=np.float32)
    temporal_std=np.asarray(stats["temporal_std"],dtype=np.float32)
    static_mean=np.asarray(stats["static_mean"],dtype=np.float32)
    static_std=np.asarray(stats["static_std"],dtype=np.float32)
    if len(temporal_mean)!=TEMPORAL_COUNT or len(temporal_std)!=TEMPORAL_COUNT:
        raise RuntimeError("Temporal normalization dimension mismatch")
    if len(static_mean)!=STATIC_COUNT or len(static_std)!=STATIC_COUNT:
        raise RuntimeError("Static normalization dimension mismatch")
    temporal_std=np.where(temporal_std==0,1.0,temporal_std)
    static_std=np.where(static_std==0,1.0,static_std)
    return temporal_mean,temporal_std,static_mean,static_std
class ICU_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=LSTM_INPUT_SIZE,hidden_size=HIDDEN_SIZE,num_layers=NUM_LAYERS,batch_first=True,dropout=DROPOUT)
        self.static_network=nn.Sequential(
            nn.Linear(STATIC_COUNT,STATIC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        self.fusion=nn.Sequential(
            nn.Linear(HIDDEN_SIZE+STATIC_HIDDEN,FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_HIDDEN,1)
        )
    def forward(self,temporal,static,padding_mask):
        lengths=torch.clamp((padding_mask.round()==0).sum(dim=1).long(),min=1,max=MAX_SEQ_LEN)
        packed=nn.utils.rnn.pack_padded_sequence(temporal,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.lstm(packed)
        temporal_output=hidden[-1]
        static_output=self.static_network(static)
        fused=torch.cat([temporal_output,static_output],dim=1)
        return self.fusion(fused).squeeze(1)
def load_lstm_model():
    if not os.path.exists(LSTM_MODEL):
        raise FileNotFoundError(LSTM_MODEL)
    checkpoint=torch.load(LSTM_MODEL,map_location="cpu",weights_only=False)
    model=ICU_LSTM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("LSTM model loaded")
    if "epoch" in checkpoint:
        print(f"LSTM best epoch: {checkpoint['epoch']}")
    if "best_val_roc_auc" in checkpoint:
        print(f"LSTM validation ROC-AUC: {checkpoint['best_val_roc_auc']}")
    if "best_val_pr_auc" in checkpoint:
        print(f"LSTM validation PR-AUC: {checkpoint['best_val_pr_auc']}")
    return model
def validate_lstm_dataset():
    if not os.path.exists(LSTM_VALIDATION):
        raise FileNotFoundError(LSTM_VALIDATION)
    dataset=ds.dataset(LSTM_VALIDATION,format="parquet")
    required={"stay_id","terminal_window_id","sequence_length","temporal","missing_mask","padding_mask","static","label"}
    if not required.issubset(set(dataset.schema.names)):
        raise RuntimeError("LSTM validation dataset is missing required columns")
    row_count=dataset.count_rows()
    if row_count!=EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LSTM_ROWS} LSTM rows but found {row_count}")
    print(f"LSTM validation rows: {row_count}")
def flatten_list_array(array):
    return pc.list_flatten(pc.list_flatten(array))
def generate_lstm_predictions(model):
    dataset=ds.dataset(LSTM_VALIDATION,format="parquet")
    temporal_mean,temporal_std,static_mean,static_std=load_normalization_statistics()
    frames=[]
    columns=["stay_id","terminal_window_id","sequence_length","temporal","missing_mask","padding_mask","static","label"]
    for batch in dataset.to_batches(columns=columns,batch_size=LSTM_ARROW_BATCH_SIZE):
        row_count=batch.num_rows
        stay_id=np.asarray(batch.column("stay_id").to_numpy(zero_copy_only=False),dtype=np.int64)
        terminal_window_id=np.asarray(batch.column("terminal_window_id").to_numpy(zero_copy_only=False),dtype=np.int64)
        sequence_length=np.asarray(batch.column("sequence_length").to_numpy(zero_copy_only=False),dtype=np.int64)
        label_array=batch.column("label")
        if label_array.null_count>0:
            raise RuntimeError("LSTM validation contains null labels")
        label=np.asarray(label_array.to_numpy(zero_copy_only=False),dtype=np.int8)
        temporal_array=batch.column("temporal")
        missing_array=batch.column("missing_mask")
        padding_array=batch.column("padding_mask")
        static_array=batch.column("static")
        temporal_flat=np.asarray(flatten_list_array(temporal_array).to_numpy(zero_copy_only=False),dtype=np.float32)
        missing_flat=np.asarray(flatten_list_array(missing_array).to_numpy(zero_copy_only=False),dtype=np.float32)
        padding_flat=np.asarray(pc.list_flatten(padding_array).to_numpy(zero_copy_only=False),dtype=np.float32)
        static_flat=np.asarray(pc.list_flatten(static_array).to_numpy(zero_copy_only=False),dtype=np.float32)
        temporal=temporal_flat.reshape(row_count,MAX_SEQ_LEN,TEMPORAL_COUNT)
        missing_mask=missing_flat.reshape(row_count,MAX_SEQ_LEN,TEMPORAL_COUNT)
        padding_mask=padding_flat.reshape(row_count,MAX_SEQ_LEN)
        static=static_flat.reshape(row_count,STATIC_COUNT)
        if temporal.shape!=(row_count,MAX_SEQ_LEN,TEMPORAL_COUNT):
            raise RuntimeError(f"Unexpected temporal shape: {temporal.shape}")
        if missing_mask.shape!=(row_count,MAX_SEQ_LEN,TEMPORAL_COUNT):
            raise RuntimeError(f"Unexpected missing mask shape: {missing_mask.shape}")
        if padding_mask.shape!=(row_count,MAX_SEQ_LEN):
            raise RuntimeError(f"Unexpected padding mask shape: {padding_mask.shape}")
        if static.shape!=(row_count,STATIC_COUNT):
            raise RuntimeError(f"Unexpected static shape: {static.shape}")
        if not np.isin(padding_mask,[0,1]).all():
            raise RuntimeError("Padding mask contains values other than 0 and 1")
        valid_lengths=(padding_mask==0).sum(axis=1)
        if not np.array_equal(valid_lengths,sequence_length):
            raise RuntimeError("Sequence length does not match padding mask")
        for i in range(row_count):
            length=sequence_length[i]
            if length<1 or length>MAX_SEQ_LEN:
                raise RuntimeError(f"Invalid sequence length: {length}")
            if not np.all(padding_mask[i,:length]==0):
                raise RuntimeError("Valid timesteps are not at the beginning")
            if length<MAX_SEQ_LEN and not np.all(padding_mask[i,length:]==1):
                raise RuntimeError("Padding timesteps are not at the end")
        if not np.isfinite(temporal).all():
            raise RuntimeError("Invalid LSTM temporal values before normalization")
        if not np.isfinite(static).all():
            raise RuntimeError("Invalid LSTM static values before normalization")
        if not np.isfinite(temporal).all():
            raise RuntimeError("Invalid LSTM temporal values after normalization")
        if not np.isfinite(static).all():
            raise RuntimeError("Invalid LSTM static values after normalization")
        lstm_input=np.concatenate([temporal,missing_mask],axis=2)
        if lstm_input.shape[2]!=LSTM_INPUT_SIZE:
            raise RuntimeError(f"Expected LSTM input width {LSTM_INPUT_SIZE} but found {lstm_input.shape[2]}")
        with torch.no_grad():
            temporal_tensor=torch.from_numpy(lstm_input).float()
            static_tensor=torch.from_numpy(static).float()
            padding_tensor=torch.from_numpy(padding_mask).float()
            logits=model(temporal_tensor,static_tensor,padding_tensor)
            probabilities=torch.sigmoid(logits).cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise RuntimeError("LSTM predictions contain NaN or infinite values")
        if ((probabilities<0)|(probabilities>1)).any():
            raise RuntimeError("LSTM predictions are outside [0,1]")
        frames.append(pd.DataFrame({"stay_id":stay_id,"window_id":terminal_window_id,"label":label,"lstm_probability":probabilities.astype(np.float32)}))
    result=pd.concat(frames,ignore_index=True)
    if len(result)!=EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LSTM_ROWS} LSTM predictions but found {len(result)}")
    lstm_roc_auc=roc_auc_score(result["label"],result["lstm_probability"])
    lstm_pr_auc=average_precision_score(result["label"],result["lstm_probability"])
    print(f"LSTM validation ROC-AUC: {lstm_roc_auc:.6f}")
    print(f"LSTM validation PR-AUC: {lstm_pr_auc:.6f}")
    if result.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate LSTM prediction keys found")
    output=os.path.join(OUTPUT_DIR,"lstm_validation_predictions.parquet")
    result.to_parquet(output,index=False)
    print(f"LSTM predictions saved: {len(result)}")
    return result
def align_predictions(xgb_predictions,lstm_predictions):
    if len(xgb_predictions)!=EXPECTED_XGB_ROWS:
        raise RuntimeError("Unexpected XGBoost prediction count")
    if len(lstm_predictions)!=EXPECTED_LSTM_ROWS:
        raise RuntimeError("Unexpected LSTM prediction count")
    if xgb_predictions.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate XGBoost keys found")
    if lstm_predictions.duplicated(["stay_id","window_id"]).any():
        raise RuntimeError("Duplicate LSTM keys found")
    xgb_stays=set(xgb_predictions["stay_id"].unique())
    lstm_stays=set(lstm_predictions["stay_id"].unique())
    if xgb_stays!=lstm_stays:
        raise RuntimeError("XGBoost and LSTM stay sets do not match")
    merged=xgb_predictions.merge(lstm_predictions,on=["stay_id","window_id"],how="inner",suffixes=("_xgb","_lstm"),validate="one_to_one")
    if len(merged)!=EXPECTED_LSTM_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LSTM_ROWS} aligned rows but found {len(merged)}")
    if not np.array_equal(merged["label_xgb"].to_numpy(),merged["label_lstm"].to_numpy()):
        raise RuntimeError("XGBoost and LSTM labels do not match")
    result=merged[["stay_id","window_id","label_xgb","xgb_probability","lstm_probability"]].rename(columns={"label_xgb":"label"})
    if result["xgb_probability"].isna().any():
        raise RuntimeError("Aligned XGBoost predictions contain NaN values")
    if result["lstm_probability"].isna().any():
        raise RuntimeError("Aligned LSTM predictions contain NaN values")
    output=os.path.join(OUTPUT_DIR,"validation_raw_predictions.parquet")
    result.to_parquet(output,index=False)
    print(f"Aligned validation rows: {len(result)}")
    print(f"Aligned validation stays: {result['stay_id'].nunique()}")
    return result
def main():
    validate_xgb_files()
    xgb_model,best_iteration=load_xgb_model()
    xgb_predictions=generate_xgb_predictions(xgb_model,best_iteration)
    validate_lstm_dataset()
    lstm_model=load_lstm_model()
    lstm_predictions=generate_lstm_predictions(lstm_model)
    if len(lstm_predictions)!=EXPECTED_LSTM_ROWS:
        raise RuntimeError("LSTM prediction count is not 184,926")
    align_predictions(xgb_predictions,lstm_predictions)
    print("Prediction generation completed successfully")
if __name__=="__main__":
    main()
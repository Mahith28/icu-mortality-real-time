import os
import sys
import glob
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from pyspark.sql import SparkSession
MODEL_PATH="/home/mahith/BDA_PROJECT/xgb_clean/checkpoints/icu_mortality_xgboost_clean.json"
TEST_PATH="/home/mahith/BDA_PROJECT/xgb_clean/test_data/test_features"
HDFS_TEST="hdfs://localhost:9000/user/mahith/icu/clean_features/test_features"
OUTPUT_DIR="/home/mahith/BDA_PROJECT/xgb_clean/results"
EXPECTED_FEATURES=137
EXPECTED_BEST_ITERATION=243
EXPECTED_TEST_ROWS=1664239
BATCH_SIZE=10000
ADDITIVITY_TOLERANCE=1e-3
os.makedirs(OUTPUT_DIR,exist_ok=True)
def vector_struct_to_dense(v,expected_len):
    if hasattr(v,"toArray"):
        result=v.toArray().astype(np.float32)
        if len(result)!=expected_len:
            raise RuntimeError(f"Expected {expected_len} values but found {len(result)}")
        return result
    if not isinstance(v,dict):
        raise TypeError(f"Unexpected vector type: {type(v)}")
    vector_type=v["type"]
    vector_size=v["size"]
    if vector_size!=expected_len:
        raise RuntimeError(f"Expected vector size {expected_len} but found {vector_size}")
    if vector_type==0:
        dense=np.zeros(expected_len,dtype=np.float32)
        indices=v["indices"]
        values=v["values"]
        if indices is not None and len(indices)>0:
            dense[np.asarray(indices,dtype=np.int32)]=np.asarray(values,dtype=np.float32)
        return dense
    if vector_type==1:
        values=np.asarray(v["values"],dtype=np.float32)
        if len(values)!=expected_len:
            raise RuntimeError(f"Expected {expected_len} values but found {len(values)}")
        return values
    raise RuntimeError(f"Unknown Spark vector type: {vector_type}")
print("XGBoost :",xgb.__version__)
print("SHAP :",shap.__version__)
print("\nLOADING MODEL")
booster=xgb.Booster()
booster.load_model(MODEL_PATH)
print("Model boosting rounds :",booster.num_boosted_rounds())
print("Model features :",booster.num_features())
print("Best iteration :",booster.best_iteration)
if booster.num_features()!=EXPECTED_FEATURES:
    raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {booster.num_features()}")
if booster.best_iteration!=EXPECTED_BEST_ITERATION:
    raise ValueError(f"Expected best iteration {EXPECTED_BEST_ITERATION}, got {booster.best_iteration}")
truncated_booster=booster[:EXPECTED_BEST_ITERATION+1]
print("Trees used :",truncated_booster.num_boosted_rounds())
print("\nCHECKING LOCAL TEST DATA")
files=sorted(glob.glob(os.path.join(TEST_PATH,"*.parquet")))
if not files:
    print("Local test data not found")
    print("COPYING TEST DATA FROM HDFS")
    if os.path.exists(TEST_PATH):
        shutil.rmtree(TEST_PATH)
    os.makedirs(TEST_PATH,exist_ok=True)
    subprocess.run(
        ["hdfs","dfs","-get",HDFS_TEST+"/*",TEST_PATH],
        check=True
    )
    files=sorted(glob.glob(os.path.join(TEST_PATH,"*.parquet")))
if not files:
    raise FileNotFoundError("No test Parquet files found")
print("Test partitions :",len(files))
print("\nCHECKING TEST ROW COUNT")
total_test_rows=0
for file in files:
    total_test_rows+=len(pd.read_parquet(file,columns=["label"]))
print("Test rows :",f"{total_test_rows:,}")
if total_test_rows!=EXPECTED_TEST_ROWS:
    raise ValueError(f"Expected {EXPECTED_TEST_ROWS} rows, got {total_test_rows}")
print("\nREADING FEATURE METADATA")
spark=SparkSession.builder.master("local[*]").appName("SHAPFeatureMetadata").getOrCreate()
test_df=spark.read.parquet(TEST_PATH)
metadata=test_df.schema["features"].metadata
feature_names=[]
if "ml_attr" not in metadata:
    spark.stop()
    raise ValueError("ml_attr metadata not found in features column")
attrs=metadata["ml_attr"]["attrs"]
for group in attrs.values():
    for item in group:
        feature_names.append((int(item["idx"]),item["name"]))
feature_names=[name for _,name in sorted(feature_names)]
spark.stop()
if len(feature_names)!=EXPECTED_FEATURES:
    raise ValueError(f"Expected {EXPECTED_FEATURES} feature names, got {len(feature_names)}")
print("Feature names :",len(feature_names))
print("\nCREATING SHAP EXPLAINER")
explainer=shap.TreeExplainer(
    truncated_booster,
    model_output="raw"
)
expected_value=float(
    np.asarray(explainer.expected_value).reshape(-1)[0]
)
print("SHAP expected value :",explainer.expected_value)
print("\nSTARTING FULL TEST SHAP")
start_time=time.time()
sum_abs_shap=np.zeros(
    EXPECTED_FEATURES,
    dtype=np.float64
)
total_rows=0
additivity_checked=False
additivity_stats={}
for partition,file in enumerate(files,1):
    print(f"\nTEST PARTITION {partition}/{len(files)}")
    df=pd.read_parquet(file)
    print("Rows :",len(df))
    for start in range(0,len(df),BATCH_SIZE):
        batch=df.iloc[start:start+BATCH_SIZE]
        X=np.vstack([
            vector_struct_to_dense(v,EXPECTED_FEATURES)
            for v in batch["features"]
        ]).astype(np.float32)
        if X.shape!=(len(batch),EXPECTED_FEATURES):
            raise ValueError(f"Unexpected feature shape: {X.shape}")
        if np.isinf(X).any():
            raise ValueError("Infinite feature values detected.")
        shap_values=np.asarray(
            explainer.shap_values(X)
        )
        if shap_values.ndim!=2:
            raise ValueError(f"Unexpected SHAP dimensions: {shap_values.ndim}")
        if shap_values.shape!=(len(batch),EXPECTED_FEATURES):
            raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")
        if not additivity_checked:
            print("\nSHAP ADDITIVITY CHECK")
            shap_sums=expected_value+shap_values.sum(axis=1)
            model_margins=truncated_booster.predict(
                xgb.DMatrix(X),
                output_margin=True
            )
            diffs=np.abs(shap_sums-model_margins)
            mean_diff=float(diffs.mean())
            median_diff=float(np.median(diffs))
            max_diff=float(diffs.max())
            min_diff=float(diffs.min())
            std_diff=float(diffs.std())
            print("Mean diff :",mean_diff)
            print("Median diff :",median_diff)
            print("Max diff :",max_diff)
            print("Min diff :",min_diff)
            print("Std diff :",std_diff)
            print("Tolerance :",ADDITIVITY_TOLERANCE)
            if max_diff>=ADDITIVITY_TOLERANCE:
                raise ValueError(f"SHAP additivity check failed: max difference={max_diff}")
            print("PASS: SHAP additivity within tolerance")
            additivity_stats={
                "mean_difference":mean_diff,
                "median_difference":median_diff,
                "max_difference":max_diff,
                "min_difference":min_diff,
                "std_difference":std_diff,
                "tolerance":ADDITIVITY_TOLERANCE,
                "rows_checked":len(batch)
            }
            additivity_checked=True
        sum_abs_shap+=np.sum(
            np.abs(shap_values),
            axis=0
        )
        total_rows+=len(batch)
        print(f"Processed : {total_rows:,}")
elapsed=time.time()-start_time
print("\nFULL SHAP COMPLETED")
print("Total rows :",f"{total_rows:,}")
print("Runtime :",f"{elapsed/60:.2f} minutes")
if total_rows!=EXPECTED_TEST_ROWS:
    raise ValueError(f"Expected {EXPECTED_TEST_ROWS} rows, got {total_rows}")
if not additivity_checked:
    raise RuntimeError("Additivity check was not performed")
print("\nCALCULATING FEATURE IMPORTANCE")
mean_abs_shap=sum_abs_shap/total_rows
importance=pd.DataFrame({
    "feature_index":range(EXPECTED_FEATURES),
    "feature":feature_names,
    "mean_abs_shap":mean_abs_shap
})
importance=importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)
importance.insert(
    0,
    "rank",
    range(1,EXPECTED_FEATURES+1)
)
print("\nTOP 20 FEATURES")
print(importance.head(20).to_string(index=False))
print("\nSAVING RESULTS")
csv_path=os.path.join(OUTPUT_DIR,"shap_feature_importance.csv")
importance.to_csv(csv_path,index=False)
print("CSV :",csv_path)
top=importance.head(20).sort_values("mean_abs_shap")
plt.figure(figsize=(12,9))
plt.barh(
    top["feature"],
    top["mean_abs_shap"]
)
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("XGBoost SHAP Feature Importance - Full Test Set")
plt.tight_layout()
png_path=os.path.join(OUTPUT_DIR,"shap_feature_importance.png")
plt.savefig(png_path,dpi=300,bbox_inches="tight")
plt.close()
print("PNG :",png_path)
info={
    "model_path":MODEL_PATH,
    "test_path":TEST_PATH,
    "xgboost_version":xgb.__version__,
    "shap_version":shap.__version__,
    "model_boosting_rounds":int(booster.num_boosted_rounds()),
    "best_iteration":int(booster.best_iteration),
    "trees_used":int(EXPECTED_BEST_ITERATION+1),
    "feature_count":EXPECTED_FEATURES,
    "test_rows":int(total_rows),
    "batch_size":BATCH_SIZE,
    "model_output":"raw",
    "expected_value":expected_value,
    "additivity_check":"passed",
    "additivity":additivity_stats,
    "shap_runtime_minutes":float(elapsed/60)
}
json_path=os.path.join(OUTPUT_DIR,"shap_analysis_info.json")
with open(json_path,"w") as f:
    json.dump(info,f,indent=2)
print("JSON :",json_path)
print("\nSTAGE 06 SHAP COMPLETED")
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.functions import vector_to_array
import os
import shutil
BASE = "hdfs://localhost:9000/user/mahith/icu"
TRAIN = f"{BASE}/train"
VALIDATION = f"{BASE}/validation"
TEST = f"{BASE}/test"
CLEAN = f"{BASE}/clean_features"
LOCAL_PIPELINE = "/home/mahith/BDA_PROJECT/checkpoints/preprocessing_model_subject_safe"
spark = SparkSession.builder.appName("SubjectSafeCleanFeatureGeneration").master("local[2]").config("spark.driver.memory", "6g").config("spark.sql.shuffle.partitions", "32").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("REFIT CLEAN PREPROCESSING PIPELINE")
datasets = {"TRAIN": spark.read.parquet(TRAIN), "VALIDATION": spark.read.parquet(VALIDATION), "TEST": spark.read.parquet(TEST)}
for name, df in datasets.items():
    print(f"{name} rows: {df.count():,}")
categorical_cols = ["gender", "race", "insurance", "admission_type", "admission_location", "first_careunit"]
indexed_cols = ["gender_index", "race_index", "insurance_index", "admission_type_index", "admission_location_index", "first_careunit_index"]
ohe_cols = ["gender_ohe", "race_ohe", "insurance_ohe", "admission_type_ohe", "admission_location_ohe", "first_careunit_ohe"]
numeric_cols = [
    "Heart_Rate_mean", "Heart_Rate_min", "Heart_Rate_max", "Heart_Rate_last",
    "SBP_mean", "SBP_min", "SBP_max", "SBP_last",
    "DBP_mean", "DBP_min", "DBP_max", "DBP_last",
    "MAP_mean", "MAP_min", "MAP_max", "MAP_last",
    "Respiratory_Rate_mean", "Respiratory_Rate_min", "Respiratory_Rate_max", "Respiratory_Rate_last",
    "Temperature_mean", "Temperature_min", "Temperature_max", "Temperature_last",
    "SpO2_mean", "SpO2_min", "SpO2_max", "SpO2_last",
    "GCS_Eye_mean", "GCS_Eye_min", "GCS_Eye_max", "GCS_Eye_last",
    "GCS_Verbal_mean", "GCS_Verbal_min", "GCS_Verbal_max", "GCS_Verbal_last",
    "GCS_Motor_mean", "GCS_Motor_min", "GCS_Motor_max", "GCS_Motor_last",
    "Heart_Rate_missing", "SBP_missing", "DBP_missing", "MAP_missing", "Respiratory_Rate_missing",
    "Temperature_missing", "SpO2_missing", "GCS_Eye_missing", "GCS_Verbal_missing", "GCS_Motor_missing",
    "HR_trend", "MAP_trend", "RR_trend", "GCS_Total_last", "anchor_age"
]
assembler_inputs = numeric_cols + ohe_cols
stages = []
for input_col, output_col in zip(categorical_cols, indexed_cols):
    stages.append(StringIndexer(inputCol=input_col, outputCol=output_col, handleInvalid="keep", stringOrderType="frequencyDesc"))
stages.append(OneHotEncoder(inputCols=indexed_cols, outputCols=ohe_cols, dropLast=True, handleInvalid="keep"))
stages.append(VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep"))
pipeline = Pipeline(stages=stages)
print("\nFITTING PIPELINE ON TRAIN ONLY")
pipeline_model = pipeline.fit(datasets["TRAIN"])
print("PIPELINE FIT COMPLETE")
print("\nPIPELINE STAGES")
for i, stage in enumerate(pipeline_model.stages):
    print(f"Stage {i}: {stage.__class__.__name__}")
assembler_model = pipeline_model.stages[-1]
print("\nASSEMBLER INPUT COUNT")
print(f"Assembler inputs: {len(assembler_model.getInputCols())}")
print("\nTRANSFORMING DATASETS")
transformed = {}
for name, df in datasets.items():
    result = pipeline_model.transform(df).select("stay_id", "subject_id", "window_id", "features", F.col("mortality").alias("label")).cache()
    result.count()
    transformed[name] = result
    print(f"{name} transformed")
print("\nVERIFYING FEATURE DIMENSIONS")
dimension_results = {}
for name, df in transformed.items():
    dimensions = df.select(F.size(vector_to_array("features")).alias("dim")).distinct().collect()
    dimensions = sorted([row["dim"] for row in dimensions])
    dimension_results[name] = dimensions
    print(f"{name} feature dimensions: {dimensions}")
print("\nVERIFYING ROW COUNTS")
expected_counts = {"TRAIN": 7_459_304, "VALIDATION": 1_610_439, "TEST": 1_664_239}
row_count_results = {}
for name, df in transformed.items():
    count = df.count()
    row_count_results[name] = count
    print(f"{name}: {count:,}")
print("\nVERIFYING NULLS")
null_results = {}
for name, df in transformed.items():
    null_count = df.filter(F.col("features").isNull() | F.col("label").isNull() | F.col("stay_id").isNull() | F.col("subject_id").isNull() | F.col("window_id").isNull()).count()
    null_results[name] = null_count
    print(f"{name} null rows: {null_count:,}")
print("\nFINAL VALIDATION")
row_counts_ok = all(row_count_results[name] == expected_counts[name] for name in expected_counts)
dimensions_ok = all(dimension_results[name] == [137] for name in dimension_results)
nulls_ok = all(null_results[name] == 0 for name in null_results)
assembler_ok = len(assembler_model.getInputCols()) == 61
pipeline_ok = row_counts_ok and dimensions_ok and nulls_ok and assembler_ok
print(f"Row counts preserved: {'PASS' if row_counts_ok else 'FAIL'}")
print(f"137 features in all splits: {'PASS' if dimensions_ok else 'FAIL'}")
print(f"No null key/feature/label rows: {'PASS' if nulls_ok else 'FAIL'}")
print(f"Assembler input count 61: {'PASS' if assembler_ok else 'FAIL'}")
if pipeline_ok:
    print("\nSAVING SUBJECT-SAFE PREPROCESSING MODEL")
    if os.path.exists(LOCAL_PIPELINE):
        shutil.rmtree(LOCAL_PIPELINE)
    pipeline_model.write().overwrite().save(LOCAL_PIPELINE)
    print(f"Saved: {LOCAL_PIPELINE}")
    print("\nWRITING SUBJECT-SAFE CLEAN FEATURES")
    output_paths = {"TRAIN": f"{CLEAN}/train_features", "VALIDATION": f"{CLEAN}/validation_features", "TEST": f"{CLEAN}/test_features"}
    for name, df in transformed.items():
        output_path = output_paths[name]
        print(f"\nWriting {name}")
        print(f"Path: {output_path}")
        df.write.mode("overwrite").parquet(output_path)
        print(f"{name} written successfully")
    print("\nVERIFYING WRITTEN CLEAN FEATURES")
    write_ok = True
    for name, output_path in output_paths.items():
        written_count = spark.read.parquet(output_path).count()
        print(f"{name} written rows: {written_count:,}")
        if written_count != expected_counts[name]:
            write_ok = False
    if write_ok:
        print("\nFINAL RESULT: PREPROCESSING AND FEATURE GENERATION PASSED")
        print("SAFE TO RUN FINAL LEAKAGE AUDIT")
    else:
        print("\nFINAL RESULT: WRITE VALIDATION FAILED")
        print("DO NOT RUN XGBOOST")
else:
    print("\nFINAL RESULT: PREPROCESSING VALIDATION FAILED")
    print("CLEAN FEATURES WERE NOT WRITTEN")
    print("DO NOT RUN XGBOOST")
for df in transformed.values():
    df.unpersist()
spark.stop()
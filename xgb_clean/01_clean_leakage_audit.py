from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
BASE = "hdfs://localhost:9000/user/mahith/icu"
CLEAN = f"{BASE}/clean_features"
PIPELINE_PATH = "/home/mahith/BDA_PROJECT/checkpoints/preprocessing_model_subject_safe"
TRAIN = f"{CLEAN}/train_features"
VALIDATION = f"{CLEAN}/validation_features"
TEST = f"{CLEAN}/test_features"
EXPECTED_ROWS = {"TRAIN": 7_459_304, "VALIDATION": 1_610_439, "TEST": 1_664_239}
spark = SparkSession.builder.appName("FinalCleanLeakageAudit").master("local[2]").config("spark.driver.memory", "6g").config("spark.sql.shuffle.partitions", "32").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("FINAL CLEAN FEATURE LEAKAGE AUDIT")
datasets = {"TRAIN": spark.read.parquet(TRAIN), "VALIDATION": spark.read.parquet(VALIDATION), "TEST": spark.read.parquet(TEST)}
print("\nCHECKING ROW COUNTS")
row_counts = {}
for name, df in datasets.items():
    row_counts[name] = df.count()
    print(f"{name}: {row_counts[name]:,}")
print("\nCHECKING SCHEMAS")
for name, df in datasets.items():
    print(f"{name} columns: {df.columns}")
print("\nCHECKING FEATURE DIMENSIONS")
dimension_results = {}
for name, df in datasets.items():
    dimensions = df.select(F.size(vector_to_array("features")).alias("dim")).distinct().collect()
    dimensions = sorted([row["dim"] for row in dimensions])
    dimension_results[name] = dimensions
    print(f"{name} feature dimensions: {dimensions}")
print("\nCHECKING NULLS")
null_results = {}
for name, df in datasets.items():
    null_count = df.filter(F.col("features").isNull() | F.col("label").isNull() | F.col("stay_id").isNull() | F.col("subject_id").isNull() | F.col("window_id").isNull()).count()
    null_results[name] = null_count
    print(f"{name} null rows: {null_count:,}")
print("\nCHECKING DUPLICATE WINDOWS")
duplicate_results = {}
for name, df in datasets.items():
    duplicates = df.groupBy("stay_id", "window_id").count().filter(F.col("count") > 1).count()
    duplicate_results[name] = duplicates
    print(f"{name} duplicate (stay_id, window_id): {duplicates:,}")
print("\nCHECKING STAY OVERLAP")
train_stays = datasets["TRAIN"].select("stay_id").distinct()
validation_stays = datasets["VALIDATION"].select("stay_id").distinct()
test_stays = datasets["TEST"].select("stay_id").distinct()
train_validation_stays = train_stays.join(validation_stays, "stay_id", "inner").count()
train_test_stays = train_stays.join(test_stays, "stay_id", "inner").count()
validation_test_stays = validation_stays.join(test_stays, "stay_id", "inner").count()
print(f"TRAIN-VALIDATION stay overlap: {train_validation_stays:,}")
print(f"TRAIN-TEST stay overlap: {train_test_stays:,}")
print(f"VALIDATION-TEST stay overlap: {validation_test_stays:,}")
print("\nCHECKING SUBJECT OVERLAP")
train_subjects = datasets["TRAIN"].select("subject_id").distinct()
validation_subjects = datasets["VALIDATION"].select("subject_id").distinct()
test_subjects = datasets["TEST"].select("subject_id").distinct()
train_validation_subjects = train_subjects.join(validation_subjects, "subject_id", "inner").count()
train_test_subjects = train_subjects.join(test_subjects, "subject_id", "inner").count()
validation_test_subjects = validation_subjects.join(test_subjects, "subject_id", "inner").count()
print(f"TRAIN-VALIDATION subject overlap: {train_validation_subjects:,}")
print(f"TRAIN-TEST subject overlap: {train_test_subjects:,}")
print(f"VALIDATION-TEST subject overlap: {validation_test_subjects:,}")
print("\nCHECKING LABEL DISTRIBUTIONS")
label_results = {}
for name, df in datasets.items():
    labels = df.groupBy("label").count().orderBy("label").collect()
    label_results[name] = {row["label"]: row["count"] for row in labels}
    print(f"{name}: {label_results[name]}")
print("\nCHECKING FEATURE DATASET SCHEMA")
expected_columns = ["stay_id", "subject_id", "window_id", "features", "label"]
schema_results = {}
for name, df in datasets.items():
    schema_results[name] = df.columns == expected_columns
    print(f"{name} schema: {'PASS' if schema_results[name] else 'FAIL'}")
print("\nCHECKING FITTED PREPROCESSING PIPELINE")
pipeline_model = PipelineModel.load(PIPELINE_PATH)
assembler = pipeline_model.stages[-1]
assembler_inputs = assembler.getInputCols()
print(f"Pipeline stages: {len(pipeline_model.stages)}")
print(f"Assembler input count: {len(assembler_inputs)}")
print(f"Assembler inputs: {assembler_inputs}")
forbidden_predictive_columns = ["mortality", "outtime", "los", "last_careunit", "subject_id", "hadm_id"]
forbidden_in_assembler = [col for col in forbidden_predictive_columns if col in assembler_inputs]
print(f"Forbidden columns in assembler: {forbidden_in_assembler}")
assembler_leakage_ok = len(forbidden_in_assembler) == 0
assembler_count_ok = len(assembler_inputs) == 61
print(f"Assembler input count 61: {'PASS' if assembler_count_ok else 'FAIL'}")
print(f"Forbidden feature check: {'PASS' if assembler_leakage_ok else 'FAIL'}")
print("\nFINAL AUDIT")
rows_ok = all(row_counts[name] == EXPECTED_ROWS[name] for name in EXPECTED_ROWS)
dimensions_ok = all(dimension_results[name] == [137] for name in dimension_results)
nulls_ok = all(null_results[name] == 0 for name in null_results)
duplicates_ok = all(duplicate_results[name] == 0 for name in duplicate_results)
stay_overlap_ok = train_validation_stays == 0 and train_test_stays == 0 and validation_test_stays == 0
subject_overlap_ok = train_validation_subjects == 0 and train_test_subjects == 0 and validation_test_subjects == 0
schema_ok = all(schema_results.values())
all_checks_ok = rows_ok and dimensions_ok and nulls_ok and duplicates_ok and stay_overlap_ok and subject_overlap_ok and schema_ok and assembler_count_ok and assembler_leakage_ok
print(f"Row counts: {'PASS' if rows_ok else 'FAIL'}")
print(f"137 features: {'PASS' if dimensions_ok else 'FAIL'}")
print(f"Null check: {'PASS' if nulls_ok else 'FAIL'}")
print(f"Duplicate window check: {'PASS' if duplicates_ok else 'FAIL'}")
print(f"Stay overlap check: {'PASS' if stay_overlap_ok else 'FAIL'}")
print(f"Subject overlap check: {'PASS' if subject_overlap_ok else 'FAIL'}")
print(f"Clean feature schema: {'PASS' if schema_ok else 'FAIL'}")
print(f"Assembler input count: {'PASS' if assembler_count_ok else 'FAIL'}")
print(f"Assembler leakage check: {'PASS' if assembler_leakage_ok else 'FAIL'}")
if all_checks_ok:
    print("\nFINAL RESULT: LEAKAGE AUDIT PASSED")
    print("CLEAN DATA IS SAFE FOR XGBOOST TRAINING")
else:
    print("\nFINAL RESULT: LEAKAGE AUDIT FAILED")
    print("DO NOT TRAIN XGBOOST")
spark.stop()
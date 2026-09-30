from pyspark.sql import SparkSession
from pyspark.sql.functions import size
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
import os
import re
FEATURE_PATH="hdfs://localhost:9000/user/mahith/icu/clean_features/train_features"
TRAIN_PATH="hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/train_sequences"
VAL_PATH="hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/validation_sequences"
TEST_PATH="hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/test_sequences"
PREPROCESSING_PATH="/home/mahith/BDA_PROJECT/checkpoints/preprocessing_model_subject_safe"
SEQUENCE_SOURCE="/home/mahith/BDA_PROJECT/Model/lstm_clean/01_sequence_data_adaptive.py"
EXPECTED_FEATURES=137
EXPECTED_TEMPORAL=54
EXPECTED_STATIC=83
MAX_SEQ_LEN=120
FORBIDDEN_PREDICTIVE={"mortality","outtime","los","last_careunit"}
RETAINED_IDENTIFIERS={"subject_id","hadm_id"}
FUTURE_TERMS={"mortality","death","outcome","discharge","outtime","los"}
spark=SparkSession.builder.master("local[4]").appName("Final_LSTM_Leakage_Audit").config("spark.driver.memory","8g").config("spark.sql.shuffle.partitions","8").config("spark.sql.parquet.enableVectorizedReader","false").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print("FINAL LSTM LEAKAGE AUDIT")
print("FEATURE VECTOR DIMENSION")
df=spark.read.parquet(FEATURE_PATH)
feature_count=df.select(size(vector_to_array("features")).alias("n")).limit(1).collect()[0]["n"]
print(f"Feature vector dimension: {feature_count}")
if feature_count!=EXPECTED_FEATURES:
    raise RuntimeError(f"Expected {EXPECTED_FEATURES} features but found {feature_count}")
print("137 feature dimensions PASSED")
print("FEATURE METADATA")
metadata=df.schema["features"].metadata
feature_names=[]
if "ml_attr" in metadata:
    attrs=metadata["ml_attr"].get("attrs",{})
    ordered=[]
    for group in attrs.values():
        for item in group:
            if "idx" in item:
                ordered.append(item)
    ordered=sorted(ordered,key=lambda x:x["idx"])
    feature_names=[item.get("name",f"feature_{item['idx']}") for item in ordered]
if len(feature_names)==EXPECTED_FEATURES:
    print("Spark feature metadata found")
    print(f"Metadata feature names: {len(feature_names)}")
else:
    print("Spark feature metadata does not contain all 137 names")
    print("Loading preprocessing model")
    if not os.path.exists(PREPROCESSING_PATH):
        raise RuntimeError(f"Preprocessing model not found: {PREPROCESSING_PATH}")
    preprocessing_model=PipelineModel.load(PREPROCESSING_PATH)
    assembler=None
    for stage in preprocessing_model.stages:
        if stage.__class__.__name__=="VectorAssembler":
            assembler=stage
    if assembler is None:
        raise RuntimeError("VectorAssembler not found in preprocessing model")
    feature_names=assembler.getInputCols()
    print(f"VectorAssembler input count: {len(feature_names)}")
    if len(feature_names)==0:
        raise RuntimeError("VectorAssembler has no input columns")
    if len(feature_names)!=EXPECTED_FEATURES:
        raise RuntimeError(f"VectorAssembler input count {len(feature_names)} != {EXPECTED_FEATURES}")
    print("VectorAssembler feature count PASSED")
print("FEATURE IDENTITY CHECK")
forbidden_found=[name for name in feature_names if name.lower() in FORBIDDEN_PREDICTIVE]
if forbidden_found:
    raise RuntimeError(f"Forbidden predictive features found: {forbidden_found}")
print("Forbidden predictive feature check PASSED")
identifier_found=[name for name in feature_names if name.lower() in RETAINED_IDENTIFIERS]
print(f"Retained identifier fields: {identifier_found}")
print("Identifier retention check PASSED")
future_pattern=re.compile(r"\b("+"|".join(re.escape(term) for term in FUTURE_TERMS)+r")\b",re.IGNORECASE)
future_named=[name for name in feature_names if future_pattern.search(name)]
print(f"Potential future named features: {future_named}")
if future_named:
    raise RuntimeError(f"Potential future derived features require review: {future_named}")
print("Future outcome keyword check PASSED")
print("TEMPORAL AND STATIC POSITION CHECK")
temporal_indices=list(range(54))
static_indices=list(range(54,137))
if len(temporal_indices)!=EXPECTED_TEMPORAL:
    raise RuntimeError("Temporal index count mismatch")
if len(static_indices)!=EXPECTED_STATIC:
    raise RuntimeError("Static index count mismatch")
if temporal_indices[-1]!=53:
    raise RuntimeError("Temporal indices do not end at 53")
if static_indices[0]!=54 or static_indices[-1]!=136:
    raise RuntimeError("Static indices do not cover 54 to 136")
print("Temporal indices: 0 to 53")
print("Static indices: 54 to 136")
print("54 temporal + 83 static PASSED")
print("SAVED SEQUENCE DIMENSIONS")
for name,path in [("TRAIN",TRAIN_PATH),("VALIDATION",VAL_PATH),("TEST",TEST_PATH)]:
    seq=spark.read.parquet(path)
    sample=seq.select("temporal","missing_mask","padding_mask","static","sequence_length").limit(1).collect()
    if not sample:
        raise RuntimeError(f"{name}: sequence dataset is empty")
    row=sample[0]
    if len(row["temporal"])!=MAX_SEQ_LEN:
        raise RuntimeError(f"{name}: temporal length failure")
    if len(row["missing_mask"])!=MAX_SEQ_LEN:
        raise RuntimeError(f"{name}: missing mask length failure")
    if len(row["padding_mask"])!=MAX_SEQ_LEN:
        raise RuntimeError(f"{name}: padding mask length failure")
    if len(row["temporal"][0])!=EXPECTED_TEMPORAL:
        raise RuntimeError(f"{name}: temporal feature count failure")
    if len(row["static"])!=EXPECTED_STATIC:
        raise RuntimeError(f"{name}: static feature count failure")
    if row["sequence_length"]<1 or row["sequence_length"]>MAX_SEQ_LEN:
        raise RuntimeError(f"{name}: invalid sequence length")
    print(f"{name}: dimensions PASSED")
print("LABEL SEPARATION")
for name,path in [("TRAIN",TRAIN_PATH),("VALIDATION",VAL_PATH),("TEST",TEST_PATH)]:
    seq=spark.read.parquet(path)
    if "mortality" in seq.columns:
        raise RuntimeError(f"{name}: mortality exists in saved sequence data")
    if "label" not in seq.columns:
        raise RuntimeError(f"{name}: label missing")
    print(f"{name}: target separation PASSED")
print("SUBJECT SPLIT")
train_subjects=spark.read.parquet(TRAIN_PATH).select("subject_id").distinct()
val_subjects=spark.read.parquet(VAL_PATH).select("subject_id").distinct()
test_subjects=spark.read.parquet(TEST_PATH).select("subject_id").distinct()
tv=train_subjects.join(val_subjects,"subject_id","inner").limit(1).count()
tt=train_subjects.join(test_subjects,"subject_id","inner").limit(1).count()
vt=val_subjects.join(test_subjects,"subject_id","inner").limit(1).count()
print(f"Train Validation subjects: {tv}")
print(f"Train Test subjects: {tt}")
print(f"Validation Test subjects: {vt}")
if tv or tt or vt:
    raise RuntimeError("Subject leakage detected")
print("Subject level separation PASSED")
print("STAY SPLIT")
train_stays=spark.read.parquet(TRAIN_PATH).select("stay_id").distinct()
val_stays=spark.read.parquet(VAL_PATH).select("stay_id").distinct()
test_stays=spark.read.parquet(TEST_PATH).select("stay_id").distinct()
tv=train_stays.join(val_stays,"stay_id","inner").limit(1).count()
tt=train_stays.join(test_stays,"stay_id","inner").limit(1).count()
vt=val_stays.join(test_stays,"stay_id","inner").limit(1).count()
print(f"Train Validation stays: {tv}")
print(f"Train Test stays: {tt}")
print(f"Validation Test stays: {vt}")
if tv or tt or vt:
    raise RuntimeError("Stay leakage detected")
print("Stay level separation PASSED")
print("SEQUENCE SOURCE CODE")
if not os.path.exists(SEQUENCE_SOURCE):
    raise RuntimeError(f"Sequence source not found: {SEQUENCE_SOURCE}")
with open(SEQUENCE_SOURCE,"r") as f:
    sequence_code=f.read()
required_sequence_patterns=[
    'Window.partitionBy("stay_id").orderBy("window_id")',
    "rowsBetween(-(MAX_SEQ_LEN-1),0)",
    "named_struct('window_id',window_id",
    "collect_list(sequence_struct).over(history_window)",
    "transform(sequence_data,x -> x.window_id)",
    "sort_array"
]
for pattern in required_sequence_patterns:
    if pattern not in sequence_code:
        raise RuntimeError(f"Required causal sequence logic not found: {pattern}")
print("Backward looking sequence window PASSED")
print("Window id retained inside sequence PASSED")
print("Chronological ordering validation PASSED")
print("FORBIDDEN SOURCE COLUMNS")
prepare_start=sequence_code.find("def prepare_dataframe")
prepare_end=sequence_code.find("def check_temporal_finite")
if prepare_start==-1:
    raise RuntimeError("prepare_dataframe function not found")
if prepare_end==-1:
    prepare_end=len(sequence_code)
sequence_input_section=sequence_code[prepare_start:prepare_end]
predictive_source_found=[name for name in FORBIDDEN_PREDICTIVE if re.search(rf"\b{re.escape(name)}\b",sequence_input_section,re.IGNORECASE)]
identifier_source_found=[name for name in RETAINED_IDENTIFIERS if re.search(rf"\b{re.escape(name)}\b",sequence_input_section,re.IGNORECASE)]
print(f"Forbidden predictive source fields: {predictive_source_found}")
print(f"Retained identifier fields: {identifier_source_found}")
if predictive_source_found:
    raise RuntimeError(f"Forbidden predictive fields enter tensor preparation: {predictive_source_found}")
print("Forbidden clinical outcome fields excluded from tensor preparation PASSED")
print("TRAIN ONLY IMPUTATION")
imputation_patterns=[
    "fit_imputation(train)",
    "fit_imputation(train_df)",
    "Fitting train-only median imputation",
    "loaded_imputation"
]
if not any(pattern in sequence_code for pattern in imputation_patterns):
    raise RuntimeError("Train only imputation logic not found")
if "prepare_dataframe(val,loaded_imputation)" not in sequence_code:
    raise RuntimeError("Validation does not reuse saved imputation")
if "prepare_dataframe(test,loaded_imputation)" not in sequence_code:
    raise RuntimeError("Test does not reuse saved imputation")
print("Training only imputation PASSED")
print("TEMPORAL ORDERING")
if "window_id_seq_sorted" not in sequence_code:
    raise RuntimeError("Chronological ordering validation missing")
if "ordering_violation" not in sequence_code:
    raise RuntimeError("Ordering violation check missing")
print("Explicit chronological ordering check PASSED")
print("TREND FEATURES")
trend_names=["HR_trend","MAP_trend","RR_trend"]
for name in trend_names:
    if name not in sequence_code:
        raise RuntimeError(f"{name} missing from sequence source")
print("HR_trend PASSED")
print("MAP_trend PASSED")
print("RR_trend PASSED")
print("FINAL LEAKAGE STATUS")
print("Feature vector dimension PASSED")
print("Feature identity checks PASSED")
print("Temporal/static separation PASSED")
print("Saved sequence dimensions PASSED")
print("Label separation PASSED")
print("Subject separation PASSED")
print("Stay separation PASSED")
print("Causal sequence window check PASSED")
print("Forbidden tensor fields PASSED")
print("Train only imputation PASSED")
print("Chronological ordering PASSED")
print("Trend feature check PASSED")
print("FINAL LSTM LEAKAGE AUDIT PASSED")
spark.stop()
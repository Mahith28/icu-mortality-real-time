from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, expr
import math
TRAIN_PATH="hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/train_sequences"
VAL_PATH="hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/validation_sequences"
TEST_PATH="hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/test_sequences"
EXPECTED_TEMPORAL=54
EXPECTED_STATIC=83
MAX_SEQ_LEN=120
spark=(
    SparkSession.builder
    .appName("LSTM_Leakage_Audit")
    .config("spark.sql.parquet.enableVectorizedReader","false")
    .config("spark.sql.shuffle.partitions","8")
    .config("spark.default.parallelism","8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
def load_dataset(path):
    return spark.read.parquet(path)
def basic_checks(name,path):
    df=load_dataset(path)
    required_columns={
        "stay_id",
        "subject_id",
        "terminal_window_id",
        "sequence_length",
        "label",
        "temporal",
        "missing_mask",
        "padding_mask",
        "static"
    }
    missing=required_columns-set(df.columns)
    if missing:
        raise RuntimeError(f"{name}: missing columns: {missing}")
    print(f"{name}: required columns PASSED")
    rows=df.count()
    print(f"{name} rows: {rows}")
    bad_labels=df.filter(
        (col("label")!=0)&(col("label")!=1)
    ).limit(1).count()
    if bad_labels:
        raise RuntimeError(f"{name}: invalid labels detected")
    print(f"{name}: label check PASSED")
    bad_length=df.filter(
        (col("sequence_length")<1)|
        (col("sequence_length")>MAX_SEQ_LEN)
    ).limit(1).count()
    if bad_length:
        raise RuntimeError(f"{name}: invalid sequence length detected")
    print(f"{name}: sequence length PASSED")
    sample=df.select(
        "sequence_length",
        "temporal",
        "missing_mask",
        "padding_mask",
        "static",
        "terminal_window_id"
    ).limit(1).collect()
    if not sample:
        raise RuntimeError(f"{name}: dataset is empty")
    row=sample[0]
    temporal=row["temporal"]
    missing_mask=row["missing_mask"]
    padding_mask=row["padding_mask"]
    static=row["static"]
    if len(temporal)!=MAX_SEQ_LEN:
        raise RuntimeError(
            f"{name}: temporal length is {len(temporal)}, expected {MAX_SEQ_LEN}"
        )
    if len(missing_mask)!=MAX_SEQ_LEN:
        raise RuntimeError(
            f"{name}: missing-mask length is {len(missing_mask)}, expected {MAX_SEQ_LEN}"
        )
    if len(padding_mask)!=MAX_SEQ_LEN:
        raise RuntimeError(
            f"{name}: padding-mask length is {len(padding_mask)}, expected {MAX_SEQ_LEN}"
        )
    if len(static)!=EXPECTED_STATIC:
        raise RuntimeError(
            f"{name}: static feature count is {len(static)}, expected {EXPECTED_STATIC}"
        )
    if len(temporal[0])!=EXPECTED_TEMPORAL:
        raise RuntimeError(
            f"{name}: temporal feature count is {len(temporal[0])}, expected {EXPECTED_TEMPORAL}"
        )
    print(f"{name}: fixed padding length PASSED")
    print(f"{name}: feature dimensions PASSED")
    sample_length=row["sequence_length"]
    expected_padding=MAX_SEQ_LEN-sample_length
    actual_padding=sum(padding_mask)
    if actual_padding!=expected_padding:
        raise RuntimeError(
            f"{name}: padding mask does not match sequence_length"
        )
    print(f"{name}: padding-mask ↔ sequence_length PASSED")
    if any(x not in (0,1) for x in padding_mask):
        raise RuntimeError(f"{name}: invalid padding-mask values")
    print(f"{name}: padding-mask values PASSED")
    if row["terminal_window_id"]<0:
        raise RuntimeError(f"{name}: invalid terminal_window_id")
    print(f"{name}: terminal window check PASSED")
def partition_tensor_check(name,path):
    df=load_dataset(path).select(
        "temporal",
        "missing_mask",
        "padding_mask",
        "static"
    )
    def check_partition(rows):
        for row in rows:
            temporal=row["temporal"]
            missing_mask=row["missing_mask"]
            padding_mask=row["padding_mask"]
            static=row["static"]
            if len(temporal)!=MAX_SEQ_LEN:
                yield f"{name}: invalid temporal length"
                return
            if len(missing_mask)!=MAX_SEQ_LEN:
                yield f"{name}: invalid missing-mask length"
                return
            if len(padding_mask)!=MAX_SEQ_LEN:
                yield f"{name}: invalid padding-mask length"
                return
            if len(static)!=EXPECTED_STATIC:
                yield f"{name}: invalid static dimension"
                return
            for timestep in temporal:
                if len(timestep)!=EXPECTED_TEMPORAL:
                    yield f"{name}: invalid temporal feature dimension"
                    return
                for value in timestep:
                    if value is None or not math.isfinite(float(value)):
                        yield f"{name}: invalid numerical value in temporal tensor"
                        return
            for timestep in missing_mask:
                for value in timestep:
                    if value not in (0,1):
                        yield f"{name}: invalid missing-mask value"
                        return
            for value in padding_mask:
                if value not in (0,1):
                    yield f"{name}: invalid padding-mask value"
                    return
            for value in static:
                if value is None or not math.isfinite(float(value)):
                    yield f"{name}: invalid numerical value in static tensor"
                    return
    bad=df.rdd.mapPartitions(check_partition).take(1)
    if bad:
        raise RuntimeError(bad[0])
    print(f"{name}: full tensor numerical/mask scan PASSED")
def subject_overlap_check():
    train_subjects=load_dataset(TRAIN_PATH).select("subject_id").distinct()
    val_subjects=load_dataset(VAL_PATH).select("subject_id").distinct()
    test_subjects=load_dataset(TEST_PATH).select("subject_id").distinct()
    train_val=train_subjects.join(
        val_subjects,
        "subject_id",
        "inner"
    ).limit(1).count()
    train_test=train_subjects.join(
        test_subjects,
        "subject_id",
        "inner"
    ).limit(1).count()
    val_test=val_subjects.join(
        test_subjects,
        "subject_id",
        "inner"
    ).limit(1).count()
    print(f"Train Validation subjects: {train_val}")
    print(f"Train Test subjects: {train_test}")
    print(f"Validation Test subjects: {val_test}")
    if train_val or train_test or val_test:
        raise RuntimeError("PATIENT-LEVEL LEAKAGE DETECTED")
    print("Patient-level split check PASSED")
def sequence_length_padding_check(name,path):
    df=load_dataset(path).select(
        "sequence_length",
        "padding_mask"
    )
    def check_partition(rows):
        for row in rows:
            sequence_length=row["sequence_length"]
            padding_mask=row["padding_mask"]
            if len(padding_mask)!=MAX_SEQ_LEN:
                yield f"{name}: padding-mask length failure"
                return
            if sequence_length<1 or sequence_length>MAX_SEQ_LEN:
                yield f"{name}: sequence length failure"
                return
            if sum(padding_mask)!=(MAX_SEQ_LEN-sequence_length):
                yield f"{name}: padding-mask ↔ sequence_length failure"
                return
            expected=([0]*sequence_length)+(
                [1]*(MAX_SEQ_LEN-sequence_length)
            )
            if padding_mask!=expected:
                yield f"{name}: padding-mask ordering failure"
                return
    bad=df.rdd.mapPartitions(check_partition).take(1)
    if bad:
        raise RuntimeError(bad[0])
    print(f"{name}: full padding relationship PASSED")
print("LSTM ADAPTIVE DATASET LEAKAGE AUDIT")
print("BASIC DATASET CHECKS")
basic_checks("TRAIN",TRAIN_PATH)
basic_checks("VALIDATION",VAL_PATH)
basic_checks("TEST",TEST_PATH)
print("FULL TENSOR INTEGRITY CHECKS")
partition_tensor_check("TRAIN",TRAIN_PATH)
partition_tensor_check("VALIDATION",VAL_PATH)
partition_tensor_check("TEST",TEST_PATH)
print("FULL PADDING RELATIONSHIP CHECKS")
sequence_length_padding_check("TRAIN",TRAIN_PATH)
sequence_length_padding_check("VALIDATION",VAL_PATH)
sequence_length_padding_check("TEST",TEST_PATH)
print("PATIENT-LEVEL SPLIT CHECK")
subject_overlap_check()
print("ALL SAVED-SEQUENCE INTEGRITY CHECKS PASSED")
spark.stop()
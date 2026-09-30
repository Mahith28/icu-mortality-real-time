import os
import json
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, when, isnan, size, array, expr, lit, concat, row_number, count, collect_list, array_repeat, sort_array
from pyspark.ml.functions import vector_to_array
TRAIN_PATH="/user/mahith/icu/clean_features/train_features"
VAL_PATH="/user/mahith/icu/clean_features/validation_features"
TEST_PATH="/user/mahith/icu/clean_features/test_features"
IMPUTATION_PATH="hdfs://localhost:9000/user/mahith/icu/lstm/artifacts/lstm_imputation_values.json"
TRAIN_OUTPUT="hdfs://localhost:9000/user/mahith/icu/lstm/train_sequences"
VALIDATION_OUTPUT="hdfs://localhost:9000/user/mahith/icu/lstm/validation_sequences"
TEST_OUTPUT="hdfs://localhost:9000/user/mahith/icu/lstm/test_sequences"
MAX_SEQ_LEN=120
EXPECTED_FEATURES=137
TEMPORAL_COUNT=54
STATIC_COUNT=83
TEMPORAL_INDICES=list(range(54))
STATIC_INDICES=list(range(54,137))
TEMPORAL_NAMES=[
    "Heart_Rate_mean","Heart_Rate_min","Heart_Rate_max","Heart_Rate_last",
    "SBP_mean","SBP_min","SBP_max","SBP_last",
    "DBP_mean","DBP_min","DBP_max","DBP_last",
    "MAP_mean","MAP_min","MAP_max","MAP_last",
    "Respiratory_Rate_mean","Respiratory_Rate_min","Respiratory_Rate_max","Respiratory_Rate_last",
    "Temperature_mean","Temperature_min","Temperature_max","Temperature_last",
    "SpO2_mean","SpO2_min","SpO2_max","SpO2_last",
    "GCS_Eye_mean","GCS_Eye_min","GCS_Eye_max","GCS_Eye_last",
    "GCS_Verbal_mean","GCS_Verbal_min","GCS_Verbal_max","GCS_Verbal_last",
    "GCS_Motor_mean","GCS_Motor_min","GCS_Motor_max","GCS_Motor_last",
    "Heart_Rate_missing","SBP_missing","DBP_missing","MAP_missing","Respiratory_Rate_missing",
    "Temperature_missing","SpO2_missing","GCS_Eye_missing","GCS_Verbal_missing","GCS_Motor_missing",
    "HR_trend","MAP_trend","RR_trend","GCS_Total_last"
]
CHECKPOINTS=[1,5,10,20,30,60,90,120]
if len(TEMPORAL_NAMES)!=TEMPORAL_COUNT:
    raise RuntimeError(f"Expected {TEMPORAL_COUNT} temporal names but found {len(TEMPORAL_NAMES)}")
def create_hdfs_directory(path):
    result=os.system(f"hdfs dfs -mkdir -p {path}")
    if result!=0:
        raise RuntimeError(f"Failed to create HDFS directory: {path}")
def delete_hdfs_directory(path):
    result=os.system(f"hdfs dfs -rm -r -f {path}")
    if result!=0:
        print(f"HDFS path not present: {path}")
def save_json_hdfs(data,path):
    local_path="/tmp/lstm_imputation_values.json"
    with open(local_path,"w") as f:
        json.dump(data,f,indent=2)
    parent=os.path.dirname(path)
    create_hdfs_directory(parent)
    result=os.system(f"hdfs dfs -put -f {local_path} {path}")
    os.remove(local_path)
    if result!=0:
        raise RuntimeError(f"Failed to save imputation artifact: {path}")
def load_json_hdfs(path):
    local_path="/tmp/lstm_imputation_values_read.json"
    result=os.system(f"hdfs dfs -get -f {path} {local_path}")
    if result!=0:
        raise RuntimeError(f"Failed to load imputation artifact: {path}")
    with open(local_path,"r") as f:
        data=json.load(f)
    os.remove(local_path)
    return data
def fit_imputation(train_df):
    print("Fitting train-only median imputation")
    medians={}
    df=train_df.select("features").withColumn("x",vector_to_array("features"))
    for i,name in enumerate(TEMPORAL_NAMES):
        print(f"Calculating training median: {name}")
        values=df.select(col("x")[i].alias("value")).filter(
            col("value").isNotNull() & ~isnan(col("value"))
        ).approxQuantile("value",[0.5],0.001)
        if not values:
            raise RuntimeError(f"No valid training values found for feature {name}")
        medians[str(i)]=float(values[0])
    print(f"Training medians calculated: {len(medians)}")
    return medians
def prepare_dataframe(df,imputation):
    df=df.withColumn("x",vector_to_array("features"))
    temporal=[]
    temporal_missing=[]
    for i in TEMPORAL_INDICES:
        value=col("x")[i]
        median=float(imputation[str(i)])
        is_missing=value.isNull() | isnan(value)
        temporal.append(when(is_missing,median).otherwise(value).cast("float"))
        temporal_missing.append(when(is_missing,1).otherwise(0).cast("int"))
    static=[col("x")[i].cast("float") for i in STATIC_INDICES]
    df=df.withColumn("temporal",array(*temporal))
    df=df.withColumn("temporal_missing",array(*temporal_missing))
    df=df.withColumn("static",array(*static))
    return df.select("stay_id","subject_id","window_id","label","temporal","temporal_missing","static")
def check_temporal_finite(df,split):
    print(f"Checking temporal values: {split}")
    condition=None
    for i in range(TEMPORAL_COUNT):
        current=col("temporal")[i].isNull() | isnan(col("temporal")[i])
        condition=current if condition is None else condition | current
    bad=df.filter(condition).limit(1).count()
    if bad>0:
        raise RuntimeError(f"Invalid temporal values found in {split}")
    print(f"Temporal values passed: {split}")
def check_static_finite(df,split):
    print(f"Checking static values: {split}")
    condition=None
    for i in range(STATIC_COUNT):
        current=col("static")[i].isNull() | isnan(col("static")[i])
        condition=current if condition is None else condition | current
    bad=df.filter(condition).limit(1).count()
    if bad>0:
        raise RuntimeError(f"Invalid static values found in {split}")
    print(f"Static values passed: {split}")
def check_static_consistency(df,split):
    print(f"Checking static consistency: {split}")
    aggregations=[]
    for i in range(STATIC_COUNT):
        aggregations.append(expr(f"min(static[{i}])").alias(f"min_{i}"))
        aggregations.append(expr(f"max(static[{i}])").alias(f"max_{i}"))
    stats=df.groupBy("stay_id").agg(*aggregations)
    violation=None
    for i in range(STATIC_COUNT):
        current=col(f"min_{i}")!=col(f"max_{i}")
        violation=current if violation is None else violation | current
    bad=stats.filter(violation).limit(1).count()
    if bad>0:
        raise RuntimeError(f"Static features vary within a stay in {split}")
    print(f"Static consistency passed: {split}")
def build_sequences(df,split,per_window=False):
    print(f"Building sequences: {split}")
    print(f"Per-window mode: {per_window}")
    order_window=Window.partitionBy("stay_id").orderBy("window_id")
    history_window=Window.partitionBy("stay_id").orderBy("window_id").rowsBetween(-(MAX_SEQ_LEN-1),0)
    count_window=Window.partitionBy("stay_id")
    df=df.withColumn("position",row_number().over(order_window))
    df=df.withColumn("stay_window_count",count("*").over(count_window))
    sequence_struct=expr(
        "named_struct('window_id',window_id,'temporal',temporal,'temporal_missing',temporal_missing)"
    )
    df=df.withColumn(
        "sequence_data",
        collect_list(sequence_struct).over(history_window)
    )
    if per_window:
        df=df.withColumn("selected",lit(True))
    else:
        checkpoint_condition=None
        for checkpoint in CHECKPOINTS:
            current=col("position")==checkpoint
            checkpoint_condition=current if checkpoint_condition is None else checkpoint_condition | current
        checkpoint_condition=checkpoint_condition | (col("position")==col("stay_window_count"))
        df=df.withColumn("selected",checkpoint_condition)
    df=df.filter(col("selected"))
    df=df.withColumn("window_id_seq",expr("transform(sequence_data,x -> x.window_id)"))
    df=df.withColumn("window_id_seq_sorted",sort_array(col("window_id_seq")))
    ordering_violation=df.filter(col("window_id_seq_sorted")!=col("window_id_seq")).limit(1).count()
    if ordering_violation>0:
        raise RuntimeError(f"Non-chronological ordering detected in {split}")
    print(f"Temporal ordering passed: {split}")
    df=df.drop("window_id_seq","window_id_seq_sorted")
    df=df.withColumn(
        "sequence_length",
        size(col("sequence_data"))
    )
    df=df.withColumn(
        "temporal_raw",
        expr("transform(sequence_data,x -> x.temporal)")
    )
    df=df.withColumn(
        "missing_raw",
        expr("transform(sequence_data,x -> x.temporal_missing)")
    )
    df=df.withColumn(
        "temporal",
        concat(
            col("temporal_raw"),
            array_repeat(
                array_repeat(
                    lit(0.0).cast("float"),
                    TEMPORAL_COUNT
                ),
                MAX_SEQ_LEN-size(col("temporal_raw"))
            )
        ).cast("array<array<float>>")
    )
    df=df.withColumn(
        "missing_mask",
        concat(
            col("missing_raw"),
            array_repeat(
                array_repeat(
                    lit(0).cast("int"),
                    TEMPORAL_COUNT
                ),
                MAX_SEQ_LEN-size(col("missing_raw"))
            )
        ).cast("array<array<int>>")
    )
    df=df.withColumn(
        "padding_mask",
        concat(
            array_repeat(
                lit(0).cast("int"),
                col("sequence_length")
            ),
            array_repeat(
                lit(1).cast("int"),
                MAX_SEQ_LEN-col("sequence_length")
            )
        ).cast("array<int>")
    )
    result=df.select(
        col("stay_id").cast("long"),
        col("subject_id").cast("long"),
        col("window_id").cast("long").alias("terminal_window_id"),
        col("sequence_length").cast("int"),
        col("label").cast("int"),
        col("temporal").cast("array<array<float>>"),
        col("missing_mask").cast("array<array<int>>"),
        col("padding_mask").cast("array<int>"),
        col("static").cast("array<float>")
    )
    return result
def validate_sequence_dataframe(df,split):
    print(f"Validating sequences: {split}")
    required=[
        "stay_id",
        "subject_id",
        "terminal_window_id",
        "sequence_length",
        "label",
        "temporal",
        "missing_mask",
        "padding_mask",
        "static"
    ]
    for name in required:
        if name not in df.columns:
            raise RuntimeError(f"Missing required column: {name}")
    sample=df.limit(1).collect()
    if not sample:
        raise RuntimeError(f"No sequences generated for {split}")
    row=sample[0]
    if len(row.temporal)!=MAX_SEQ_LEN:
        raise RuntimeError(f"Invalid temporal length in {split}")
    if len(row.missing_mask)!=MAX_SEQ_LEN:
        raise RuntimeError(f"Invalid missing mask length in {split}")
    if len(row.padding_mask)!=MAX_SEQ_LEN:
        raise RuntimeError(f"Invalid padding mask length in {split}")
    if len(row.temporal[0])!=TEMPORAL_COUNT:
        raise RuntimeError(f"Invalid temporal feature count in {split}")
    if len(row.missing_mask[0])!=TEMPORAL_COUNT:
        raise RuntimeError(f"Invalid missingness feature count in {split}")
    if len(row.static)!=STATIC_COUNT:
        raise RuntimeError(f"Invalid static feature count in {split}")
    if row.sequence_length<1 or row.sequence_length>MAX_SEQ_LEN:
        raise RuntimeError(f"Invalid sequence length in {split}")
    print(f"{split} sequence validation passed")
def save_sequences(df,output_path):
    delete_hdfs_directory(output_path)
    df=df.select(
        col("stay_id").cast("long"),
        col("subject_id").cast("long"),
        col("terminal_window_id").cast("long"),
        col("sequence_length").cast("int"),
        col("label").cast("int"),
        col("temporal").cast("array<array<float>>"),
        col("missing_mask").cast("array<array<int>>"),
        col("padding_mask").cast("array<int>"),
        col("static").cast("array<float>")
    ).repartition(9)
    df.write.mode("overwrite").parquet(output_path)
    return df
def check_sequence_balance(df,split):
    print(f"Checking sequence balance: {split}")
    counts=df.groupBy("label").count().collect()
    result={int(row["label"]):int(row["count"]) for row in counts}
    print(f"{split} label counts: {result}")
def check_missing_mask(df,split):
    print(f"Checking missingness mask: {split}")
    result=df.selectExpr(
        "aggregate(missing_mask,0,(acc,x) -> acc + aggregate(x,0,(a,v) -> a + v)) as missing_count"
    ).agg(
        {"missing_count":"sum"}
    ).collect()[0][0]
    print(f"{split} missing-mask values: {int(result or 0)}")
def main():
    spark=SparkSession.builder \
        .master("local[8]") \
        .appName("LSTM Sequence Data") \
        .config("spark.hadoop.fs.defaultFS","hdfs://localhost:9000") \
        .config("spark.sql.shuffle.partitions","200") \
        .config("spark.driver.memory","12g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("Loading clean feature datasets")
    train=spark.read.parquet(TRAIN_PATH)
    val=spark.read.parquet(VAL_PATH)
    test=spark.read.parquet(TEST_PATH)
    train_count=train.count()
    val_count=val.count()
    test_count=test.count()
    print(f"Training rows: {train_count}")
    print(f"Validation rows: {val_count}")
    print(f"Test rows: {test_count}")
    print("Checking feature dimensions")
    for df,name in [(train,"train"),(val,"validation"),(test,"test")]:
        feature_count=df.select(
            size(vector_to_array("features")).alias("feature_count")
        ).limit(1).collect()[0]["feature_count"]
        if feature_count!=EXPECTED_FEATURES:
            raise RuntimeError(f"{name}: expected {EXPECTED_FEATURES} features but found {feature_count}")
    print("All datasets contain 137 features")
    print("Temporal features: 54")
    print("Static features: 83")
    imputation=fit_imputation(train)
    save_json_hdfs(imputation,IMPUTATION_PATH)
    loaded_imputation=load_json_hdfs(IMPUTATION_PATH)
    if loaded_imputation!=imputation:
        raise RuntimeError("Imputation artifact round-trip failed")
    print("Train-only imputation artifact verified")
    train_df=prepare_dataframe(train,loaded_imputation)
    val_df=prepare_dataframe(val,loaded_imputation)
    test_df=prepare_dataframe(test,loaded_imputation)
    print("Prepared temporal, missingness, and static features")
    check_temporal_finite(train_df,"train")
    check_temporal_finite(val_df,"validation")
    check_temporal_finite(test_df,"test")
    check_static_finite(train_df,"train")
    check_static_finite(val_df,"validation")
    check_static_finite(test_df,"test")
    check_static_consistency(train_df,"train")
    check_static_consistency(val_df,"validation")
    check_static_consistency(test_df,"test")
    train_sequence_df=build_sequences(train_df,"train",per_window=False)
    val_sequence_df=build_sequences(val_df,"validation",per_window=False)
    test_sequence_df=build_sequences(test_df,"test",per_window=True)
    train_sequence_df=save_sequences(train_sequence_df,TRAIN_OUTPUT)
    val_sequence_df=save_sequences(val_sequence_df,VALIDATION_OUTPUT)
    test_sequence_df=save_sequences(test_sequence_df,TEST_OUTPUT)
    validate_sequence_dataframe(train_sequence_df,"train")
    validate_sequence_dataframe(val_sequence_df,"validation")
    validate_sequence_dataframe(test_sequence_df,"test")
    check_sequence_balance(train_sequence_df,"train")
    check_sequence_balance(val_sequence_df,"validation")
    check_sequence_balance(test_sequence_df,"test")
    check_missing_mask(train_sequence_df,"train")
    check_missing_mask(val_sequence_df,"validation")
    check_missing_mask(test_sequence_df,"test")
    train_sequences_count=train_sequence_df.count()
    val_sequences_count=val_sequence_df.count()
    test_sequences_count=test_sequence_df.count()
    print(f"Train sequences: {train_sequences_count}")
    print(f"Validation sequences: {val_sequences_count}")
    print(f"Test sequences: {test_sequences_count}")
    print("Sequence generation completed successfully")
    spark.stop()
if __name__=="__main__":
    main()

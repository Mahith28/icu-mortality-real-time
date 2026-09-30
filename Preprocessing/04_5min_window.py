from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from functools import reduce
spark = SparkSession.builder \
    .appName("Create 5 Minute Window") \
    .master("local[4]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "64") \
    .getOrCreate()
print("Creating 5 Minute Windows")
master_dataset = spark.read.parquet("/home/mahith/BDA_PROJECT/processed_data/master_dataset.parquet")
vitals_dataset = spark.read.parquet("/home/mahith/BDA_PROJECT/processed_data/vitals_dataset.parquet")
print("Datasets Loaded")
dataset = vitals_dataset.join(master_dataset, on="stay_id", how="inner")
dataset = dataset.withColumn("value", col("value").cast("double"))
dataset = dataset.withColumn(
    "minutes_from_admission",
    (unix_timestamp(col("charttime")) - unix_timestamp(col("intime"))) / 60
)
dataset = dataset.filter(col("minutes_from_admission") >= 0)
dataset = dataset.withColumn(
    "window_id",
    floor(col("minutes_from_admission") / 5)
)
dataset = dataset.cache()
print(f"Total rows: {dataset.count():,}")
print(f"Unique stays: {dataset.select('stay_id').distinct().count():,}")
raw_dupes = dataset.groupBy("stay_id", "charttime", "vital_sign") \
    .count().filter("count > 1")
print("Duplicate raw events:", raw_dupes.count())
def aggregate_vital(df, vital_name, prefix):
    vital = df.filter(col("vital_sign") == vital_name)
    stats = vital.groupBy("stay_id", "window_id").agg(
        avg("value").alias(f"{prefix}_mean"),
        min("value").alias(f"{prefix}_min"),
        max("value").alias(f"{prefix}_max")
    )
    window_spec = Window.partitionBy("stay_id", "window_id") \
        .orderBy(col("charttime").desc())
    last_vals = vital.withColumn(
        "rn", row_number().over(window_spec)
    ).filter(col("rn") == 1).select(
        "stay_id", "window_id",
        col("value").alias(f"{prefix}_last")
    )
    return stats.join(last_vals, ["stay_id", "window_id"], "inner")
heart_rate = aggregate_vital(dataset, "Heart Rate", "Heart_Rate")
sbp = aggregate_vital(dataset, "SBP", "SBP")
dbp = aggregate_vital(dataset, "DBP", "DBP")
map_df = aggregate_vital(dataset, "MAP", "MAP")
resp_rate = aggregate_vital(dataset, "Respiratory Rate", "Respiratory_Rate")
temperature = aggregate_vital(dataset, "Temperature", "Temperature")
spo2 = aggregate_vital(dataset, "SpO2", "SpO2")
gcs_eye = aggregate_vital(dataset, "GCS Eye", "GCS_Eye")
gcs_verbal = aggregate_vital(dataset, "GCS Verbal", "GCS_Verbal")
gcs_motor = aggregate_vital(dataset, "GCS Motor", "GCS_Motor")
dfs = [
    heart_rate, sbp, dbp, map_df, resp_rate,
    temperature, spo2, gcs_eye, gcs_verbal, gcs_motor
]
window_dataset = reduce(
    lambda df1, df2: df1.join(df2, ["stay_id", "window_id"], "outer"),
    dfs
)
print("All Vital Features Joined")
window_spec_fill = Window.partitionBy("stay_id") \
    .orderBy("window_id") \
    .rowsBetween(Window.unboundedPreceding, 0)
vital_prefixes = [
    "Heart_Rate", "SBP", "DBP", "MAP",
    "Respiratory_Rate", "Temperature", "SpO2",
    "GCS_Eye", "GCS_Verbal", "GCS_Motor"
]
for prefix in vital_prefixes:
    window_dataset = window_dataset.withColumn(
        f"{prefix}_missing",
        when(col(f"{prefix}_last").isNull(), 1).otherwise(0)
    )
for prefix in vital_prefixes:
    for suffix in ["_mean", "_min", "_max", "_last"]:
        col_name = prefix + suffix
        window_dataset = window_dataset.withColumn(
            col_name,
            last(col(col_name), ignorenulls=True).over(window_spec_fill)
        )
print("LOCF Applied")
window_dataset = window_dataset.withColumn(
    "HR_trend",
    col("Heart_Rate_last") - col("Heart_Rate_mean")
)
window_dataset = window_dataset.withColumn(
    "MAP_trend",
    col("MAP_last") - col("MAP_mean")
)
window_dataset = window_dataset.withColumn(
    "RR_trend",
    col("Respiratory_Rate_last") - col("Respiratory_Rate_mean")
)
window_dataset = window_dataset.withColumn(
    "GCS_Total_last",
    coalesce(col("GCS_Eye_last"), lit(0)) +
    coalesce(col("GCS_Verbal_last"), lit(0)) +
    coalesce(col("GCS_Motor_last"), lit(0))
)
patient_info = master_dataset.select(
    "stay_id", "subject_id", "hadm_id", "gender", "anchor_age",
    "admission_type", "admission_location", "insurance", "race",
    "first_careunit", "last_careunit",
    "intime", "outtime", "los", "mortality"
).dropDuplicates(["stay_id"])
window_dataset = window_dataset.join(patient_info, "stay_id", "left")
for col_name in ["race", "insurance", "admission_location"]:
    window_dataset = window_dataset.withColumn(
        col_name,
        when(col(col_name) == "UNKNOWN", None).otherwise(col(col_name))
    )
print("\nRunning Full Null Audit\n")
null_counts = window_dataset.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in window_dataset.columns
])
null_counts.show(truncate=False)
print("\nSanity Check Values:")
window_dataset.select(
    "Heart_Rate_mean", "SBP_mean", "Temperature_mean"
).show(5)
window_dataset.write.mode("overwrite") \
    .parquet("/home/mahith/BDA_PROJECT/processed_data/window_dataset.parquet")
print("\nREAL-TIME Completed Successfully!")
spark.stop()
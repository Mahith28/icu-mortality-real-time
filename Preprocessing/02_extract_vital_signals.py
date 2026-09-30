from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
spark = SparkSession.builder \
    .appName("Vitals Extraction") \
    .getOrCreate()
print("Extracting Vital Signs from CHARTEVENTS")
MIMIC_PATH = "/home/mahith/Datasets/mimic/physionet.org/files/mimiciv/3.1"
ICU = MIMIC_PATH + "/icu"
print("\nLoading CHARTEVENTS\n")
chartevents = spark.read.csv(
    ICU + "/chartevents.csv.gz",
    header=True,
    inferSchema=True
)
print("Chartevents Loaded Successfully")
print("\nFiltering Required Vital Signs\n")
vital_itemids = [
    220045,   # Heart Rate 
    220179,   # SBP 
    220180,   # DBP
    220181,   # MAP
    220210,   # Respiratory Rate
    223762,   # Temperature
    220277,   # SpO2
    220739,   # GCS Eye
    223900,   # GCS Verbal
    223901    # GCS Motor
]
vitals = chartevents.filter(
    col("itemid").isin(vital_itemids)
)
print("Vitals Filtered Successfully")
print("\nSelecting Correct Value Column\n")
vitals = vitals.select(
    "stay_id",
    "charttime",
    "itemid",
    col("valuenum").cast("double").alias("value")
).where(
    col("stay_id").isNotNull() &
    col("charttime").isNotNull() &
    col("valuenum").isNotNull()
)
print("Value Column Fixed (Using valuenum)")
print("\nMapping ItemIDs to Vital Names\n")
vitals = vitals.withColumn(
    "vital_sign",
    when(col("itemid") == 220045, "Heart Rate")
    .when(col("itemid") == 220179, "SBP")
    .when(col("itemid") == 220180, "DBP")
    .when(col("itemid") == 220181, "MAP")
    .when(col("itemid") == 220210, "Respiratory Rate")
    .when(col("itemid") == 223762, "Temperature")
    .when(col("itemid") == 220277, "SpO2")
    .when(col("itemid") == 220739, "GCS Eye")
    .when(col("itemid") == 223900, "GCS Verbal")
    .when(col("itemid") == 223901, "GCS Motor")
)
print("Mapping Completed")
vitals_dataset = vitals.select("stay_id", "charttime", "vital_sign", "value")
print("\nChecking Vital Names\n")
vitals_dataset.select("vital_sign").distinct().show()
print("\nTotal Records:", vitals_dataset.count())
print("\nSaving Vitals Dataset\n")
vitals_dataset.write.mode("overwrite") \
    .parquet("/home/mahith/BDA_PROJECT/processed_data/vitals_dataset.parquet")
print("Vitals Dataset Saved Successfully")
print("Completed Successfully")
spark.stop()
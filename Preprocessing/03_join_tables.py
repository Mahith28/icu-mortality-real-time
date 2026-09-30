from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("Join Core Tables") \
    .getOrCreate()
print("Joining Core MIMIC-IV Tables")
MIMIC_PATH = "/home/mahith/Datasets/mimic/physionet.org/files/mimiciv/3.1"
HOSP = MIMIC_PATH + "/hosp"
ICU = MIMIC_PATH + "/icu"
print("\nLoading MIMIC-IV Tables\n")
patients = spark.read.csv(
    HOSP + "/patients.csv.gz",
    header=True,
    inferSchema=True
)
admissions = spark.read.csv(
    HOSP + "/admissions.csv.gz",
    header=True,
    inferSchema=True
)
icustays = spark.read.csv(
    ICU + "/icustays.csv.gz",
    header=True,
    inferSchema=True
)
print("Patients Table Loaded")
print("Admissions Table Loaded")
print("ICU Stays Table Loaded")
patients = patients.select("subject_id", "gender", "anchor_age")
admissions = admissions.select("subject_id", "hadm_id", "admission_type", "admission_location", "insurance", "race", "hospital_expire_flag")
icustays = icustays.select("subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los")
master_dataset = patients \
    .join(admissions, on="subject_id", how="inner") \
    .join(icustays, on=["subject_id", "hadm_id"], how="inner")
master_dataset = master_dataset.withColumnRenamed("hospital_expire_flag", "mortality")
master_dataset = master_dataset.select("subject_id", "hadm_id", "stay_id", "gender", "anchor_age", "admission_type", "admission_location", "insurance", "race", "first_careunit", "last_careunit", "intime", "outtime", "los", "mortality")
duplicate_stays = master_dataset.groupBy("stay_id") \
    .count() \
    .filter("count > 1") \
    .count()
print("MASTER DATASET INFORMATION")
print(f"Number of Rows          : {master_dataset.count():,}")
print(f"Number of Columns       : {len(master_dataset.columns)}")
print(f"Duplicate ICU Stay IDs  : {duplicate_stays}")
print("MASTER DATASET SCHEMA")
master_dataset.printSchema()
print("MASTER DATASET COLUMNS")
for column in master_dataset.columns:
    print(column)
print("MASTER DATASET SAMPLE")
master_dataset.show(10, truncate=False)
print("\nSaving Master Dataset\n")
master_dataset.write \
    .mode("overwrite") \
    .parquet("/home/mahith/BDA_PROJECT/processed_data/master_dataset.parquet")
print("Master Dataset Saved Successfully")
print("\n")
print("Completed Successfully")
spark.stop()
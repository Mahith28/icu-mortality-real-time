from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("ICU Mortality Prediction") \
    .getOrCreate()
print("Spark Started Successfully!")
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
print("DATASET INFORMATION")
print(f"Patients Records   : {patients.count():,}")
print(f"Admissions Records : {admissions.count():,}")
print(f"ICU Stay Records   : {icustays.count():,}")
print("PATIENTS TABLE")
patients.show(5, truncate=False)
print("ADMISSIONS TABLE")
admissions.show(5, truncate=False)
print("ICUSTAYS TABLE")
icustays.show(5, truncate=False)
print("PATIENTS SCHEMA")
patients.printSchema()
print("ADMISSIONS SCHEMA")
admissions.printSchema()
print("ICUSTAYS SCHEMA")
icustays.printSchema()
print("\n")
print("Completed Successfully")
spark.stop()
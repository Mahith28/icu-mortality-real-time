from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.master("local[2]").appName("SubjectSplitAudit").config("spark.driver.memory", "6g").config("spark.sql.shuffle.partitions", "32").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
BASE = "hdfs://localhost:9000/user/mahith/icu"
EXPECTED_ROWS = {"TRAIN": 7_459_304, "VALIDATION": 1_610_439, "TEST": 1_664_239}
EXPECTED_TOTAL_ROWS = 10_733_982
EXPECTED_TOTAL_SUBJECTS = 65_366
EXPECTED_TOTAL_STAYS = 94_438
print("SUBJECT-LEVEL SPLIT AUDIT")
train = spark.read.parquet(f"{BASE}/train")
validation = spark.read.parquet(f"{BASE}/validation")
test = spark.read.parquet(f"{BASE}/test")
datasets = {"TRAIN": train, "VALIDATION": validation, "TEST": test}
print("\nCHECKING COLUMNS")
for name, df in datasets.items():
    print(f"{name} columns: {df.columns}")
print("\nCHECKING ROW COUNTS")
row_counts = {}
for name, df in datasets.items():
    row_counts[name] = df.count()
    print(f"{name}: {row_counts[name]:,}")
total_rows = sum(row_counts.values())
print(f"TOTAL: {total_rows:,}")
print("\nCHECKING SUBJECT COUNTS")
subject_counts = {}
for name, df in datasets.items():
    subject_counts[name] = df.select("subject_id").distinct().count()
    print(f"{name} subjects: {subject_counts[name]:,}")
total_subjects = sum(subject_counts.values())
print(f"TOTAL: {total_subjects:,}")
print("\nCHECKING STAY COUNTS")
stay_counts = {}
for name, df in datasets.items():
    stay_counts[name] = df.select("stay_id").distinct().count()
    print(f"{name} stays: {stay_counts[name]:,}")
total_stays = sum(stay_counts.values())
print(f"TOTAL: {total_stays:,}")
print("\nCHECKING NULL IDS")
null_subjects = {}
null_stays = {}
for name, df in datasets.items():
    null_subjects[name] = df.filter(F.col("subject_id").isNull()).count()
    null_stays[name] = df.filter(F.col("stay_id").isNull()).count()
    print(f"{name} null subject_id rows: {null_subjects[name]:,}")
    print(f"{name} null stay_id rows: {null_stays[name]:,}")
print("\nCHECKING SUBJECT OVERLAP")
train_subjects = train.select("subject_id").distinct()
validation_subjects = validation.select("subject_id").distinct()
test_subjects = test.select("subject_id").distinct()
train_validation = train_subjects.join(validation_subjects, "subject_id", "inner").count()
train_test = train_subjects.join(test_subjects, "subject_id", "inner").count()
validation_test = validation_subjects.join(test_subjects, "subject_id", "inner").count()
print(f"Train Validation subject overlap: {train_validation}")
print(f"Train Test subject overlap: {train_test}")
print(f"Validation Test subject overlap: {validation_test}")
print("\nCHECKING STAY OVERLAP")
train_stays = train.select("stay_id").distinct()
validation_stays = validation.select("stay_id").distinct()
test_stays = test.select("stay_id").distinct()
train_validation_stays = train_stays.join(validation_stays, "stay_id", "inner").count()
train_test_stays = train_stays.join(test_stays, "stay_id", "inner").count()
validation_test_stays = validation_stays.join(test_stays, "stay_id", "inner").count()
print(f"Train Validation stay overlap: {train_validation_stays}")
print(f"Train Test stay overlap: {train_test_stays}")
print(f"Validation Test stay overlap: {validation_test_stays}")
print("\nFINAL AUDIT")
rows_ok = all(row_counts[name] == EXPECTED_ROWS[name] for name in EXPECTED_ROWS)
row_conservation_ok = total_rows == EXPECTED_TOTAL_ROWS
subject_conservation_ok = total_subjects == EXPECTED_TOTAL_SUBJECTS
stay_conservation_ok = total_stays == EXPECTED_TOTAL_STAYS
null_subjects_ok = all(null_subjects[name] == 0 for name in null_subjects)
null_stays_ok = all(null_stays[name] == 0 for name in null_stays)
subject_overlap_ok = train_validation == 0 and train_test == 0 and validation_test == 0
stay_overlap_ok = train_validation_stays == 0 and train_test_stays == 0 and validation_test_stays == 0
all_checks_ok = rows_ok and row_conservation_ok and subject_conservation_ok and stay_conservation_ok and null_subjects_ok and null_stays_ok and subject_overlap_ok and stay_overlap_ok
print(f"Row counts: {'PASS' if rows_ok else 'FAIL'}")
print(f"Row conservation: {'PASS' if row_conservation_ok else 'FAIL'}")
print(f"Subject conservation: {'PASS' if subject_conservation_ok else 'FAIL'}")
print(f"Stay conservation: {'PASS' if stay_conservation_ok else 'FAIL'}")
print(f"Null subject_id check: {'PASS' if null_subjects_ok else 'FAIL'}")
print(f"Null stay_id check: {'PASS' if null_stays_ok else 'FAIL'}")
print(f"Subject overlap: {'PASS' if subject_overlap_ok else 'FAIL'}")
print(f"Stay overlap: {'PASS' if stay_overlap_ok else 'FAIL'}")
if all_checks_ok:
    print("\nFINAL RESULT: SUBJECT-LEVEL SPLIT AUDIT PASSED")
    print("SAFE TO PROCEED WITH CLEAN XGBOOST TRAINING")
else:
    print("\nFINAL RESULT: SUBJECT-LEVEL SPLIT AUDIT FAILED")
    print("DO NOT START XGBOOST TRAINING")
spark.stop()
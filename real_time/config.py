import os
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/mahith/BDA_PROJECT"
MODEL_DIR = os.path.join(BASE_DIR, "Model")
MIMIC_PATH = "/home/mahith/Datasets/mimic/physionet.org/files/mimiciv/3.1"
CHARTEVENTS_PATH = os.path.join(MIMIC_PATH, "icu", "chartevents.csv.gz")
MASTER_DATASET_PATH = os.path.join(BASE_DIR, "processed_data", "master_dataset.parquet")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "xgb_clean", "checkpoints", "icu_mortality_xgboost_clean.json")
XGB_BEST_ITERATION = 243
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_clean", "checkpoints", "lstm_adaptive_best.pt")
LSTM_IMPUTATION_PATH = "hdfs://localhost:9000/user/mahith/icu/lstm/artifacts/lstm_imputation_values.json"
LSTM_IMPUTATION_LOCAL_PATH = os.path.join(MODEL_DIR, "lstm_clean", "results", "lstm_imputation_values.json")
LSTM_NORMALIZATION_PATH = os.path.join(MODEL_DIR, "lstm_clean", "results", "lstm_normalization_stats.json")
CALIBRATION_PATH = os.path.join(MODEL_DIR, "ensemble", "results", "calibration_models.json")
ENSEMBLE_THRESHOLD_PATH = os.path.join(MODEL_DIR, "ensemble", "results", "locked_ensemble_threshold.json")
OFFLINE_TEST_PREDICTIONS_PATH = os.path.join(MODEL_DIR, "ensemble", "results", "final_test_predictions.parquet")
CLEAN_PIPELINE = "/home/mahith/BDA_PROJECT/Preprocessing/checkpoints/preprocessing_model_subject_safe"
TEST_SPLIT_PATH = "hdfs://localhost:9000/user/mahith/icu/test"
TEST_FEATURES_PATH = "hdfs://localhost:9000/user/mahith/icu/clean_features/test_features"
TEST_SEQUENCES_PATH = "hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/test_sequences"
TRAIN_FEATURES_PATH = "hdfs://localhost:9000/user/mahith/icu/clean_features/train_features"
VAL_FEATURES_PATH = "hdfs://localhost:9000/user/mahith/icu/clean_features/validation_features"
VAL_SEQUENCES_PATH = "hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/validation_sequences"
ARTIFACT_DIR = os.path.join(MODEL_DIR, "real_time", "artifacts")
INFERENCE_BASELINE_PATH = os.path.join(ARTIFACT_DIR, "inference_baseline.json")
FEATURE_COUNT = 137
TEMPORAL_FEATURES = 54
STATIC_FEATURES = 83
ASSEMBLER_INPUT_COUNT = 61
LSTM_INPUT_SIZE = 108
MAX_SEQ_LEN = 120
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_STATIC_HIDDEN = 64
LSTM_FUSION_HIDDEN = 64
DROPOUT = 0.30
XGB_WEIGHT = 0.48
LSTM_WEIGHT = 0.52
ENSEMBLE_THRESHOLD = 0.108
XGB_PLATT_INTERCEPT = -1.6827572668493167
XGB_PLATT_COEFFICIENT = 1.2964222172584006
LSTM_PLATT_INTERCEPT = -1.4879018224914522
LSTM_PLATT_COEFFICIENT = 0.6916860069585927
WINDOW_MINUTES = 5
WATERMARK_MINUTES = 5
STATE_TIMEOUT_HOURS = 6
ALERT_COOLDOWN_WINDOWS = 3
KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
OBSERVATION_TOPIC = "icu_observations"
STATIC_TOPIC = "icu_static"
RISK_TOPIC = "risk_scores"
EXPLANATION_TOPIC = "risk_explanations"
SPARK_KAFKA_PACKAGE = "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "real_time", "checkpoints")
STATE_DIR = os.path.join(MODEL_DIR, "real_time", "state")
FAST_STORE_PATH = os.path.join(STATE_DIR, "current_state.db")
HDFS_OUTPUT_PATH = "hdfs://localhost:9000/user/mahith/icu/real_time"
HDFS_RISK_PATH = HDFS_OUTPUT_PATH + "/risk_scores"
HDFS_OBSERVATION_PATH = HDFS_OUTPUT_PATH + "/observations"
HDFS_EXPLANATION_PATH = HDFS_OUTPUT_PATH + "/explanations"
REPLAY_SPEED = 600
VITAL_ITEM_IDS = {
    220045: "Heart_Rate",
    220179: "SBP",
    220180: "DBP",
    220181: "MAP",
    220210: "Respiratory_Rate",
    223762: "Temperature",
    220277: "SpO2",
    220739: "GCS_Eye",
    223900: "GCS_Verbal",
    223901: "GCS_Motor"
}
REALTIME_VITAL_NAME_MAP = {
    "heart_rate": "Heart Rate",
    "sbp": "SBP",
    "dbp": "DBP",
    "map": "MAP",
    "resp_rate": "Respiratory Rate",
    "temperature": "Temperature",
    "spo2": "SpO2",
    "gcs_eye": "GCS Eye",
    "gcs_verbal": "GCS Verbal",
    "gcs_motor": "GCS Motor"
}
REALTIME_VITAL_VALIDATION_RANGES = {
    "Heart Rate": (20.0, 250.0),
    "SBP": (40.0, 300.0),
    "DBP": (20.0, 200.0),
    "MAP": (20.0, 250.0),
    "Respiratory Rate": (4.0, 100.0),
    "Temperature": (25.0, 45.0),
    "SpO2": (50.0, 100.0),
    "GCS Eye": (1.0, 4.0),
    "GCS Verbal": (1.0, 5.0),
    "GCS Motor": (1.0, 6.0)
}
REALTIME_VITAL_SIGNS = tuple(REALTIME_VITAL_NAME_MAP.values())
TEMPORAL_FEATURE_NAMES = [
    "Heart_Rate_mean",
    "Heart_Rate_min",
    "Heart_Rate_max",
    "Heart_Rate_last",
    "SBP_mean",
    "SBP_min",
    "SBP_max",
    "SBP_last",
    "DBP_mean",
    "DBP_min",
    "DBP_max",
    "DBP_last",
    "MAP_mean",
    "MAP_min",
    "MAP_max",
    "MAP_last",
    "Respiratory_Rate_mean",
    "Respiratory_Rate_min",
    "Respiratory_Rate_max",
    "Respiratory_Rate_last",
    "Temperature_mean",
    "Temperature_min",
    "Temperature_max",
    "Temperature_last",
    "SpO2_mean",
    "SpO2_min",
    "SpO2_max",
    "SpO2_last",
    "GCS_Eye_mean",
    "GCS_Eye_min",
    "GCS_Eye_max",
    "GCS_Eye_last",
    "GCS_Verbal_mean",
    "GCS_Verbal_min",
    "GCS_Verbal_max",
    "GCS_Verbal_last",
    "GCS_Motor_mean",
    "GCS_Motor_min",
    "GCS_Motor_max",
    "GCS_Motor_last",
    "Heart_Rate_missing",
    "SBP_missing",
    "DBP_missing",
    "MAP_missing",
    "Respiratory_Rate_missing",
    "Temperature_missing",
    "SpO2_missing",
    "GCS_Eye_missing",
    "GCS_Verbal_missing",
    "GCS_Motor_missing",
    "HR_trend",
    "MAP_trend",
    "RR_trend",
    "GCS_Total_last"
]
EXPECTED_REALTIME_VITALS = {
    "Heart Rate",
    "SBP",
    "DBP",
    "MAP",
    "Respiratory Rate",
    "Temperature",
    "SpO2",
    "GCS Eye",
    "GCS Verbal",
    "GCS Motor"
}
if set(REALTIME_VITAL_NAME_MAP.values()) != EXPECTED_REALTIME_VITALS:
    raise RuntimeError("Realtime vital mapping does not match expected vital names")
if set(REALTIME_VITAL_VALIDATION_RANGES) != EXPECTED_REALTIME_VITALS:
    raise RuntimeError("Realtime vital ranges do not match expected vital names")
for display_name in REALTIME_VITAL_SIGNS:
    feature_name = display_name.replace(" ", "_")
    if feature_name not in VITAL_ITEM_IDS.values():
        raise RuntimeError(f"Realtime vital contract mismatch: {display_name}")
for vital, (low, high) in REALTIME_VITAL_VALIDATION_RANGES.items():
    if low >= high:
        raise RuntimeError(f"Invalid realtime validation range: {vital}")
EXPECTED_PRODUCER_VITALS = {
    "heart_rate",
    "sbp",
    "dbp",
    "resp_rate",
    "spo2",
    "temperature"
}
if not EXPECTED_PRODUCER_VITALS.issubset(REALTIME_VITAL_NAME_MAP):
    raise RuntimeError("Producer vital contract does not match realtime vital mapping")
STATIC_FEATURE_START = 54
STATIC_FEATURE_END = 137
REALTIME_TEMPERATURE_UNIT = "C"
if len(TEMPORAL_FEATURE_NAMES) != TEMPORAL_FEATURES:
    raise RuntimeError("Temporal feature count does not match configuration")
if TEMPORAL_FEATURES + STATIC_FEATURES != FEATURE_COUNT:
    raise RuntimeError("Feature count does not match configuration")
if LSTM_INPUT_SIZE != TEMPORAL_FEATURES * 2:
    raise RuntimeError("LSTM input size does not match configuration")
if STATIC_FEATURE_START != TEMPORAL_FEATURES:
    raise RuntimeError("Static feature start does not match temporal feature count")
if STATIC_FEATURE_END != FEATURE_COUNT:
    raise RuntimeError("Static feature end does not match feature count")
if ASSEMBLER_INPUT_COUNT != 61:
    raise RuntimeError("VectorAssembler input count does not match configuration")
if XGB_BEST_ITERATION < 0:
    raise RuntimeError("XGBoost best iteration must be non-negative")
if abs((XGB_WEIGHT + LSTM_WEIGHT) - 1.0) > 1e-12:
    raise RuntimeError("Ensemble weights must sum to 1")
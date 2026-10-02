import os
import sys
import time
import copy
import importlib.util
import json
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.fs as pafs
import pyarrow.dataset as ds
import pyarrow.compute as pc
REAL_TIME_DIR = os.path.dirname(os.path.abspath(__file__))
classpath = os.popen("hadoop classpath --glob").read().strip()
if not classpath:
    raise RuntimeError("Hadoop CLASSPATH is empty")
os.environ["CLASSPATH"] = classpath
import config as C
import artifact
spec = importlib.util.spec_from_file_location("realtime_inference", os.path.join(REAL_TIME_DIR, "02_realtime_inference.py"))
realtime_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(realtime_module)
RealtimeInference = realtime_module.RealtimeInference
ORCHESTRATOR_PATH = os.path.join(REAL_TIME_DIR, "08_realtime_orchestrator.py")
orchestrator_spec = importlib.util.spec_from_file_location("realtime_orchestrator", ORCHESTRATOR_PATH)
orchestrator_module = importlib.util.module_from_spec(orchestrator_spec)
orchestrator_spec.loader.exec_module(orchestrator_module)
RealtimeOrchestrator = orchestrator_module.RealtimeOrchestrator
preprocessing = importlib.util.spec_from_file_location("realtime_preprocessing", os.path.join(REAL_TIME_DIR, "01_realtime_preprocessing.py"))
preprocessing_module = importlib.util.module_from_spec(preprocessing)
preprocessing.loader.exec_module(preprocessing_module)
RealtimePreprocessor = preprocessing_module.RealtimePreprocessor
PROJECT_ROOT = "/home/mahith/BDA_PROJECT"
TEMPORAL_FEATURE_NAMES = preprocessing_module.TEMPORAL_FEATURE_NAMES
def load_parity_data(stay_id):
    window_columns = ["stay_id", "window_id"] + TEMPORAL_FEATURE_NAMES + ["HR_trend", "MAP_trend", "RR_trend", "GCS_Total_last", "anchor_age", "gender", "admission_type", "admission_location", "insurance", "race", "first_careunit", "intime"]
    window_dataset = ds.dataset(os.path.join(PROJECT_ROOT, "processed_data/window_dataset.parquet"), format="parquet")
    vitals_dataset = ds.dataset(os.path.join(PROJECT_ROOT, "processed_data/vitals_dataset.parquet"), format="parquet")
    master_dataset = ds.dataset(os.path.join(PROJECT_ROOT, "processed_data/master_dataset.parquet"), format="parquet")
    window_table = window_dataset.to_table(columns=window_columns, filter=pc.field("stay_id") == stay_id)
    vitals_table = vitals_dataset.to_table(columns=["stay_id", "charttime", "vital_sign", "value"], filter=pc.field("stay_id") == stay_id)
    master_table = master_dataset.to_table(columns=["stay_id", "anchor_age", "gender", "admission_type", "admission_location", "insurance", "race", "first_careunit", "intime"], filter=pc.field("stay_id") == stay_id)
    return window_table.to_pandas(), vitals_table.to_pandas(), master_table.to_pandas()
logger = logging.getLogger("parity_test")
patient_state = importlib.util.spec_from_file_location("patient_state", os.path.join(REAL_TIME_DIR, "04_patient_state.py"))
patient_state_module = importlib.util.module_from_spec(patient_state)
patient_state.loader.exec_module(patient_state_module)
PatientStateManager = patient_state_module.PatientStateManager
change_detection = importlib.util.spec_from_file_location("change_detection", os.path.join(REAL_TIME_DIR, "07_change_detection.py"))
change_detection_module = importlib.util.module_from_spec(change_detection)
change_detection.loader.exec_module(change_detection_module)
ChangeDetector = change_detection_module.ChangeDetector
AlertLogger = change_detection_module.AlertLogger
SEQUENCE_PATH = "/user/mahith/icu/lstm_adaptive/test_sequences"
FEATURE_PATH = "/user/mahith/icu/clean_features/test_features"
REFERENCE_PATH = "/home/mahith/BDA_PROJECT/ensemble/results/test_calibrated_predictions.parquet"
FINAL_REFERENCE_PATH = "/home/mahith/BDA_PROJECT/ensemble/results/final_test_predictions.parquet"
SAMPLE_ROWS = 3000
NEAR_THRESHOLD_ROWS = min(100, SAMPLE_ROWS // 10)
SEQUENCE_ID_BATCH_SIZE = 4096
SEQUENCE_BATCH_SIZE = 64
FEATURE_ID_BATCH_SIZE = 8192
FEATURE_BATCH_SIZE = 512
XGB_TOLERANCE = 1e-6
XGB_CALIBRATED_TOLERANCE = 1e-6
LSTM_LOGIT_TOLERANCE = 6e-3
LSTM_RAW_TOLERANCE = 1.5e-3
LSTM_CALIBRATED_TOLERANCE = 5e-4
ENSEMBLE_TOLERANCE = 2.6e-4
ENSEMBLE_WEIGHT_TOLERANCE = 1e-6
ENSEMBLE_THRESHOLD = C.ENSEMBLE_THRESHOLD
REPORT_ONLY = os.environ.get("PARITY_REPORT_ONLY", "0") == "1"
def get_hdfs():
    return pafs.HadoopFileSystem("localhost", 9000)
def find_parquet_files(fs, path):
    selector = pafs.FileSelector(path, recursive=False)
    infos = fs.get_file_info(selector)
    files = []
    for info in infos:
        if info.type == pafs.FileType.File and info.path.endswith(".parquet"):
            files.append(info.path)
    return sorted(files)
def read_reference():
    columns = [
        "stay_id",
        "window_id",
        "label",
        "xgb_probability",
        "lstm_probability",
        "xgb_calibrated_probability",
        "lstm_calibrated_probability",
        "ensemble_probability"
    ]
    return pq.read_table(REFERENCE_PATH, columns=columns).to_pandas()
def read_final_reference():
    columns = ["stay_id", "window_id", "ensemble_prediction"]
    return pq.read_table(FINAL_REFERENCE_PATH, columns=columns).to_pandas()
def prepare_reference(reference):
    reference = reference.copy()
    reference["stay_id"] = reference["stay_id"].astype(np.int64)
    reference["window_id"] = reference["window_id"].astype(np.int64)
    reference["label"] = reference["label"].astype(np.int8)
    reference["ensemble_expected"] = C.XGB_WEIGHT * reference["xgb_calibrated_probability"] + C.LSTM_WEIGHT * reference["lstm_calibrated_probability"]
    reference["ensemble_difference"] = abs(reference["ensemble_probability"] - reference["ensemble_expected"])
    return reference
def composite_keys(stay_id, window_id):
    stay_id = np.asarray(stay_id, dtype=np.int64)
    window_id = np.asarray(window_id, dtype=np.int64)
    if np.any(window_id < 0) or np.any(window_id >= 1000000):
        raise RuntimeError("window_id must be in [0, 1000000) for composite keys")
    return stay_id * 1000000 + window_id
def check_reference_weights(reference):
    maximum = float(reference["ensemble_difference"].max())
    print(f"Reference ensemble maximum difference: {maximum:.10f}")
    if maximum > ENSEMBLE_WEIGHT_TOLERANCE:
        raise RuntimeError(f"Reference ensemble calculation mismatch: {maximum:.10f}")
    print("Reference ensemble calculation is consistent")
def select_near_threshold_keys(reference):
    threshold_distance = abs(reference["ensemble_probability"].to_numpy(np.float64) - ENSEMBLE_THRESHOLD)
    count = min(NEAR_THRESHOLD_ROWS, len(reference))
    positions = np.argsort(threshold_distance)[:count]
    selected = reference.iloc[positions].copy()
    selected["key"] = composite_keys(selected["stay_id"].to_numpy(), selected["window_id"].to_numpy())
    return selected
def read_sequence_row_group_ids(fs, sequence_files, near_keys, reference_keys):
    print("sequence row-group id scan")
    row_groups = []
    near_locations = {}
    group_index = 0
    total_seconds = 0.0
    all_sequence_keys = []
    for path in sequence_files:
        pf = pq.ParquetFile(path, filesystem=fs)
        for rg in range(pf.metadata.num_row_groups):
            start = time.perf_counter()
            table = pf.read_row_group(rg, columns=["stay_id", "terminal_window_id"])
            stay_id = table["stay_id"].to_numpy(zero_copy_only=False)
            window_id = table["terminal_window_id"].to_numpy(zero_copy_only=False)
            keys = composite_keys(stay_id, window_id)
            all_sequence_keys.append(keys)
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            first_count = min(64, len(keys))
            sample_positions = np.unique(np.linspace(0, len(keys) - 1, first_count, dtype=np.int64)) if len(keys) else np.array([], dtype=np.int64)
            row_groups.append({
                "file": path,
                "row_group": rg,
                "rows": len(keys),
                "sample_keys": keys[sample_positions],
                "index": group_index
            })
            near_positions = np.flatnonzero(np.isin(keys, near_keys))
            for position in near_positions:
                near_locations[int(keys[position])] = (path, rg)
            print(f"Sequence ID scan file={os.path.basename(path)} rg={rg} rows={len(keys)} seconds={elapsed:.3f}")
            group_index += 1
    print(f"Sequence row groups: {len(row_groups)}")
    print(f"Sequence ID scan seconds: {total_seconds:.3f}")
    sequence_keys = np.concatenate(all_sequence_keys)
    if len(sequence_keys) != len(reference_keys) or not np.array_equal(np.sort(sequence_keys), np.sort(reference_keys)):
        raise RuntimeError("Sequence composite keys do not exactly match reference keys")
    print("Sequence/reference key-set check: PASS")
    missing_near = set(near_keys.tolist()).difference(near_locations.keys())
    if missing_near:
        raise RuntimeError(f"Near-threshold sequence keys not found: {len(missing_near)}")
    return row_groups, near_locations
def choose_sequence_row_groups(row_groups, near_locations):
    if not row_groups:
        raise RuntimeError("No sequence row groups found")
    base_target = max(1, int(np.ceil((SAMPLE_ROWS - NEAR_THRESHOLD_ROWS) / 64)))
    step = max(1, int(np.ceil(len(row_groups) / base_target)))
    base_indices = list(range(0, len(row_groups), step))
    selected_indices = set(base_indices)
    near_pairs = set(near_locations.values())
    pair_to_index = {(item["file"], item["row_group"]): item["index"] for item in row_groups}
    for pair in near_pairs:
        selected_indices.add(pair_to_index[pair])
    selected = [row_groups[index] for index in sorted(selected_indices)]
    print(f"Selected sequence row groups: {len(selected)}")
    print(f"Base row-group step: {step}")
    print(f"Near-threshold row groups added: {len(near_pairs)}")
    return selected, set(base_indices)
def sparse_features_to_numpy(v):
    if v is None:
        raise RuntimeError("Missing feature vector")
    size = int(v["size"])
    if size != C.FEATURE_COUNT:
        raise RuntimeError(f"Expected {C.FEATURE_COUNT} features, found {size}")
    vector_type = int(v["type"])
    if vector_type == 0:
        dense = np.zeros(size, dtype=np.float32)
        if v["indices"] is not None and len(v["indices"]) > 0:
            dense[np.asarray(v["indices"], dtype=np.int64)] = np.asarray(v["values"], dtype=np.float32)
        return dense
    if vector_type == 1:
        return np.asarray(v["values"], dtype=np.float32)
    raise RuntimeError(f"Unknown vector type: {vector_type}")
def read_matching_features(fs, feature_files, sample_keys):
    columns = ["stay_id", "window_id", "features", "label"]
    target_keys = set(sample_keys.tolist())
    feature_locations = {}
    total_seconds = 0.0
    print("feature row-group id scan")
    for path in feature_files:
        pf = pq.ParquetFile(path, filesystem=fs)
        for rg in range(pf.metadata.num_row_groups):
            start = time.perf_counter()
            table = pf.read_row_group(rg, columns=["stay_id", "window_id"])
            keys = composite_keys(table["stay_id"].to_numpy(zero_copy_only=False), table["window_id"].to_numpy(zero_copy_only=False))
            elapsed = time.perf_counter() - start
            total_seconds += elapsed
            positions = np.flatnonzero(np.isin(keys, sample_keys))
            if len(positions):
                feature_locations[(path, rg)] = positions
            print(f"Feature ID scan file={os.path.basename(path)} rg={rg} rows={len(keys)} matches={len(positions)} seconds={elapsed:.3f}")
    print(f"Feature ID scan seconds: {total_seconds:.3f}")
    if not feature_locations:
        raise RuntimeError("No feature row groups contain sampled keys")
    feature_lookup = {}
    target_keys = set(sample_keys.tolist())
    print("feature heavy-column reads")
    for (path, rg), positions in feature_locations.items():
        start = time.perf_counter()
        pf = pq.ParquetFile(path, filesystem=fs)
        table = pf.read_row_group(rg, columns=columns)
        keys = composite_keys(table["stay_id"].to_numpy(zero_copy_only=False), table["window_id"].to_numpy(zero_copy_only=False))
        mask = np.isin(keys, sample_keys)
        filtered = table.filter(pa.array(mask))
        df = filtered.to_pandas()
        elapsed = time.perf_counter() - start
        print(f"Feature heavy read file={os.path.basename(path)} rg={rg} matched={len(df)} seconds={elapsed:.3f}")
        filtered_keys = composite_keys(df["stay_id"].to_numpy(np.int64), df["window_id"].to_numpy(np.int64))
        for position, key in enumerate(filtered_keys):
            key = int(key)
            if key in feature_lookup:
                raise RuntimeError(f"Duplicate feature key: {key}")
            feature_lookup[key] = sparse_features_to_numpy(df.iloc[position]["features"])
    missing = target_keys.difference(feature_lookup.keys())
    if missing:
        raise RuntimeError(f"Missing XGBoost feature rows: {len(missing)}")
    print(f"Matching XGBoost feature rows: {len(feature_lookup)}")
    return feature_lookup
def read_matching_sequences(fs, selected_row_groups, sample_keys):
    columns = [
        "stay_id",
        "terminal_window_id",
        "sequence_length",
        "label",
        "temporal",
        "missing_mask",
        "padding_mask",
        "static"
    ]
    target_keys = set(sample_keys.tolist())
    sequence_rows = []
    total_seconds = 0.0
    print("sequence heavy-column reads")
    for group in selected_row_groups:
        start = time.perf_counter()
        pf = pq.ParquetFile(group["file"], filesystem=fs)
        batch_count = 0
        matched_count = 0
        for batch in pf.iter_batches(batch_size=SEQUENCE_BATCH_SIZE, columns=columns, row_groups=[group["row_group"]]):
            batch_count += 1
            ids = composite_keys(batch["stay_id"].to_numpy(zero_copy_only=False), batch["terminal_window_id"].to_numpy(zero_copy_only=False))
            mask = np.isin(ids, sample_keys)
            if not mask.any():
                continue
            filtered = batch.filter(pa.array(mask))
            df = filtered.to_pandas()
            matched_count += len(df)
            for position in range(len(df)):
                key = int(composite_keys(np.array([int(df.iloc[position]["stay_id"])]), np.array([int(df.iloc[position]["terminal_window_id"])]))[0])
                if key not in target_keys:
                    continue
                sequence_rows.append(df.iloc[position])
                target_keys.remove(key)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        print(f"Sequence heavy read file={os.path.basename(group['file'])} rg={group['row_group']} batches={batch_count} matched={matched_count} seconds={elapsed:.3f}")
    if target_keys:
        raise RuntimeError(f"Missing sequence rows: {len(target_keys)}")
    result = pd.DataFrame(sequence_rows)
    pair_keys = composite_keys(result["stay_id"].to_numpy(np.int64), result["terminal_window_id"].to_numpy(np.int64))
    if pd.Series(pair_keys).duplicated().any():
        raise RuntimeError("Duplicate sequence composite keys found")
    print(f"Matching sequence rows: {len(result)}")
    print(f"Sequence heavy-read seconds: {total_seconds:.3f}")
    return result
def convert_sequence(row):
    temporal = np.stack([np.asarray(values, dtype=np.float32) for values in row["temporal"]])
    missing_mask = np.stack([np.asarray(values, dtype=np.float32) for values in row["missing_mask"]])
    padding_mask = np.asarray(row["padding_mask"], dtype=np.float32)
    static = np.asarray(row["static"], dtype=np.float32)
    sequence_length = int(row["sequence_length"])
    if temporal.shape != (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES):
        raise RuntimeError(f"Invalid temporal shape: {temporal.shape}")
    if missing_mask.shape != (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES):
        raise RuntimeError(f"Invalid missing mask shape: {missing_mask.shape}")
    if padding_mask.shape != (C.MAX_SEQ_LEN,):
        raise RuntimeError(f"Invalid padding mask shape: {padding_mask.shape}")
    if static.shape != (C.STATIC_FEATURES,):
        raise RuntimeError(f"Invalid static shape: {static.shape}")
    if sequence_length <= 0 or sequence_length > C.MAX_SEQ_LEN:
        raise RuntimeError(f"Invalid sequence length: {sequence_length}")
    if not np.all(np.isin(padding_mask, [0, 1])):
        raise RuntimeError("Padding mask contains invalid values")
    if not np.all(padding_mask[:sequence_length] == 0):
        raise RuntimeError("Valid sequence rows are not contiguous at the beginning")
    if not np.all(padding_mask[sequence_length:] == 1):
        raise RuntimeError("Padding rows are not at the end")
    if sequence_length < C.MAX_SEQ_LEN:
        if not np.all(temporal[sequence_length:] == 0):
            raise RuntimeError("Padded temporal rows are not zero")
        if not np.all(missing_mask[sequence_length:] == 0):
            raise RuntimeError("Padded missing-mask rows are not zero")
    if not np.all(np.isfinite(temporal[:sequence_length])):
        raise RuntimeError("Valid temporal values contain non-finite values")
    if not np.all(np.isfinite(static)):
        raise RuntimeError("Static values contain non-finite values")
    return temporal, missing_mask, padding_mask, static
def calculate_missingness(missing_mask, sequence_length):
    if sequence_length <= 0:
        return 0.0
    return float(np.mean(missing_mask[:sequence_length]))
def compare_value(offline, realtime, tolerance):
    difference = abs(float(offline) - float(realtime))
    return difference, difference <= tolerance
def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99.9": float(np.percentile(values, 99.9)),
        "max": float(np.max(values))
    }
def print_percentile_line(name, values):
    summary = percentile_summary(values)
    print(f"{name}: p50={summary['p50']:.10f} p95={summary['p95']:.10f} p99={summary['p99']:.10f} p99.9={summary['p99.9']:.10f} max={summary['max']:.10f}")
def print_group_statistics(valid, title, mask):
    group = valid[mask]
    print(f"{title} rows: {len(group)}")
    if group.empty:
        return
    for name in ["xgb_diff", "xgb_calibrated_diff", "lstm_logit_diff", "lstm_diff", "lstm_calibrated_diff", "ensemble_diff"]:
        print_percentile_line(name, group[name])
def test_preprocessing_parity():
    stay_id = 30000646
    offline_windows, vitals, master = load_parity_data(stay_id)
    if master.empty:
        raise RuntimeError(f"No master data found for stay {stay_id}")
    if offline_windows.empty:
        raise RuntimeError(f"No offline windows found for stay {stay_id}")
    master_row = master.iloc[0]
    intime = master_row["intime"]
    static_data = {
        "anchor_age": float(master_row["anchor_age"]),
        "gender": master_row["gender"],
        "race": master_row["race"],
        "insurance": master_row["insurance"],
        "admission_type": master_row["admission_type"],
        "admission_location": master_row["admission_location"],
        "first_careunit": master_row["first_careunit"]
    }
    vitals = vitals.sort_values("charttime").reset_index(drop=True)
    offline_windows = offline_windows.sort_values("window_id").reset_index(drop=True)
    preprocessor = RealtimePreprocessor()
    preprocessor.start()
    realtime_results = []
    for window_id in offline_windows["window_id"].astype(int):
        window_start = pd.Timestamp(intime) + pd.Timedelta(minutes=int(window_id) * C.WINDOW_MINUTES)
        window_end = window_start + pd.Timedelta(minutes=C.WINDOW_MINUTES)
        window_vitals = vitals[(vitals["charttime"] >= window_start) & (vitals["charttime"] < window_end)]
        for _, observation in window_vitals.iterrows():
            preprocessor.add_observation(
                stay_id,
                observation["charttime"],
                intime,
                observation["vital_sign"],
                observation["value"]
            )
        result = preprocessor.get_final_window(stay_id, int(window_id), static_data)
        realtime_results.append(result)
    if len(realtime_results) != len(offline_windows):
        raise RuntimeError(f"Realtime/offline window count mismatch: {len(realtime_results)} != {len(offline_windows)}")
    temporal_differences = []
    missing_matches = 0
    static_differences = []
    for offline_row, realtime_result in zip(offline_windows.itertuples(index=False), realtime_results):
        offline_temporal = np.asarray([getattr(offline_row, name) for name in TEMPORAL_FEATURE_NAMES], dtype=np.float64)
        realtime_temporal = np.asarray(realtime_result["features"][:C.TEMPORAL_FEATURES], dtype=np.float64)
        offline_missing = np.asarray([getattr(offline_row, name) for name in TEMPORAL_FEATURE_NAMES if name.endswith("_missing")], dtype=np.float64)
        realtime_missing = np.asarray([realtime_result["features"][TEMPORAL_FEATURE_NAMES.index(name)] for name in TEMPORAL_FEATURE_NAMES if name.endswith("_missing")], dtype=np.float64)
        finite_mask = np.isfinite(offline_temporal) & np.isfinite(realtime_temporal)
        if finite_mask.any():
            temporal_differences.extend(np.abs(offline_temporal[finite_mask] - realtime_temporal[finite_mask]).tolist())
        if np.array_equal(offline_missing, realtime_missing):
            missing_matches += 1
        if not np.allclose(offline_temporal, realtime_temporal, rtol=0.0, atol=1e-6, equal_nan=True):
            differences = np.where(~np.isclose(offline_temporal, realtime_temporal, rtol=0.0, atol=1e-6, equal_nan=True))[0]
            names = [TEMPORAL_FEATURE_NAMES[index] for index in differences]
            raise RuntimeError(f"Preprocessing mismatch at window {int(offline_row.window_id)}: {names}")
    max_temporal_difference = max(temporal_differences) if temporal_differences else 0.0
    if missing_matches != len(offline_windows):
        raise RuntimeError(f"Missing-mask parity failed: {missing_matches}/{len(offline_windows)}")
    print(f"Preprocessing parity stay: {stay_id}")
    print(f"Offline windows checked: {len(offline_windows)}")
    print(f"Missing-mask matches: {missing_matches}/{len(offline_windows)}")
    print(f"Maximum temporal difference: {max_temporal_difference:.10f}")
    print("OFFLINE-REALTIME PREPROCESSING PARITY PASSED")
def main():
    test_preprocessing_parity()
    print("REAL-TIME STAGE A STREAMING PARITY SAMPLE")
    print(f"Feature count: {C.FEATURE_COUNT}")
    print(f"Temporal features: {C.TEMPORAL_FEATURES}")
    print(f"Static features: {C.STATIC_FEATURES}")
    print(f"Maximum sequence length: {C.MAX_SEQ_LEN}")
    print(f"Ensemble threshold: {ENSEMBLE_THRESHOLD}")
    print(f"Sample target rows: {SAMPLE_ROWS}")
    print(f"Near-threshold rows: {NEAR_THRESHOLD_ROWS}")
    print(f"Device: {C.DEVICE}")
    print(f"PyTorch version: {artifact.torch.__version__}")
    print(f"CUDA available: {artifact.torch.cuda.is_available()}")
    if artifact.torch.cuda.is_available():
        print(f"CUDA version: {artifact.torch.version.cuda}")
        print(f"cuDNN version: {artifact.torch.backends.cudnn.version()}")
        print(f"TF32 matmul allowed: {artifact.torch.backends.cuda.matmul.allow_tf32}")
        print(f"TF32 cuDNN allowed: {artifact.torch.backends.cudnn.allow_tf32}")
    if C.DEVICE.type != "cuda":
        raise RuntimeError("C.DEVICE must be CUDA for this parity run")
    if abs(C.ENSEMBLE_THRESHOLD - 0.108) >= 1e-12:
        raise RuntimeError(f"Unexpected ensemble threshold: {C.ENSEMBLE_THRESHOLD}")
    threshold_artifact_path = os.path.join(REAL_TIME_DIR, "..", "..", "ensemble", "results", "locked_ensemble_threshold.json")
    with open(threshold_artifact_path, "r") as file:
        threshold_artifact = json.load(file)
    locked_threshold = float(threshold_artifact["threshold"])
    if abs(locked_threshold - C.ENSEMBLE_THRESHOLD) >= 1e-12:
        raise RuntimeError(f"Locked threshold mismatch: artifact={locked_threshold} config={C.ENSEMBLE_THRESHOLD}")
    print(f"Locked threshold check: PASS ({locked_threshold})")
    if os.environ.get("PARITY_TF32", "1") == "0":
        artifact.torch.backends.cudnn.allow_tf32 = False
        artifact.torch.backends.cuda.matmul.allow_tf32 = False
    print(f"cudnn.allow_tf32 in effect: {artifact.torch.backends.cudnn.allow_tf32}")
    fs = get_hdfs()
    sequence_files = find_parquet_files(fs, SEQUENCE_PATH)
    feature_files = find_parquet_files(fs, FEATURE_PATH)
    print(f"Sequence files found: {len(sequence_files)}")
    print(f"Feature files found: {len(feature_files)}")
    reference = prepare_reference(read_reference())
    final_reference = read_final_reference()
    print(f"Reference rows: {len(reference)}")
    print(f"Final reference rows: {len(final_reference)}")
    if len(reference) != len(final_reference):
        raise RuntimeError("Reference and final reference row counts differ")
    check_reference_weights(reference)
    reference["key"] = composite_keys(reference["stay_id"].to_numpy(), reference["window_id"].to_numpy())
    if reference["key"].duplicated().any():
        raise RuntimeError("Duplicate keys found in reference")
    final_reference["stay_id"] = final_reference["stay_id"].astype(np.int64)
    final_reference["window_id"] = final_reference["window_id"].astype(np.int64)
    final_reference["key"] = composite_keys(final_reference["stay_id"].to_numpy(), final_reference["window_id"].to_numpy())
    if final_reference["key"].duplicated().any():
        raise RuntimeError("Duplicate keys found in final reference")
    near_reference = select_near_threshold_keys(reference)
    near_keys = near_reference["key"].to_numpy(np.int64)
    near_key_set = set(near_keys.tolist())
    reference_keys = reference["key"].to_numpy(np.int64)
    sequence_row_groups, near_locations = read_sequence_row_group_ids(fs, sequence_files, near_keys, reference_keys)
    selected_row_groups, base_indices = choose_sequence_row_groups(sequence_row_groups, near_locations)
    sample_keys = []
    sample_sources = {}
    for index in sorted(base_indices):
        group = sequence_row_groups[index]
        for key in group["sample_keys"]:
            key = int(key)
            if key not in sample_sources:
                sample_keys.append(key)
                sample_sources[key] = "layout_sample"
    for key in near_keys:
        key = int(key)
        if key not in sample_sources:
            sample_keys.append(key)
            sample_sources[key] = "near_threshold"
    sample_keys = np.asarray(sample_keys, dtype=np.int64)
    if len(sample_keys) == 0:
        raise RuntimeError("No sample keys selected")
    if len(set(sample_keys.tolist())) != len(sample_keys):
        raise RuntimeError("Duplicate sample composite keys found")
    sample_reference = reference[reference["key"].isin(sample_keys)].copy()
    if len(sample_reference) != len(sample_keys):
        raise RuntimeError("Sample keys missing from reference")
    print(f"Selected sample rows: {len(sample_reference)}")
    print(f"Layout sample rows: {(sample_reference['key'].map(sample_sources) == 'layout_sample').sum()}")
    print(f"Exact near-threshold rows: {(sample_reference['key'].map(sample_sources) == 'near_threshold').sum()}")
    final_sample = final_reference[final_reference["key"].isin(sample_keys)].copy()
    if len(final_sample) != len(sample_reference):
        raise RuntimeError("Missing final reference rows in sample")
    feature_lookup = read_matching_features(fs, feature_files, sample_keys)
    sequence_df = read_matching_sequences(fs, selected_row_groups, sample_keys)
    reference_lookup = sample_reference.set_index("key")
    final_lookup = final_sample.set_index("key")
    print("Starting real-time inference")
    inference = RealtimeInference()
    inference.start()
    if os.environ.get("PARITY_TF32", "1") == "0":
        artifact.torch.backends.cudnn.allow_tf32 = False
        artifact.torch.backends.cuda.matmul.allow_tf32 = False
    print(f"cudnn.allow_tf32 after start: {artifact.torch.backends.cudnn.allow_tf32}")
    if not inference.lstm_model.training:
        print("LSTM model training mode: False")
    else:
        raise RuntimeError("LSTM model is in training mode")
    results = []
    for _, row in sequence_df.iterrows():
        key = int(composite_keys(np.array([int(row["stay_id"])]), np.array([int(row["terminal_window_id"])]))[0])
        if key not in feature_lookup:
            raise RuntimeError(f"Missing features for key {key}")
        if key not in reference_lookup.index:
            raise RuntimeError(f"Missing reference for key {key}")
        if key not in final_lookup.index:
            raise RuntimeError(f"Missing final reference for key {key}")
        try:
            temporal, missing_mask, padding_mask, static = convert_sequence(row)
            if int(row["stay_id"]) == 37921301 and int(row["terminal_window_id"]) == 305:
                path = os.path.join(REAL_TIME_DIR, "debug_lstm_37921301_305.npz")
                np.savez(
                    path,
                    temporal=temporal,
                    missing_mask=missing_mask,
                    padding_mask=padding_mask,
                    static=static,
                    offline_lstm_probability=float(reference_lookup.loc[key, "lstm_probability"]),
                    sequence_length=int(row["sequence_length"])
                )
                print("DEBUG TARGET SAVED: debug_lstm_37921301_305.npz")
            features = feature_lookup[key]
            output = inference.predict(features=features, temporal=temporal, missing_mask=missing_mask, padding_mask=padding_mask, static=static)
            expected_sequence_length = int(row["sequence_length"])
            if int(output["sequence_length"]) != expected_sequence_length:
                raise RuntimeError(f"Sequence length mismatch for key {key}: source={expected_sequence_length} realtime={output['sequence_length']}")
            if not np.isfinite(float(output["ensemble_probability"])) or not 0.0 <= float(output["ensemble_probability"]) <= 1.0:
                raise RuntimeError(f"Invalid ensemble probability for key {key}: {output['ensemble_probability']}")
            expected_prediction = int(float(output["ensemble_probability"]) >= C.ENSEMBLE_THRESHOLD)
            if int(output["prediction"]) != expected_prediction:
                raise RuntimeError(f"Prediction mismatch for key {key}: probability={output['ensemble_probability']} prediction={output['prediction']} expected={expected_prediction}")
            if abs(float(output["threshold"]) - C.ENSEMBLE_THRESHOLD) >= 1e-12:
                raise RuntimeError(f"Threshold mismatch for key {key}: output={output['threshold']} config={C.ENSEMBLE_THRESHOLD}")
            manager = PatientStateManager()
            detector = ChangeDetector(threshold=C.ENSEMBLE_THRESHOLD)
            alert_logger = AlertLogger()
            orchestrator = RealtimeOrchestrator(manager, detector, inference, alert_logger)
            sequence_length = expected_sequence_length
            terminal_window_id = int(row["terminal_window_id"])
            first_window_id = terminal_window_id - sequence_length + 1
            if first_window_id < 0:
                raise RuntimeError(f"Invalid terminal window id for sequence length: key={key}")
            for position in range(sequence_length):
                window_id = first_window_id + position
                timestamp = float(position + 1)
                window_temporal = temporal[position]
                window_missing_mask = missing_mask[position]
                window_xgb_features = np.concatenate([np.where(window_missing_mask == 1, np.nan, window_temporal), static]).astype(np.float32)
                orchestrator_result = orchestrator.process_window(
                    int(row["stay_id"]),
                    window_id,
                    timestamp,
                    window_temporal,
                    window_missing_mask,
                    static,
                    window_xgb_features
                )
            if orchestrator_result["reason"] != "processed":
                raise RuntimeError(f"Orchestrator did not process key {key}: {orchestrator_result['reason']}")
            orchestrator_output = orchestrator_result["inference"]
            for field in ["xgb_raw_probability", "xgb_probability", "lstm_logit", "lstm_raw_probability", "lstm_probability", "ensemble_probability", "threshold", "prediction", "sequence_length", "model_version"]:
                if isinstance(output[field], (int, str)) and output[field] != orchestrator_output[field]:
                    raise RuntimeError(f"Orchestrator mismatch for key {key}: field={field}")
                if isinstance(output[field], (float, np.floating)) and abs(float(output[field]) - float(orchestrator_output[field])) >= 1e-12:
                    raise RuntimeError(f"Orchestrator mismatch for key {key}: field={field}")
        except Exception as error:
            results.append({
                "key": key,
                "stay_id": int(row["stay_id"]),
                "window_id": int(row["terminal_window_id"]),
                "error": f"{type(error).__name__}: {error}"
            })
            continue
        reference_row = reference_lookup.loc[key]
        final_row = final_lookup.loc[key]
        if int(row["label"]) != int(reference_row["label"]):
            raise RuntimeError(f"Label mismatch for key {key}: sequence={int(row['label'])} reference={int(reference_row['label'])}")
        xgb_diff, xgb_pass = compare_value(reference_row["xgb_probability"], output["xgb_raw_probability"], XGB_TOLERANCE)
        xgb_calibrated_diff, xgb_calibrated_pass = compare_value(reference_row["xgb_calibrated_probability"], output["xgb_probability"], XGB_CALIBRATED_TOLERANCE)
        lstm_diff, lstm_pass = compare_value(reference_row["lstm_probability"], output["lstm_raw_probability"], LSTM_RAW_TOLERANCE)
        lstm_calibrated_diff, lstm_calibrated_pass = compare_value(reference_row["lstm_calibrated_probability"], output["lstm_probability"], LSTM_CALIBRATED_TOLERANCE)
        ensemble_diff, ensemble_pass = compare_value(reference_row["ensemble_probability"], output["ensemble_probability"], ENSEMBLE_TOLERANCE)
        reference_lstm_logit = artifact.logit(float(reference_row["lstm_probability"]))
        realtime_lstm_logit = float(output["lstm_logit"])
        lstm_logit_diff = abs(reference_lstm_logit - realtime_lstm_logit)
        lstm_signed_logit_diff = realtime_lstm_logit - reference_lstm_logit
        lstm_logit_pass = lstm_logit_diff <= LSTM_LOGIT_TOLERANCE
        reference_ensemble_expected = C.XGB_WEIGHT * float(reference_row["xgb_calibrated_probability"]) + C.LSTM_WEIGHT * float(reference_row["lstm_calibrated_probability"])
        ensemble_weight_diff = abs(float(reference_row["ensemble_probability"]) - reference_ensemble_expected)
        ensemble_weight_pass = ensemble_weight_diff <= ENSEMBLE_WEIGHT_TOLERANCE
        threshold_straddle = (
            (float(reference_row["ensemble_probability"]) < ENSEMBLE_THRESHOLD <= float(output["ensemble_probability"]))
            or
            (float(output["ensemble_probability"]) < ENSEMBLE_THRESHOLD <= float(reference_row["ensemble_probability"]))
        )
        prediction_match = int(output["prediction"]) == int(final_row["ensemble_prediction"])
        prediction_match_or_straddle = prediction_match or (threshold_straddle and ensemble_pass)
        source = sample_sources.get(key, "layout_sample")
        results.append({
            "key": key,
            "stay_id": int(row["stay_id"]),
            "window_id": int(row["terminal_window_id"]),
            "sequence_length": int(row["sequence_length"]),
            "sample_source": source,
            "label": int(row["label"]),
            "missing_rate": calculate_missingness(missing_mask, int(row["sequence_length"])),
            "xgb_diff": xgb_diff,
            "xgb_calibrated_diff": xgb_calibrated_diff,
            "lstm_logit_diff": lstm_logit_diff,
            "lstm_signed_logit_diff": lstm_signed_logit_diff,
            "lstm_diff": lstm_diff,
            "lstm_signed_diff": float(output["lstm_raw_probability"]) - float(reference_row["lstm_probability"]),
            "lstm_calibrated_diff": lstm_calibrated_diff,
            "ensemble_weight_diff": ensemble_weight_diff,
            "ensemble_diff": ensemble_diff,
            "threshold_distance": abs(float(reference_row["ensemble_probability"]) - ENSEMBLE_THRESHOLD),
            "threshold_straddle": threshold_straddle,
            "offline_prediction": int(final_row["ensemble_prediction"]),
            "realtime_prediction": int(output["prediction"]),
            "prediction_match": prediction_match,
            "prediction_match_or_straddle": prediction_match_or_straddle,
            "xgb_pass": xgb_pass,
            "xgb_calibrated_pass": xgb_calibrated_pass,
            "lstm_logit_pass": lstm_logit_pass,
            "lstm_pass": lstm_pass,
            "lstm_calibrated_pass": lstm_calibrated_pass,
            "ensemble_weight_pass": ensemble_weight_pass,
            "ensemble_pass": ensemble_pass,
            "probability_checks_pass": xgb_pass and xgb_calibrated_pass and lstm_logit_pass and lstm_pass and lstm_calibrated_pass and ensemble_pass and ensemble_weight_pass,
            "error": ""
        })
    results_df = pd.DataFrame(results)
    errors = results_df[results_df["error"] != ""]
    valid = results_df[results_df["error"] == ""]
    print(f"Rows processed: {len(results_df)}")
    print(f"Rows with errors: {len(errors)}")
    if not errors.empty:
        print(errors[["stay_id", "window_id", "error"]].head(10).to_string(index=False))
    if valid.empty:
        print("STAGE A PARITY SAMPLE FAILED")
        if not REPORT_ONLY:
            sys.exit(1)
        return
    print("PARITY STATISTICS - ALL VALID ROWS")
    for name in ["xgb_diff", "xgb_calibrated_diff", "lstm_logit_diff", "lstm_diff", "lstm_calibrated_diff", "ensemble_weight_diff", "ensemble_diff"]:
        print_percentile_line(name, valid[name])
    print("PARITY STATISTICS - LAYOUT SAMPLE ONLY")
    print_group_statistics(valid, "Layout sample", valid["sample_source"] == "layout_sample")
    print("PARITY STATISTICS - EXACT NEAR-THRESHOLD ROWS")
    print_group_statistics(valid, "Near-threshold sample", valid["sample_source"] == "near_threshold")
    print("PARITY STATISTICS - BY SEQUENCE LENGTH")
    by_length = valid.groupby("sequence_length").agg(
        rows=("key", "size"),
        xgb_p99=("xgb_diff", lambda x: float(np.percentile(x, 99))),
        xgb_calibrated_p99=("xgb_calibrated_diff", lambda x: float(np.percentile(x, 99))),
        lstm_logit_p99=("lstm_logit_diff", lambda x: float(np.percentile(x, 99))),
        lstm_calibrated_p99=("lstm_calibrated_diff", lambda x: float(np.percentile(x, 99))),
        ensemble_p99=("ensemble_diff", lambda x: float(np.percentile(x, 99)))
    ).reset_index()
    print(by_length.to_string(index=False))
    print(f"LSTM signed logit mean: {valid['lstm_signed_logit_diff'].mean():.10f}")
    print(f"LSTM signed logit median: {valid['lstm_signed_logit_diff'].median():.10f}")
    print(f"LSTM positive signed logit differences: {(valid['lstm_signed_logit_diff'] > 0).sum()}")
    print(f"LSTM negative signed logit differences: {(valid['lstm_signed_logit_diff'] < 0).sum()}")
    print(f"Threshold straddles: {valid['threshold_straddle'].sum()}")
    print(f"Exact prediction matches: {valid['prediction_match'].sum()}/{len(valid)}")
    print(f"Prediction matches or threshold straddles: {valid['prediction_match_or_straddle'].sum()}/{len(valid)}")
    print(f"XGB raw rows passing: {valid['xgb_pass'].sum()}/{len(valid)}")
    print(f"XGB calibrated rows passing: {valid['xgb_calibrated_pass'].sum()}/{len(valid)}")
    print(f"LSTM logit rows passing: {valid['lstm_logit_pass'].sum()}/{len(valid)}")
    print(f"LSTM raw rows passing: {valid['lstm_pass'].sum()}/{len(valid)}")
    print(f"LSTM calibrated rows passing: {valid['lstm_calibrated_pass'].sum()}/{len(valid)}")
    print(f"Ensemble rows passing: {valid['ensemble_pass'].sum()}/{len(valid)}")
    print(f"Probability rows passing: {valid['probability_checks_pass'].sum()}/{len(valid)}")
    print(f"Reference ensemble-weight rows passing: {valid['ensemble_weight_pass'].sum()}/{len(valid)}")
    output_path = os.path.join(REAL_TIME_DIR, "stage_a_streaming_sample_results.parquet")
    results_df.to_parquet(output_path, index=False)
    print(f"Results saved: {output_path}")
    probability_pass = bool(valid["probability_checks_pass"].all())
    prediction_pass = bool(valid["prediction_match_or_straddle"].all())
    error_pass = errors.empty
    if not probability_pass or not prediction_pass or not error_pass:
        print("PARITY SAMPLE FAILED")
        if REPORT_ONLY:
            print("PARITY_REPORT_ONLY=1: report generated without failing the process")
            return
        sys.exit(1)
    print("STREAMING PARITY SAMPLE PASSED")
if __name__ == "__main__":
    main()
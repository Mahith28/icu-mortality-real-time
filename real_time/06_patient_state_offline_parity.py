import os
import sys
import json
import importlib.util
import numpy as np
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
SEQUENCE_PATH = "hdfs://localhost:9000/user/mahith/icu/lstm_adaptive/test_sequences"
FEATURE_PATH = "hdfs://localhost:9000/user/mahith/icu/clean_features/test_features"
IMPUTATION_PATH = "/home/mahith/BDA_PROJECT/lstm_clean/results/lstm_imputation_values.json"
SHORT_STAYS = 3
LONG_STAYS = 5
VERY_LONG_STAYS = 1
GAPPED_STAYS = 5
VERY_LONG_MIN = 240
INTERLEAVE_SEED = 42
def load_patient_state():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_patient_state.py")
    spec = importlib.util.spec_from_file_location("patient_state_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load PatientState module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PatientStateManager
def get_hdfs_filesystem():
    classpath = os.popen("hadoop classpath --glob").read().strip()
    if not classpath:
        raise RuntimeError("Hadoop classpath is empty")
    os.environ["CLASSPATH"] = classpath
    return pafs.HadoopFileSystem(host="localhost", port=9000, user="mahith")
def list_parquet_files(filesystem, path):
    selector = pafs.FileSelector(path, recursive=False, allow_not_found=False)
    infos = filesystem.get_file_info(selector)
    files = []
    for info in infos:
        if info.type == pafs.FileType.File and info.path.endswith(".parquet"):
            files.append(info.path)
    files.sort()
    if not files:
        raise RuntimeError(f"No parquet files found in HDFS path {path}")
    return files
def build_dataset(filesystem, path):
    parquet_files = list_parquet_files(filesystem, path)
    return ds.dataset(parquet_files, format="parquet", filesystem=filesystem)
def load_imputation_values():
    if not os.path.exists(IMPUTATION_PATH):
        raise RuntimeError(f"Offline imputation file not found: {IMPUTATION_PATH}")
    with open(IMPUTATION_PATH) as f:
        values = json.load(f)
    if isinstance(values, list):
        values = np.asarray(values, dtype=np.float32)
    elif isinstance(values, dict):
        if "values" in values:
            values = np.asarray(values["values"], dtype=np.float32)
        elif "temporal_imputation" in values:
            values = np.asarray(values["temporal_imputation"], dtype=np.float32)
        elif all(str(key).isdigit() for key in values):
            values = np.asarray([values[str(i)] for i in range(C.TEMPORAL_FEATURES)], dtype=np.float32)
        else:
            raise RuntimeError("Unsupported imputation dictionary format")
    else:
        raise RuntimeError("Unsupported imputation file format")
    if values.shape != (C.TEMPORAL_FEATURES,):
        raise RuntimeError(f"Invalid imputation vector shape: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("Offline imputation values contain non-finite values")
    return values
def sparse_to_dense(features):
    if features is None:
        raise RuntimeError("Feature vector is missing")
    values = np.asarray(features["values"], dtype=np.float32)
    indices = features.get("indices")
    if indices is None:
        if values.shape != (C.FEATURE_COUNT,):
            raise RuntimeError(f"Invalid dense feature shape: {values.shape}")
        return values.copy()
    size = features.get("size")
    if size != C.FEATURE_COUNT:
        raise RuntimeError(f"Expected {C.FEATURE_COUNT} features, got {size}")
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) != len(values):
        raise RuntimeError("Sparse feature index/value mismatch")
    if len(indices) and (indices.min() < 0 or indices.max() >= C.FEATURE_COUNT):
        raise RuntimeError("Sparse feature index out of range")
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("Duplicate sparse indices")
    dense = np.zeros(C.FEATURE_COUNT, dtype=np.float32)
    dense[indices] = values
    return dense
def prepare_window(features, imputation):
    raw_features = sparse_to_dense(features)
    if np.isinf(raw_features).any():
        raise RuntimeError("Raw feature vector contains infinite values")
    raw_temporal = raw_features[:C.TEMPORAL_FEATURES].copy()
    static = raw_features[C.TEMPORAL_FEATURES:].copy()
    if np.isinf(static).any():
        raise RuntimeError("Static features contain infinite values")
    missing_mask = np.isnan(raw_temporal).astype(np.float32)
    temporal = raw_temporal.copy()
    missing_positions = np.isnan(temporal)
    temporal[missing_positions] = imputation[missing_positions]
    if not np.isfinite(temporal).all():
        raise RuntimeError("Temporal features contain non-finite values after imputation")
    if not np.isfinite(static).all():
        raise RuntimeError("Static features contain non-finite values")
    return temporal.astype(np.float32), missing_mask.astype(np.float32), static.astype(np.float32), raw_features.astype(np.float32)
def compare_array(name, actual, expected, stay_id, window_id, equal_nan=False):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise RuntimeError(f"{name} shape mismatch at stay {stay_id}, window {window_id}: actual={actual.shape}, expected={expected.shape}")
    if np.array_equal(actual, expected, equal_nan=equal_nan):
        return
    if equal_nan:
        mismatch = ~np.isclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    else:
        mismatch = actual != expected
    indices = np.flatnonzero(mismatch)
    if len(indices) == 0:
        raise RuntimeError(f"{name} mismatch at stay {stay_id}, window {window_id}")
    first = int(indices[0])
    actual_value = actual.reshape(-1)[first]
    expected_value = expected.reshape(-1)[first]
    finite_actual = np.isfinite(actual)
    finite_expected = np.isfinite(expected)
    if finite_actual.any() and finite_expected.any():
        common = finite_actual & finite_expected
        if common.any():
            max_difference = float(np.max(np.abs(actual[common] - expected[common])))
        else:
            max_difference = float("nan")
    else:
        max_difference = float("nan")
    raise RuntimeError(f"{name} mismatch at stay {stay_id}, window {window_id}, index={first}, actual={actual_value}, expected={expected_value}, max_abs_difference={max_difference}")
def validate_reference_row(row):
    sequence_length = int(row["sequence_length"])
    if sequence_length < 1 or sequence_length > C.MAX_SEQ_LEN:
        raise RuntimeError(f"Invalid sequence length {sequence_length}")
    temporal = np.asarray(row["temporal"], dtype=np.float32)
    missing_mask = np.asarray(row["missing_mask"], dtype=np.float32)
    padding_mask = np.asarray(row["padding_mask"], dtype=np.float32)
    static = np.asarray(row["static"], dtype=np.float32)
    if temporal.shape != (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES):
        raise RuntimeError(f"Invalid temporal shape: {temporal.shape}")
    if missing_mask.shape != (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES):
        raise RuntimeError(f"Invalid missing mask shape: {missing_mask.shape}")
    if padding_mask.shape != (C.MAX_SEQ_LEN,):
        raise RuntimeError(f"Invalid padding mask shape: {padding_mask.shape}")
    if static.shape != (C.STATIC_FEATURES,):
        raise RuntimeError(f"Invalid static shape: {static.shape}")
    if not np.isin(missing_mask, (0.0, 1.0)).all():
        raise RuntimeError("Reference missing mask contains values other than 0 or 1")
    expected_padding = np.ones(C.MAX_SEQ_LEN, dtype=np.float32)
    expected_padding[:sequence_length] = 0.0
    compare_array("Reference padding mask", padding_mask, expected_padding, int(row["stay_id"]), int(row["terminal_window_id"]))
    if not np.isfinite(temporal[:sequence_length]).all():
        raise RuntimeError(f"Reference temporal values contain non-finite values at stay {row['stay_id']}, window {row['terminal_window_id']}")
    if not np.isfinite(static).all():
        raise RuntimeError(f"Reference static values contain non-finite values at stay {row['stay_id']}, window {row['terminal_window_id']}")
    return temporal, missing_mask, padding_mask, static, sequence_length
def get_stay_statistics(dataset):
    table = dataset.to_table(columns=["stay_id", "window_id"])
    stay_ids = np.asarray(table["stay_id"])
    window_ids = np.asarray(table["window_id"])
    if len(stay_ids) == 0:
        raise RuntimeError("Feature dataset contains no rows")
    order = np.lexsort((window_ids, stay_ids))
    sorted_stays = stay_ids[order]
    sorted_windows = window_ids[order]
    unique_stays, counts = np.unique(sorted_stays, return_counts=True)
    counts = {int(stay_id): int(count) for stay_id, count in zip(unique_stays, counts)}
    if len(sorted_stays) > 1:
        same_stay = sorted_stays[1:] == sorted_stays[:-1]
        gaps = np.diff(sorted_windows)
        gap_stays = np.unique(sorted_stays[1:][same_stay & (gaps != 1)])
        gapped_candidates = [(counts[int(stay_id)], int(stay_id)) for stay_id in gap_stays]
    else:
        gapped_candidates = []
    gapped_candidates.sort()
    return counts, gapped_candidates
def get_reference_stays(dataset):
    table = dataset.to_table(columns=["stay_id"])
    stay_ids = np.asarray(table["stay_id"])
    return set(int(stay_id) for stay_id in np.unique(stay_ids))
def select_stays(feature_dataset, sequence_dataset):
    counts, gapped_candidates = get_stay_statistics(feature_dataset)
    reference_stays = get_reference_stays(sequence_dataset)
    common_stays = set(counts).intersection(reference_stays)
    if not common_stays:
        raise RuntimeError("No stays are shared between feature and reference datasets")
    short_candidates = sorted((counts[stay_id], stay_id) for stay_id in common_stays if counts[stay_id] < C.MAX_SEQ_LEN)
    long_candidates = sorted((counts[stay_id], stay_id) for stay_id in common_stays if counts[stay_id] > C.MAX_SEQ_LEN)
    very_long_candidates = sorted((counts[stay_id], stay_id) for stay_id in common_stays if counts[stay_id] >= VERY_LONG_MIN)
    gapped_candidates = [(count, stay_id) for count, stay_id in gapped_candidates if stay_id in common_stays]
    gapped_candidates.sort()
    short_stays = [stay_id for _, stay_id in short_candidates[:SHORT_STAYS]]
    long_stays = [stay_id for _, stay_id in long_candidates[:LONG_STAYS]]
    very_long_stays = [stay_id for _, stay_id in very_long_candidates[:VERY_LONG_STAYS]]
    gapped_stays = [stay_id for _, stay_id in gapped_candidates[:GAPPED_STAYS]]
    gapped_long = [stay_id for count, stay_id in gapped_candidates if count > C.MAX_SEQ_LEN][:1]
    gapped_stays = list(dict.fromkeys(gapped_stays + gapped_long))
    selected = list(dict.fromkeys(short_stays + long_stays + very_long_stays + gapped_stays))
    if len(short_stays) < SHORT_STAYS:
        raise RuntimeError(f"Could not select {SHORT_STAYS} short stays")
    if len(long_stays) < LONG_STAYS:
        raise RuntimeError(f"Could not select {LONG_STAYS} long stays")
    if len(very_long_stays) < VERY_LONG_STAYS:
        raise RuntimeError(f"Could not select {VERY_LONG_STAYS} very long stays")
    if len(gapped_stays) < GAPPED_STAYS:
        raise RuntimeError(f"Could not select {GAPPED_STAYS} gapped stays")
    if not any(count > C.MAX_SEQ_LEN for count, stay_id in gapped_candidates if stay_id in gapped_stays):
        raise RuntimeError("No gapped stay with more than 120 windows was selected")
    return selected, counts
def scan_rows(dataset, columns, selected_stays, batch_size):
    selected_stays = set(selected_stays)
    for fragment in dataset.get_fragments():
        parquet_file = pq.ParquetFile(fragment.path, filesystem=dataset.filesystem)
        for row_group in range(parquet_file.num_row_groups):
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns, row_groups=[row_group], use_threads=False):
                table = pa.Table.from_batches([batch])
                stay_array = table.column("stay_id").to_pylist()
                selected_indices = [i for i, stay_id in enumerate(stay_array) if stay_id in selected_stays]
                if not selected_indices:
                    continue
                table = table.take(pa.array(selected_indices))
                yield from table.to_pylist()
def load_reference_rows(dataset, selected_stays):
    columns = ["stay_id", "subject_id", "terminal_window_id", "sequence_length", "label", "temporal", "missing_mask", "padding_mask", "static"]
    rows = {}
    for row in scan_rows(dataset, columns, selected_stays, batch_size=8):
        for key in ("temporal", "missing_mask", "padding_mask", "static"):
            row[key] = np.asarray(row[key], dtype=np.float32)
        key = (int(row["stay_id"]), int(row["terminal_window_id"]))
        if key in rows:
            raise RuntimeError(f"Duplicate reference row {key}")
        rows[key] = row
    return rows
def load_feature_rows(dataset, selected_stays):
    columns = ["stay_id", "subject_id", "window_id", "features", "label"]
    rows = {}
    for row in scan_rows(dataset, columns, selected_stays, batch_size=128):
        key = (int(row["stay_id"]), int(row["window_id"]))
        if key in rows:
            raise RuntimeError(f"Duplicate feature row {key}")
        rows[key] = row
    return rows
def build_interleaved_order(feature_rows, selected_stays):
    by_stay = {stay_id: [] for stay_id in selected_stays}
    for stay_id, window_id in feature_rows:
        if stay_id in by_stay:
            by_stay[stay_id].append(window_id)
    for stay_id in by_stay:
        by_stay[stay_id].sort()
    rng = np.random.default_rng(INTERLEAVE_SEED)
    active = [stay_id for stay_id in selected_stays if by_stay[stay_id]]
    pairs = []
    while active:
        rng.shuffle(active)
        next_active = []
        for stay_id in active:
            window_ids = by_stay[stay_id]
            if window_ids:
                window_id = window_ids.pop(0)
                pairs.append((stay_id, window_id))
            if window_ids:
                next_active.append(stay_id)
        active = next_active
    return pairs
def main():
    PatientStateManager = load_patient_state()
    print("STAGE B OFFLINE PARITY")
    print(f"Sequence path: {SEQUENCE_PATH}")
    print(f"Feature path: {FEATURE_PATH}")
    print(f"Expected temporal features: {C.TEMPORAL_FEATURES}")
    print(f"Expected static features: {C.STATIC_FEATURES}")
    print(f"Expected total features: {C.FEATURE_COUNT}")
    print(f"Maximum sequence length: {C.MAX_SEQ_LEN}")
    filesystem = get_hdfs_filesystem()
    sequence_dataset = build_dataset(filesystem, "/user/mahith/icu/lstm_adaptive/test_sequences")
    feature_dataset = build_dataset(filesystem, "/user/mahith/icu/clean_features/test_features")
    imputation = load_imputation_values()
    selected_stays, stay_counts = select_stays(feature_dataset, sequence_dataset)
    print(f"Selected stays: {len(selected_stays)}")
    for stay_id in selected_stays:
        print(f"Stay {stay_id}: {stay_counts[stay_id]} feature windows")
    print("Loading reference rows")
    reference_rows = load_reference_rows(sequence_dataset, selected_stays)
    print(f"Loaded reference rows: {len(reference_rows)}")
    print("Loading feature rows")
    feature_rows = load_feature_rows(feature_dataset, selected_stays)
    print(f"Loaded feature rows: {len(feature_rows)}")
    expected_pairs = set(reference_rows)
    actual_pairs = set(feature_rows)
    reference_only = expected_pairs - actual_pairs
    feature_only = actual_pairs - expected_pairs
    if reference_only:
        first = sorted(reference_only)[0]
        raise RuntimeError(f"Reference-only row found: stay {first[0]}, window {first[1]}")
    if feature_only:
        first = sorted(feature_only)[0]
        raise RuntimeError(f"Feature-only row found: stay {first[0]}, window {first[1]}")
    pairs = build_interleaved_order(feature_rows, selected_stays)
    if len(pairs) != len(expected_pairs):
        raise RuntimeError(f"Interleaved feature count mismatch: {len(pairs)} vs {len(expected_pairs)}")
    manager = PatientStateManager()
    previous_stay = None
    distinct_stays = set()
    interleaved_transitions = 0
    checked = 0
    padded_checks = 0
    full_120_checks = 0
    boundary_121_checks = 0
    truncation_checks = 0
    gap_in_history_checks = 0
    gap_and_truncation_checks = 0
    imputed_cells_checked = 0
    label_checks = 0
    xgb_checks = 0
    lstm_checks = 0
    stay_accept_counts = {}
    for stay_id, window_id in pairs:
        if previous_stay is not None and previous_stay != stay_id:
            interleaved_transitions += 1
        previous_stay = stay_id
        distinct_stays.add(stay_id)
        feature_row = feature_rows[(stay_id, window_id)]
        reference_row = reference_rows[(stay_id, window_id)]
        try:
            temporal, missing_mask, static, xgb_features = prepare_window(feature_row["features"], imputation)
            reference_temporal, reference_missing_mask, reference_padding_mask, reference_static, reference_length = validate_reference_row(reference_row)
            timestamp = window_id * C.WINDOW_MINUTES * 60
            state, accepted = manager.add_window(stay_id, window_id, timestamp, temporal, missing_mask, static, xgb_features)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed at stay {stay_id}, window {window_id}: {exc}") from exc
        if not accepted:
            raise RuntimeError(f"Window {window_id} was unexpectedly rejected for stay {stay_id}")
        n = stay_accept_counts.get(stay_id, 0) + 1
        stay_accept_counts[stay_id] = n
        if n == C.MAX_SEQ_LEN:
            full_120_checks += 1
        if n == C.MAX_SEQ_LEN + 1:
            boundary_121_checks += 1
        if n > C.MAX_SEQ_LEN:
            truncation_checks += 1
        sequence = state.get_sequence()
        if sequence is None:
            raise RuntimeError(f"State sequence is missing for stay {stay_id}, window {window_id}")
        actual_temporal, actual_missing_mask, actual_padding_mask, actual_static, actual_length, actual_xgb = sequence
        compare_array("Temporal sequence", actual_temporal, reference_temporal, stay_id, window_id)
        compare_array("Missing mask", actual_missing_mask, reference_missing_mask, stay_id, window_id)
        compare_array("Padding mask", actual_padding_mask, reference_padding_mask, stay_id, window_id)
        compare_array("Static features", actual_static, reference_static, stay_id, window_id)
        if actual_length != reference_length:
            raise RuntimeError(f"Sequence length mismatch at stay {stay_id}, window {window_id}: actual={actual_length}, expected={reference_length}")
        reference_label = int(reference_row["label"])
        feature_label = int(feature_row["label"])
        if reference_label != feature_label:
            raise RuntimeError(f"Label mismatch at stay {stay_id}, window {window_id}: reference={reference_label}, feature={feature_label}")
        label_checks += 1
        if actual_xgb is None:
            raise RuntimeError(f"XGBoost state features missing at stay {stay_id}, window {window_id}")
        compare_array("XGBoost state vector", actual_xgb, xgb_features, stay_id, window_id, equal_nan=True)
        last = actual_length - 1
        expected_xgb_temporal = np.where(actual_missing_mask[last] == 1.0, np.nan, actual_temporal[last])
        compare_array("XGBoost temporal alignment", actual_xgb[:C.TEMPORAL_FEATURES], expected_xgb_temporal, stay_id, window_id, equal_nan=True)
        compare_array("XGBoost static alignment", actual_xgb[C.TEMPORAL_FEATURES:], actual_static, stay_id, window_id)
        xgb_checks += 1
        lstm_checks += 1
        imputed_cells_checked += int(actual_missing_mask[:actual_length].sum())
        if reference_length < C.MAX_SEQ_LEN:
            padded_checks += 1
        ids = [item[0] for item in state.window_records]
        has_gap = any(b != a + 1 for a, b in zip(ids, ids[1:]))
        if has_gap:
            gap_in_history_checks += 1
            if n > C.MAX_SEQ_LEN:
                gap_and_truncation_checks += 1
        checked += 1
    if padded_checks == 0:
        raise RuntimeError("No padded sequences were compared")
    if full_120_checks == 0:
        raise RuntimeError("No sequence was compared at the first 120-window boundary")
    if boundary_121_checks == 0:
        raise RuntimeError("No 121st-window boundary was compared")
    if truncation_checks == 0:
        raise RuntimeError("No rolling-window truncation was compared")
    if gap_in_history_checks == 0:
        raise RuntimeError("No gapped history was compared")
    if gap_and_truncation_checks == 0:
        raise RuntimeError("No gapped history was compared after the 120-window truncation boundary")
    if imputed_cells_checked == 0:
        raise RuntimeError("No imputed cells were compared; imputation path is unproven")
    if label_checks != checked:
        raise RuntimeError(f"Label checks incomplete: {label_checks}/{checked}")
    if xgb_checks != checked:
        raise RuntimeError(f"XGBoost checks incomplete: {xgb_checks}/{checked}")
    if lstm_checks != checked:
        raise RuntimeError(f"LSTM checks incomplete: {lstm_checks}/{checked}")
    if len(distinct_stays) < 2:
        raise RuntimeError("Interleaving did not include at least two distinct stays")
    if interleaved_transitions < len(distinct_stays):
        raise RuntimeError(f"Insufficient interleaved transitions: {interleaved_transitions} for {len(distinct_stays)} stays")
    print(f"Reference rows compared: {checked}")
    print(f"Padded sequence checks: {padded_checks}")
    print(f"Exactly-120 boundary checks: {full_120_checks}")
    print(f"121st-window boundary checks: {boundary_121_checks}")
    print(f"Rolling truncation checks: {truncation_checks}")
    print(f"Gapped-history checks: {gap_in_history_checks}")
    print(f"Gap-plus-truncation checks: {gap_and_truncation_checks}")
    print(f"Imputed cells checked: {imputed_cells_checked}")
    print(f"Label checks: {label_checks}")
    print(f"XGBoost alignment checks: {xgb_checks}")
    print(f"LSTM sequence checks: {lstm_checks}")
    print(f"Distinct stays: {len(distinct_stays)}")
    print(f"Interleaved transitions: {interleaved_transitions}")
    print("OFFLINE LSTM SEQUENCE PARITY: CHECKED")
    print("MISSING-MASK RECONSTRUCTION: CHECKED")
    print("STATIC FEATURE PARITY: CHECKED")
    print("LABEL PARITY: CHECKED")
    print("XGBOOST STATE STORAGE: CHECKED")
    print("INTERLEAVED MULTI-STAY REPLAY: CHECKED")
    print("120-WINDOW ROLLING BOUNDARY: CHECKED")
    print("GAPPED HISTORY: CHECKED")
    print("GAP PLUS TRUNCATION: CHECKED")
    print("REAL-TIME PREPROCESSING PARITY: NOT CLAIMED")
    print("OFFLINE XGBOOST INPUT PARITY: NOT CLAIMED")
    print("STAGE B OFFLINE PARITY PASSED")
if __name__ == "__main__":
    main()
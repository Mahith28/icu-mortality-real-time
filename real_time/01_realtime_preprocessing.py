import os
import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
import config as C
VITAL_FEATURES = {
    "Heart Rate": "Heart_Rate",
    "SBP": "SBP",
    "DBP": "DBP",
    "MAP": "MAP",
    "Respiratory Rate": "Respiratory_Rate",
    "Temperature": "Temperature",
    "SpO2": "SpO2",
    "GCS Eye": "GCS_Eye",
    "GCS Verbal": "GCS_Verbal",
    "GCS Motor": "GCS_Motor"
}
CATEGORICAL_FEATURES = ["gender", "race", "insurance", "admission_type", "admission_location", "first_careunit"]
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
class RealtimePreprocessor:
    def __init__(self):
        self.spark = None
        self.pipeline = None
        self.feature_names = None
        self.assembler_inputs = None
        self.imputation = None
        self.active_windows = {}
        self.locf_state = {}
    def start(self):
        if self.spark is not None:
            return
        self.spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("RealtimePreprocessing")
            .config("spark.sql.shuffle.partitions", "4")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("ERROR")
        if not os.path.exists(C.CLEAN_PIPELINE):
            raise RuntimeError(f"Preprocessing model not found: {C.CLEAN_PIPELINE}")
        self.pipeline = PipelineModel.load(C.CLEAN_PIPELINE)
        self._validate_pipeline()
        self._load_feature_names()
        self._load_imputation()
    def stop(self):
        if self.spark is not None:
            self.spark.stop()
            self.spark = None
    def _validate_pipeline(self):
        stages = [stage.__class__.__name__ for stage in self.pipeline.stages]
        print(f"Pipeline stages: {stages}")
        for stage in self.pipeline.stages:
            name = stage.__class__.__name__
            if name.startswith("StringIndexer") or name.startswith("OneHotEncoder") or name == "VectorAssembler":
                if stage.getHandleInvalid() != "keep":
                    raise RuntimeError(f"{name} handleInvalid is not keep")
    def _load_feature_names(self):
        assembler = None
        for stage in self.pipeline.stages:
            if stage.__class__.__name__ == "VectorAssembler":
                assembler = stage
        if assembler is None:
            raise RuntimeError("VectorAssembler not found")
        assembler_inputs = list(assembler.getInputCols())
        if len(assembler_inputs) != C.ASSEMBLER_INPUT_COUNT:
            raise RuntimeError(f"Expected {C.ASSEMBLER_INPUT_COUNT} assembler inputs but found {len(assembler_inputs)}")
        if assembler_inputs[:C.TEMPORAL_FEATURES] != TEMPORAL_FEATURE_NAMES:
            raise RuntimeError("Assembler temporal input order does not match TEMPORAL_FEATURE_NAMES")
        self.assembler_inputs = assembler_inputs
        print(f"Assembler inputs: {len(assembler_inputs)}")
        print("Temporal assembler inputs verified")
    def _load_imputation(self):
        path = os.path.join(C.ARTIFACT_DIR, "imputation_values.json")
        if not os.path.exists(path):
            raise RuntimeError(f"Imputation artifact not found: {path}")
        with open(path, "r") as file:
            data = json.load(file)
        if "values" not in data:
            raise RuntimeError("Imputation artifact does not contain 'values'")
        self.imputation = data["values"]
        if len(self.imputation) != C.TEMPORAL_FEATURES:
            raise RuntimeError(f"Expected {C.TEMPORAL_FEATURES} imputation values but found {len(self.imputation)}")
        if set(self.imputation) != set(TEMPORAL_FEATURE_NAMES):
            raise RuntimeError("Imputation feature names do not match TEMPORAL_FEATURE_NAMES")
        for name in TEMPORAL_FEATURE_NAMES:
            if not np.isfinite(float(self.imputation[name])):
                raise RuntimeError(f"Invalid imputation value for {name}")
    @staticmethod
    def _encode_state_value(value):
        """Convert a runtime value into strict-JSON-safe state data."""
        if isinstance(value, pd.Timestamp):
            if pd.isna(value):
                raise ValueError("NaT is not valid in preprocessing runtime state")
            return {"__realtime_state_type__": "timestamp", "value": value.isoformat()}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            if np.isnan(value):
                token = "nan"
            elif value > 0:
                token = "+inf"
            else:
                token = "-inf"
            return {"__realtime_state_type__": "float", "value": token}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise ValueError(f"Unsupported preprocessing state value: {type(value).__name__}")

    @staticmethod
    def _decode_state_value(value):
        """Restore values encoded by _encode_state_value."""
        if isinstance(value, dict) and "__realtime_state_type__" in value:
            kind = value.get("__realtime_state_type__")
            if kind == "timestamp" and isinstance(value.get("value"), str):
                return pd.Timestamp(value["value"])
            if kind == "float" and value.get("value") in ("nan", "+inf", "-inf"):
                return {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}[value["value"]]
            raise ValueError("Invalid encoded preprocessing state value")
        if isinstance(value, (str, bool, int, float)) or value is None:
            return value
        raise ValueError("Invalid preprocessing state value type")

    def to_state_dict(self):
        """Return JSON-safe preprocessor runtime state without mutating it."""
        active_windows = []
        for stay_id, windows in self.active_windows.items():
            if isinstance(stay_id, bool) or not isinstance(stay_id, (int, np.integer)):
                raise ValueError(f"Invalid stay ID in active_windows: {stay_id!r}")
            if not isinstance(windows, dict):
                raise ValueError(f"Invalid active-window map for stay {stay_id}")
            for window_id, observations in windows.items():
                if isinstance(window_id, bool) or not isinstance(window_id, (int, np.integer)):
                    raise ValueError(f"Invalid window ID: {window_id!r}")
                if int(window_id) < 0:
                    raise ValueError("Active-window ID must be non-negative")
                if not isinstance(observations, list) or not observations:
                    raise ValueError("Active-window observations must be non-empty lists")
                encoded_observations = []
                for observation in observations:
                    if not isinstance(observation, dict):
                        raise ValueError("Each active observation must be a dictionary")
                    required = {"stay_id", "charttime", "vital_sign", "value"}
                    if not required.issubset(observation):
                        raise ValueError("Active observation is missing required fields")
                    encoded_observations.append({
                        "stay_id": int(observation["stay_id"]),
                        "charttime": self._encode_state_value(pd.Timestamp(observation["charttime"])),
                        "vital_sign": self._encode_state_value(observation["vital_sign"]),
                        "value": self._encode_state_value(observation["value"]),
                    })
                active_windows.append({
                    "stay_id": int(stay_id),
                    "window_id": int(window_id),
                    "observations": encoded_observations,
                })

        locf_state = []
        for stay_id, values in self.locf_state.items():
            if isinstance(stay_id, bool) or not isinstance(stay_id, (int, np.integer)):
                raise ValueError(f"Invalid stay ID in locf_state: {stay_id!r}")
            if not isinstance(values, dict):
                raise ValueError(f"Invalid LOCF state for stay {stay_id}")
            locf_state.append({
                "stay_id": int(stay_id),
                "values": {str(name): self._encode_state_value(value) for name, value in values.items()},
            })

        payload = {
            "schema_version": 1,
            "active_windows": active_windows,
            "locf_state": locf_state,
        }
        # Enforce the same strict JSON rule used by the SQLite store.
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        return payload

    def restore_state_dict(self, payload):
        """Validate and restore serialized runtime state atomically."""
        if not isinstance(payload, dict):
            raise ValueError("Preprocessor state must be a dictionary")
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported preprocessor state schema version")
        if set(payload) != {"schema_version", "active_windows", "locf_state"}:
            raise ValueError("Preprocessor state has missing or unexpected fields")
        if not isinstance(payload["active_windows"], list) or not isinstance(payload["locf_state"], list):
            raise ValueError("Preprocessor state collections must be lists")

        # Build and validate temporary structures first so failures cannot leave
        # the current live state partially replaced.
        restored_windows = {}
        for entry in payload["active_windows"]:
            if not isinstance(entry, dict) or set(entry) != {"stay_id", "window_id", "observations"}:
                raise ValueError("Invalid active-window state entry")
            stay_id, window_id, observations = entry["stay_id"], entry["window_id"], entry["observations"]
            if (isinstance(stay_id, bool) or not isinstance(stay_id, int)
                    or isinstance(window_id, bool) or not isinstance(window_id, int)
                    or window_id < 0 or not isinstance(observations, list) or not observations):
                raise ValueError("Invalid active-window identifiers or observations")
            if stay_id not in restored_windows:
                restored_windows[stay_id] = {}
            if window_id in restored_windows[stay_id]:
                raise ValueError("Duplicate stay/window entry in preprocessor state")
            restored_observations = []
            for observation in observations:
                if not isinstance(observation, dict) or set(observation) != {"stay_id", "charttime", "vital_sign", "value"}:
                    raise ValueError("Invalid observation entry in preprocessor state")
                decoded_stay = observation["stay_id"]
                if isinstance(decoded_stay, bool) or not isinstance(decoded_stay, int) or decoded_stay != stay_id:
                    raise ValueError("Observation stay ID does not match its window")
                charttime = self._decode_state_value(observation["charttime"])
                vital_sign = self._decode_state_value(observation["vital_sign"])
                value = self._decode_state_value(observation["value"])
                if (not isinstance(charttime, pd.Timestamp) or pd.isna(charttime)
                        or not isinstance(vital_sign, str) or not vital_sign):
                    raise ValueError("Invalid observation timestamp or vital-sign name")
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))):
                    raise ValueError("Observation value must be numeric or None")
                restored_observations.append({
                    "stay_id": decoded_stay,
                    "charttime": charttime,
                    "vital_sign": vital_sign,
                    "value": value,
                })
            restored_windows[stay_id][window_id] = restored_observations

        restored_locf = {}
        for entry in payload["locf_state"]:
            if not isinstance(entry, dict) or set(entry) != {"stay_id", "values"}:
                raise ValueError("Invalid LOCF state entry")
            stay_id, values = entry["stay_id"], entry["values"]
            if isinstance(stay_id, bool) or not isinstance(stay_id, int) or not isinstance(values, dict):
                raise ValueError("Invalid LOCF stay ID or values")
            if stay_id in restored_locf:
                raise ValueError("Duplicate stay entry in LOCF state")
            decoded_values = {}
            for name, encoded_value in values.items():
                if not isinstance(name, str) or name not in TEMPORAL_FEATURE_NAMES:
                    raise ValueError(f"Unknown LOCF feature name: {name!r}")
                decoded_value = self._decode_state_value(encoded_value)
                if decoded_value is not None and (
                    isinstance(decoded_value, bool)
                    or not isinstance(decoded_value, (int, float, np.integer, np.floating))
                ):
                    raise ValueError(f"LOCF value for {name} must be numeric or None")
                decoded_values[name] = decoded_value
            restored_locf[stay_id] = decoded_values

        self.active_windows = restored_windows
        self.locf_state = restored_locf

    def clear_stay_state(self, stay_id):
        """Remove preprocessor state for a discharged stay."""
        key = int(stay_id)
        self.active_windows.pop(key, None)
        self.locf_state.pop(key, None)

    def get_window_id(self, charttime, intime):
        charttime = pd.Timestamp(charttime)
        intime = pd.Timestamp(intime)
        minutes = (charttime - intime).total_seconds() / 60.0
        return int(np.floor(minutes / C.WINDOW_MINUTES))
    def _validate_observations(self, observations):
        required = {"stay_id", "charttime", "vital_sign", "value"}
        missing = required.difference(observations.columns)
        if missing:
            raise RuntimeError(f"Missing observation columns: {sorted(missing)}")
    def add_observation(self, stay_id, charttime, intime, vital_sign, value):
        charttime = pd.Timestamp(charttime)
        intime = pd.Timestamp(intime)
        if charttime < intime:
            raise RuntimeError("Observation occurs before ICU intime")
        window_id = self.get_window_id(charttime, intime)
        key = int(stay_id)
        if key not in self.active_windows:
            self.active_windows[key] = {}
        if window_id not in self.active_windows[key]:
            self.active_windows[key][window_id] = []
        self.active_windows[key][window_id].append({
            "stay_id": key,
            "charttime": charttime,
            "vital_sign": vital_sign,
            "value": value
        })
        return window_id
    def get_partial_window(self, stay_id, window_id, static_data):
        key = int(stay_id)
        if key not in self.active_windows or window_id not in self.active_windows[key]:
            raise RuntimeError(f"No active observations for stay {key}, window {window_id}")
        observations = pd.DataFrame(self.active_windows[key][window_id])
        return self.transform_window(observations, static_data, stay_id, finalize=False)
    def get_final_window(self, stay_id, window_id, static_data):
        key = int(stay_id)
        if key not in self.active_windows or window_id not in self.active_windows[key]:
            raise RuntimeError(f"No active observations for stay {key}, window {window_id}")
        observations = pd.DataFrame(self.active_windows[key][window_id])
        result = self.transform_window(observations, static_data, stay_id, finalize=True)
        del self.active_windows[key][window_id]
        if not self.active_windows[key]:
            del self.active_windows[key]
        return result
    def build_temporal_features(self, observations, stay_id=None):
        data = observations.copy()
        if data.empty:
            data = pd.DataFrame(columns=["stay_id", "charttime", "vital_sign", "value"])
        self._validate_observations(data)
        data["charttime"] = pd.to_datetime(data["charttime"])
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.sort_values("charttime", kind="stable")
        row = {}
        previous = self.locf_state.get(int(stay_id), {}) if stay_id is not None else {}
        for vital_name, prefix in VITAL_FEATURES.items():
            values = data.loc[data["vital_sign"] == vital_name, "value"].dropna()
            if len(values):
                row[f"{prefix}_mean"] = float(values.mean())
                row[f"{prefix}_min"] = float(values.min())
                row[f"{prefix}_max"] = float(values.max())
                row[f"{prefix}_last"] = float(values.iloc[-1])
                row[f"{prefix}_missing"] = 0.0
            else:
                row[f"{prefix}_mean"] = previous.get(f"{prefix}_mean", np.nan)
                row[f"{prefix}_min"] = previous.get(f"{prefix}_min", np.nan)
                row[f"{prefix}_max"] = previous.get(f"{prefix}_max", np.nan)
                row[f"{prefix}_last"] = previous.get(f"{prefix}_last", np.nan)
                row[f"{prefix}_missing"] = 1.0
        row["HR_trend"] = row["Heart_Rate_last"] - row["Heart_Rate_mean"] if np.isfinite(row["Heart_Rate_last"]) and np.isfinite(row["Heart_Rate_mean"]) else np.nan
        row["MAP_trend"] = row["MAP_last"] - row["MAP_mean"] if np.isfinite(row["MAP_last"]) and np.isfinite(row["MAP_mean"]) else np.nan
        row["RR_trend"] = row["Respiratory_Rate_last"] - row["Respiratory_Rate_mean"] if np.isfinite(row["Respiratory_Rate_last"]) and np.isfinite(row["Respiratory_Rate_mean"]) else np.nan
        gcs_values = [row["GCS_Eye_last"], row["GCS_Verbal_last"], row["GCS_Motor_last"]]
        row["GCS_Total_last"] = sum(value if np.isfinite(value) else 0.0 for value in gcs_values)
        return row
    def build_window_row(self, observations, static_data, stay_id=None):
        temporal = self.build_temporal_features(observations, stay_id)
        row = dict(temporal)
        row["anchor_age"] = static_data.get("anchor_age")
        for column in CATEGORICAL_FEATURES:
            row[column] = static_data.get(column)
        return row
    def _create_schema(self):
        fields = [StructField(name, DoubleType(), True) for name in TEMPORAL_FEATURE_NAMES]
        fields.append(StructField("anchor_age", DoubleType(), True))
        fields += [StructField(column, StringType(), True) for column in CATEGORICAL_FEATURES]
        return StructType(fields)
    def _extract_feature_names(self, transformed):
        metadata = transformed.schema["features"].metadata
        if "ml_attr" not in metadata:
            raise RuntimeError("features column does not contain ml_attr metadata")
        attrs = metadata["ml_attr"].get("attrs", {})
        flat = [attribute for group in attrs.values() for attribute in group]
        flat = sorted(flat, key=lambda attribute: attribute["idx"])
        names = [attribute["name"] for attribute in flat]
        if len(names) != C.FEATURE_COUNT:
            raise RuntimeError(f"Expected {C.FEATURE_COUNT} metadata feature names but found {len(names)}")
        if names[:C.TEMPORAL_FEATURES] != TEMPORAL_FEATURE_NAMES:
            raise RuntimeError("Metadata temporal feature order does not match TEMPORAL_FEATURE_NAMES")
        return names
    def _build_missing_mask(self, raw_temporal):
        missing_mask = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
        for index in range(C.TEMPORAL_FEATURES):
            missing_mask[index] = 1.0 if not np.isfinite(raw_temporal[index]) else 0.0
        return missing_mask
    def _impute_temporal(self, raw_temporal):
        temporal = np.asarray(raw_temporal, dtype=np.float32).copy()
        for index in range(C.TEMPORAL_FEATURES):
            if not np.isfinite(temporal[index]):
                temporal[index] = float(self.imputation[TEMPORAL_FEATURE_NAMES[index]])
        if not np.isfinite(temporal).all():
            raise RuntimeError("Invalid temporal values after imputation")
        return temporal
    def transform_window(self, observations, static_data, stay_id=None, finalize=False):
        self.start()
        window = self.build_window_row(observations, static_data, stay_id)
        schema = self._create_schema()
        columns = TEMPORAL_FEATURE_NAMES + ["anchor_age"] + CATEGORICAL_FEATURES
        spark_df = self.spark.createDataFrame(pd.DataFrame([[window[name] for name in columns]], columns=columns), schema=schema)
        transformed = self.pipeline.transform(spark_df)
        if self.feature_names is None:
            self.feature_names = self._extract_feature_names(transformed)
        result = transformed.select(vector_to_array("features").alias("features")).collect()
        if len(result) != 1:
            raise RuntimeError("Expected exactly one transformed feature vector")
        features = np.asarray(result[0]["features"], dtype=np.float32)
        if features.shape != (C.FEATURE_COUNT,):
            raise RuntimeError(f"Expected {C.FEATURE_COUNT} features but got {features.shape[0]}")
        raw_temporal = np.asarray(
            [window[name] for name in TEMPORAL_FEATURE_NAMES],
            dtype=np.float32
        )
        if not np.allclose(features[:C.TEMPORAL_FEATURES], raw_temporal, equal_nan=True, atol=1e-5):
            bad = np.where(
                ~np.isclose(
                    features[:C.TEMPORAL_FEATURES],
                    raw_temporal,
                    equal_nan=True,
                    atol=1e-5
                )
            )[0]
            raise RuntimeError(f"Spark temporal output differs from Python values at: {[TEMPORAL_FEATURE_NAMES[i] for i in bad]}")
        missing_mask = self._build_missing_mask(raw_temporal)
        temporal = self._impute_temporal(raw_temporal)
        static = features[C.TEMPORAL_FEATURES:].copy()
        if static.shape != (C.STATIC_FEATURES,):
            raise RuntimeError(f"Expected {C.STATIC_FEATURES} static features but got {static.shape[0]}")
        if not np.isfinite(static).all():
            raise RuntimeError("Static features contain invalid values")
        if finalize and stay_id is not None:
            self.locf_state[int(stay_id)] = {name: row_value for name, row_value in window.items() if name in TEMPORAL_FEATURE_NAMES}
        return {
            "features": features,
            "temporal": temporal,
            "missing_mask": missing_mask,
            "static": static,
            "feature_names": self.feature_names
        }
    def should_trigger_partial(self, charttime, window_start, trigger_minutes=2):
        charttime = pd.Timestamp(charttime)
        window_start = pd.Timestamp(window_start)
        elapsed_minutes = (charttime - window_start).total_seconds() / 60.0
        return elapsed_minutes >= trigger_minutes
    def build_sequence(self, windows):
        if not windows:
            raise RuntimeError("No windows supplied")
        ordered = sorted(windows, key=lambda item: item["window_id"])
        selected = ordered[-C.MAX_SEQ_LEN:]
        temporal = np.zeros(
            (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES),
            dtype=np.float32
        )
        missing_mask = np.zeros(
            (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES),
            dtype=np.float32
        )
        padding_mask = np.ones(C.MAX_SEQ_LEN, dtype=np.float32)
        static = np.asarray(selected[-1]["static"], dtype=np.float32)
        for index, item in enumerate(selected):
            temporal[index] = np.asarray(item["temporal"], dtype=np.float32)
            missing_mask[index] = np.asarray(item["missing_mask"], dtype=np.float32)
            padding_mask[index] = 0.0
        if temporal.shape != (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES):
            raise RuntimeError("Invalid temporal sequence shape")
        if missing_mask.shape != (C.MAX_SEQ_LEN, C.TEMPORAL_FEATURES):
            raise RuntimeError("Invalid missing-mask sequence shape")
        if padding_mask.shape != (C.MAX_SEQ_LEN,):
            raise RuntimeError("Invalid padding-mask shape")
        if static.shape != (C.STATIC_FEATURES,):
            raise RuntimeError("Invalid static feature shape")
        return {
            "temporal": temporal,
            "missing_mask": missing_mask,
            "padding_mask": padding_mask,
            "static": static,
            "sequence_length": len(selected)
        }
def create_smoke_observations():
    return pd.DataFrame([
        {"stay_id": 99999999, "charttime": "2026-01-01 10:00:00", "vital_sign": "Heart Rate", "value": 82.0},
        {"stay_id": 99999999, "charttime": "2026-01-01 10:01:00", "vital_sign": "SBP", "value": 118.0},
        {"stay_id": 99999999, "charttime": "2026-01-01 10:02:00", "vital_sign": "MAP", "value": 82.0},
        {"stay_id": 99999999, "charttime": "2026-01-01 10:03:00", "vital_sign": "SpO2", "value": 97.0},
        {"stay_id": 99999999, "charttime": "2026-01-01 10:04:00", "vital_sign": "GCS Eye", "value": 4.0},
        {"stay_id": 99999999, "charttime": "2026-01-01 10:05:00", "vital_sign": "GCS Verbal", "value": 5.0},
        {"stay_id": 99999999, "charttime": "2026-01-01 10:06:00", "vital_sign": "GCS Motor", "value": 6.0}
    ])
def create_missing_observations():
    return pd.DataFrame(
        columns=["stay_id", "charttime", "vital_sign", "value"]
    )
def validate_result(result):
    if result["features"].shape != (C.FEATURE_COUNT,):
        raise RuntimeError("Smoke test feature shape failed")
    if result["temporal"].shape != (C.TEMPORAL_FEATURES,):
        raise RuntimeError("Smoke test temporal shape failed")
    if result["missing_mask"].shape != (C.TEMPORAL_FEATURES,):
        raise RuntimeError("Smoke test missing-mask shape failed")
    if result["static"].shape != (C.STATIC_FEATURES,):
        raise RuntimeError("Smoke test static shape failed")
    if not np.isfinite(result["temporal"]).all():
        raise RuntimeError("Smoke test temporal contains invalid values")
    if not np.isfinite(result["static"]).all():
        raise RuntimeError("Smoke test static contains invalid values")
    if len(result["feature_names"]) != C.FEATURE_COUNT:
        raise RuntimeError("Smoke test feature-name count failed")
def test_build_sequence():
    pre = RealtimePreprocessor()
    F, S = C.TEMPORAL_FEATURES, C.STATIC_FEATURES
    def win(i):
        return {
            "window_id": i,
            "temporal": np.full(F, float(i + 1), dtype=np.float32),
            "missing_mask": np.zeros(F, dtype=np.float32),
            "static": np.full(S, 0.5, dtype=np.float32)
        }
    seq = pre.build_sequence([win(2), win(0), win(1)])
    assert seq["sequence_length"] == 3
    assert np.all(seq["padding_mask"][:3] == 0) and np.all(seq["padding_mask"][3:] == 1)
    assert np.array_equal(seq["temporal"][:3, 0], [1.0, 2.0, 3.0])
    assert np.all(seq["temporal"][3:] == 0) and np.all(seq["missing_mask"][3:] == 0)
    seq = pre.build_sequence([win(i) for i in range(130)])
    assert seq["sequence_length"] == C.MAX_SEQ_LEN
    assert seq["temporal"][0, 0] == 11.0
    assert seq["temporal"][-1, 0] == 130.0
    assert np.all(seq["padding_mask"] == 0)
    print("build_sequence test: PASS")
def test_partial_window():
    pre = RealtimePreprocessor()
    static_data = {
        "anchor_age": 65.0,
        "gender": "M",
        "race": None,
        "insurance": "Medicare",
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "first_careunit": "Medical Intensive Care Unit (MICU)"
    }
    pre.add_observation(99999999, "2026-01-01 10:01:00", "2026-01-01 10:00:00", "Heart Rate", 110.0)
    pre.add_observation(99999999, "2026-01-01 10:02:00", "2026-01-01 10:00:00", "SBP", 90.0)
    assert pre.should_trigger_partial("2026-01-01 10:02:00", "2026-01-01 10:00:00")
    assert not pre.should_trigger_partial("2026-01-01 10:01:00", "2026-01-01 10:00:00")
    result = pre.get_partial_window(99999999, 0, static_data)
    validate_result(result)
    print("partial-window preprocessing test: PASS")
def main():
    print("REAL-TIME PREPROCESSING MODULE")
    print(f"Expected features: {C.FEATURE_COUNT}")
    print(f"Temporal features: {C.TEMPORAL_FEATURES}")
    print(f"Static features: {C.STATIC_FEATURES}")
    print(f"Maximum sequence length: {C.MAX_SEQ_LEN}")
    test_build_sequence()
    test_partial_window()
    preprocessor = RealtimePreprocessor()
    preprocessor.start()
    static_data = {
        "anchor_age": 65.0,
        "gender": "M",
        "race": None,
        "insurance": "Medicare",
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "first_careunit": "Medical Intensive Care Unit (MICU)"
    }
    observations = create_smoke_observations()
    result = preprocessor.transform_window(observations, static_data)
    validate_result(result)
    if result["feature_names"][:C.TEMPORAL_FEATURES] != TEMPORAL_FEATURE_NAMES:
        raise RuntimeError("Smoke test temporal feature-name order failed")
    heart_rate_last = result["features"][TEMPORAL_FEATURE_NAMES.index("Heart_Rate_last")]
    hr_trend = result["features"][TEMPORAL_FEATURE_NAMES.index("HR_trend")]
    gcs_total = result["features"][TEMPORAL_FEATURE_NAMES.index("GCS_Total_last")]
    temperature_missing = result["features"][TEMPORAL_FEATURE_NAMES.index("Temperature_missing")]
    temperature_mask = result["missing_mask"][TEMPORAL_FEATURE_NAMES.index("Temperature_mean")]
    if not np.isclose(heart_rate_last, 82.0):
        raise RuntimeError("Heart_Rate_last smoke test failed")
    if not np.isclose(hr_trend, 0.0):
        raise RuntimeError("HR_trend smoke test failed")
    if not np.isclose(gcs_total, 15.0):
        raise RuntimeError("GCS_Total_last smoke test failed")
    if not np.isclose(temperature_missing, 1.0):
        raise RuntimeError("Temperature_missing smoke test failed")
    if not np.isclose(temperature_mask, 1.0):
        raise RuntimeError("Temperature_mean missing-mask smoke test failed")
    print(f"Static feature names: {result['feature_names'][54:64]}")
    print("Partial-missing smoke test: PASS")
    print(f"Heart_Rate_last: {heart_rate_last}")
    print(f"HR_trend: {hr_trend}")
    print(f"GCS_Total_last: {gcs_total}")
    print(f"Temperature_missing: {temperature_missing}")
    print(f"Temperature_mean mask: {temperature_mask}")
    print(f"Output feature count: {len(result['features'])}")
    print(f"Output temporal shape: {result['temporal'].shape}")
    print(f"Output missing-mask shape: {result['missing_mask'].shape}")
    print(f"Output static shape: {result['static'].shape}")
    print(f"Feature names verified: {len(result['feature_names'])}")
    missing_observations = create_missing_observations()
    missing_result = preprocessor.transform_window(missing_observations, static_data)
    validate_result(missing_result)
    expected_ones = [
        i for i, name in enumerate(TEMPORAL_FEATURE_NAMES)
        if (name.endswith(("_mean", "_min", "_max", "_last")) and name != "GCS_Total_last") or name.endswith("_trend")
    ]
    expected = np.zeros(C.TEMPORAL_FEATURES, dtype=np.float32)
    expected[expected_ones] = 1.0
    if not np.array_equal(missing_result["missing_mask"], expected):
        raise RuntimeError("All-missing mask does not match expected pattern")
    flag_idx = [
        i for i, name in enumerate(TEMPORAL_FEATURE_NAMES)
        if name.endswith("_missing")
    ]
    if not np.all(missing_result["features"][flag_idx] == 1.0):
        raise RuntimeError("*_missing flags should be 1.0 for an empty window")
    print(f"All-missing mask ones: {int(missing_result['missing_mask'].sum())}")
    print("All-missing smoke test: PASS")
    print("REAL-TIME PREPROCESSING CHECK PASSED")
    preprocessor.stop()
if __name__ == "__main__":
    main()
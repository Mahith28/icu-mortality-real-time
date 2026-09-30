import numpy as np
NUM_FEATURES = 137
def convert_features(value):
    if value is None:
        result = np.full(NUM_FEATURES, np.nan, dtype=np.float32)
    elif hasattr(value, "toArray"):
        result = np.asarray(value.toArray(), dtype=np.float32)
    elif isinstance(value, dict) and "values" in value:
        values = value["values"]
        if "indices" in value and "size" in value:
            size = int(value["size"])
            if size != NUM_FEATURES: raise ValueError(f"Expected vector size {NUM_FEATURES}, got {size}")
            indices = value["indices"]
            if indices is None:
                result = np.zeros(NUM_FEATURES, dtype=np.float32)
            else:
                if len(indices) != len(values): raise ValueError(f"Sparse vector index/value length mismatch: {len(indices)} != {len(values)}")
                if len(indices) > 0 and (min(indices) < 0 or max(indices) >= NUM_FEATURES): raise ValueError(f"Sparse vector index out of bounds: valid range is 0-{NUM_FEATURES - 1}")

                result = np.zeros(NUM_FEATURES, dtype=np.float32)

                for index, val in zip(indices, values):
                    result[int(index)] = np.nan if val is None else float(val)
        else:
            result = np.asarray(values, dtype=np.float32)
    else:
        result = np.asarray(value, dtype=np.float32)
    if result.shape != (NUM_FEATURES,): raise ValueError(f"Expected {NUM_FEATURES} features, got shape {result.shape}")
    return result
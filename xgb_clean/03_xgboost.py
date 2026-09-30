import os
import gc
import glob
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cupy as cp
import xgboost as xgb
HDFS_TRAIN = "hdfs://localhost:9000/user/mahith/icu/clean_features/train_features"
HDFS_VALIDATION = "hdfs://localhost:9000/user/mahith/icu/clean_features/validation_features"
LOCAL_BASE = "/home/mahith/BDA_PROJECT/Model/xgb_clean/xgb_data"
TRAIN_DIR = os.path.join(LOCAL_BASE, "train_features")
VALIDATION_DIR = os.path.join(LOCAL_BASE, "validation_features")
CACHE_DIR = os.path.join(LOCAL_BASE, "cache")
MODEL_DIR = "/home/mahith/BDA_PROJECT/Model/models"
MODEL_PATH = os.path.join(MODEL_DIR, "icu_mortality_xgboost_clean.json")
EXPECTED_FEATURES = 137
BATCH_SIZE = 100000
MAX_BIN = 128
NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30
SEED = 42
print("ICU MORTALITY - CLEAN XGBOOST GPU EXTERNAL-MEMORY TRAINING")
print("\nCHECKING SOFTWARE")
print(f"XGBoost : {xgb.__version__}")
print(f"CuPy    : {cp.__version__}")
import pyarrow
print(f"PyArrow : {pyarrow.__version__}")
if not hasattr(xgb, "ExtMemQuantileDMatrix"): raise RuntimeError("ExtMemQuantileDMatrix is not available.")
if not hasattr(xgb, "DataIter"): raise RuntimeError("DataIter is not available.")
print("\nCHECKING GPU")
device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
if isinstance(device_name, bytes): device_name = device_name.decode()
free_memory, total_memory = cp.cuda.runtime.memGetInfo()
print(f"GPU : {device_name}")
print(f"GPU memory : {total_memory / (1024 ** 3):.2f} GB")
print(f"GPU free   : {free_memory / (1024 ** 3):.2f} GB")
print("\nPREPARING LOCAL TRAINING DATA")
if os.path.exists(LOCAL_BASE): shutil.rmtree(LOCAL_BASE)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
print("Copying training data from HDFS")
subprocess.run(["hdfs", "dfs", "-get", HDFS_TRAIN + "/*", TRAIN_DIR], check=True)
print("Copying validation data from HDFS")
subprocess.run(["hdfs", "dfs", "-get", HDFS_VALIDATION + "/*", VALIDATION_DIR], check=True)
print("\nCHECKING TRAINING DATA")
train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, "part-*.parquet")))
validation_files = sorted(glob.glob(os.path.join(VALIDATION_DIR, "part-*.parquet")))
print(f"Training partitions   : {len(train_files)}")
print(f"Validation partitions : {len(validation_files)}")
print(f"Batch size            : {BATCH_SIZE:,}")
print(f"Expected features     : {EXPECTED_FEATURES}")
if not train_files: raise RuntimeError("No training Parquet files found.")
if not validation_files: raise RuntimeError("No validation Parquet files found.")
for path in train_files: print(f"Train      : {os.path.basename(path)}")
for path in validation_files: print(f"Validation : {os.path.basename(path)}")
print("\nINSPECTING FEATURES COLUMN STRUCTURE")
sample_df = pd.read_parquet(train_files[0], columns=["features"])
sample_value = sample_df["features"].iloc[0]
print(f"Type : {type(sample_value)}")
if isinstance(sample_value, dict):
    print(f"Vector size : {sample_value.get('size')}")
    print(f"Vector type : {sample_value.get('type')}")
del sample_df, sample_value
gc.collect()
def vector_struct_to_dense(v, expected_len):
    if hasattr(v, "toArray"):
        result = v.toArray().astype(np.float32)
        if len(result) != expected_len: raise RuntimeError(f"Expected {expected_len} values but found {len(result)}")
        return result
    if not isinstance(v, dict): raise TypeError(f"Unexpected vector type: {type(v)}")
    vector_type = v["type"]
    vector_size = v["size"]
    if vector_size != expected_len: raise RuntimeError(f"Expected vector size {expected_len} but found {vector_size}")
    if vector_type == 0:
        dense = np.zeros(expected_len, dtype=np.float32)
        indices = v["indices"]
        values = v["values"]
        if indices is not None and len(indices) > 0: dense[np.asarray(indices, dtype=np.int32)] = np.asarray(values, dtype=np.float32)
        return dense
    if vector_type == 1:
        values = np.asarray(v["values"], dtype=np.float32)
        if len(values) != expected_len: raise RuntimeError(f"Expected {expected_len} values but found {len(values)}")
        return values
    raise RuntimeError(f"Unknown Spark vector type: {vector_type}")
def reconstruct_batch(feature_values):
    X = np.empty((len(feature_values), EXPECTED_FEATURES), dtype=np.float32)
    for i, value in enumerate(feature_values): X[i] = vector_struct_to_dense(value, EXPECTED_FEATURES)
    return X
class ParquetIterator(xgb.DataIter):
    def __init__(self, file_paths, name):
        self.file_paths = file_paths
        self.name = name
        self.current_file = 0
        self.batch_iterator = None
        self.batch_number = 0
        super().__init__(cache_prefix=os.path.join(CACHE_DIR, name.lower()), release_data=True, on_host=True, min_cache_page_bytes=0)
    def next(self, input_data):
        while True:
            if self.batch_iterator is None:
                if self.current_file >= len(self.file_paths): return False
                path = self.file_paths[self.current_file]
                print(f"\nOPENING: {os.path.basename(path)}")
                parquet_file = pq.ParquetFile(path)
                self.batch_iterator = parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["features", "label"])
                self.current_file += 1
                self.batch_number = 0
            try:
                batch = next(self.batch_iterator)
            except StopIteration:
                self.batch_iterator = None
                continue
            self.batch_number += 1
            start = time.time()
            feature_values = batch.column("features").to_pylist()
            labels = batch.column("label").to_numpy(zero_copy_only=False)
            X = reconstruct_batch(feature_values)
            y = np.asarray(labels, dtype=np.float32)
            if X.shape[1] != EXPECTED_FEATURES: raise RuntimeError(f"Expected {EXPECTED_FEATURES} features but found {X.shape[1]}")
            if np.isinf(X).any(): raise RuntimeError("Infinite feature values detected.")
            if not np.isfinite(y).all(): raise RuntimeError("Non-finite label values detected.")
            print(f"{self.name} batch : {self.batch_number}")
            print(f"Rows            : {X.shape[0]:,}")
            print(f"Features        : {X.shape[1]}")
            print(f"CPU data        : {X.nbytes / (1024 ** 2):.1f} MB")
            print(f"NaN count       : {np.isnan(X).sum():,}")
            print(f"Read time       : {time.time() - start:.2f} sec")
            X_gpu = cp.asarray(X)
            y_gpu = cp.asarray(y)
            input_data(data=X_gpu, label=y_gpu)
            del feature_values, X, y, X_gpu, y_gpu, batch
            gc.collect()
            return True
    def reset(self):
        self.current_file = 0
        self.batch_iterator = None
        self.batch_number = 0
print("\nCALCULATING CLASS WEIGHT")
train_positive_count = 0
train_negative_count = 0
for path in train_files:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["label"]):
        labels = batch.column("label").to_numpy(zero_copy_only=False)
        train_positive_count += int(np.sum(labels == 1))
        train_negative_count += int(np.sum(labels == 0))
if train_positive_count == 0: raise RuntimeError("No positive training samples found.")
if train_negative_count == 0: raise RuntimeError("No negative training samples found.")
SCALE_POS_WEIGHT = train_negative_count / train_positive_count
print(f"Training negative samples : {train_negative_count:,}")
print(f"Training positive samples : {train_positive_count:,}")
print(f"scale_pos_weight          : {SCALE_POS_WEIGHT:.6f}")
print("\nSETTING CUPY ASYNC MEMORY ALLOCATOR")
cp.cuda.set_allocator(cp.cuda.MemoryAsyncPool().malloc)
print("CuPy asynchronous memory pool enabled.")
print("\nCREATING TRAINING ITERATOR")
train_iterator = ParquetIterator(train_files, "TRAIN")
print("CREATING VALIDATION ITERATOR")
validation_iterator = ParquetIterator(validation_files, "VALIDATION")
train_matrix = None
validation_matrix = None
booster = None
loaded_booster = None
try:
    print("\nCREATING EXTERNAL-MEMORY TRAINING MATRIX")
    train_start = time.time()
    with xgb.config_context(use_cuda_async_pool=True):
        train_matrix = xgb.ExtMemQuantileDMatrix(train_iterator, max_bin=MAX_BIN, missing=np.nan, ref=None, enable_categorical=False)
    print(f"TRAINING MATRIX CREATED ({(time.time() - train_start) / 60:.2f} minutes)")
    print("\nCREATING EXTERNAL-MEMORY VALIDATION MATRIX")
    validation_start = time.time()
    with xgb.config_context(use_cuda_async_pool=True):
        validation_matrix = xgb.ExtMemQuantileDMatrix(validation_iterator, max_bin=MAX_BIN, missing=np.nan, ref=train_matrix, enable_categorical=False)
    print(f"VALIDATION MATRIX CREATED ({(time.time() - validation_start) / 60:.2f} minutes)")
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc", "aucpr"],
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": MAX_BIN,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "seed": SEED
    }
    print("\nSTARTING XGBOOST TRAINING")
    print(f"Maximum boosting rounds : {NUM_BOOST_ROUND}")
    print(f"Early stopping rounds   : {EARLY_STOPPING_ROUNDS}")
    print(f"Maximum tree bins       : {MAX_BIN}")
    print(f"Features                : {EXPECTED_FEATURES}")
    print(f"scale_pos_weight        : {SCALE_POS_WEIGHT:.6f}")
    print(f"Seed                    : {SEED}")
    training_start = time.time()
    with xgb.config_context(use_cuda_async_pool=True):
        booster = xgb.train(params=params, dtrain=train_matrix, num_boost_round=NUM_BOOST_ROUND, evals=[(train_matrix, "train"), (validation_matrix, "validation")], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose_eval=10)
    print(f"\nXGBOOST TRAINING COMPLETED ({(time.time() - training_start) / 60:.2f} minutes)")
    print("\nEARLY STOPPING RESULTS")
    print(f"Best iteration  : {booster.best_iteration}")
    print(f"Best score      : {booster.best_score}")
    print(f"Boosting rounds : {booster.num_boosted_rounds()}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("\nSAVING XGBOOST MODEL")
    booster.save_model(MODEL_PATH)
    print(f"MODEL SAVED: {MODEL_PATH}")
    print("\nVERIFYING SAVED MODEL")
    loaded_booster = xgb.Booster()
    loaded_booster.load_model(MODEL_PATH)
    print(f"Boosting rounds : {loaded_booster.num_boosted_rounds()}")
    print(f"Best iteration  : {loaded_booster.best_iteration}")
    print("\nVERIFYING FEATURE COUNT")
    feature_count = loaded_booster.num_features()
    print(f"Model features : {feature_count}")
    if feature_count != EXPECTED_FEATURES: raise RuntimeError(f"Expected {EXPECTED_FEATURES} features but model contains {feature_count}")
    print("FEATURE COUNT CHECK PASSED")
finally:
    print("\nCLEANING UP GPU MEMORY")
    if train_matrix is not None: del train_matrix
    if validation_matrix is not None: del validation_matrix
    if train_iterator is not None: del train_iterator
    if validation_iterator is not None: del validation_iterator
    if booster is not None: del booster
    if loaded_booster is not None: del loaded_booster
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
print("XGBOOST GPU TRAINING COMPLETED")
print(f"Model: {MODEL_PATH}")
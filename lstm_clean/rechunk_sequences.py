import os
import pyarrow.parquet as pq
import pyarrow as pa
BASE = "/home/mahith/BDA_PROJECT/Model/lstm_clean/local_sequences"
OUT = "/home/mahith/BDA_PROJECT/Model/lstm_clean/rechunked_sequences"
READ_BATCH_SIZE = 500
ROW_GROUP_SIZE = 500
EXPECTED_ROWS = {"train_sequences": 841619, "validation_sequences": 184926, "test_sequences": 1664239}
for split in ["train_sequences", "validation_sequences", "test_sequences"]:
    source = os.path.join(BASE, split)
    target = os.path.join(OUT, split)
    if not os.path.isdir(source):
        raise FileNotFoundError(f"Source directory not found: {source}")
    os.makedirs(target, exist_ok=True)
    for old_file in os.listdir(target):
        old_path = os.path.join(target, old_file)
        if os.path.isfile(old_path):
            os.remove(old_path)
    source_files = sorted([os.path.join(source, f) for f in os.listdir(source) if f.endswith(".parquet")])
    if not source_files:
        raise RuntimeError(f"No Parquet files found in {source}")
    output_index = 0
    total_rows = 0
    print(f"Starting {split}")
    for source_file in source_files:
        parquet_file = pq.ParquetFile(source_file)
        for batch in parquet_file.iter_batches(batch_size=READ_BATCH_SIZE, use_threads=False):
            table = pa.Table.from_batches([batch])
            output_file = os.path.join(target, f"part-{output_index:05d}.parquet")
            pq.write_table(table, output_file, compression="snappy", row_group_size=ROW_GROUP_SIZE)
            output_index += 1
            total_rows += table.num_rows
            del table
            del batch
        del parquet_file
    print(f"{split} rows written: {total_rows:,}")
    if total_rows != EXPECTED_ROWS[split]:
        raise RuntimeError(f"{split}: expected {EXPECTED_ROWS[split]:,} rows but wrote {total_rows:,}")
    output_files = [f for f in os.listdir(target) if f.endswith(".parquet")]
    print(f"{split} output files: {len(output_files):,}")
    print(f"{split} verification passed")
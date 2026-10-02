from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import *
KAFKA_BOOTSTRAP = "localhost:9092"
RISK_TOPIC = "risk_scores"
EXPLANATION_TOPIC = "risk_explanations"
RISK_OUTPUT = "hdfs://localhost:9000/user/mahith/icu/realtime/risk_scores"
EXPLANATION_OUTPUT = "hdfs://localhost:9000/user/mahith/icu/realtime/risk_explanations"
RISK_CHECKPOINT = "hdfs://localhost:9000/user/mahith/icu/realtime/checkpoints/risk_scores"
EXPLANATION_CHECKPOINT = "hdfs://localhost:9000/user/mahith/icu/realtime/checkpoints/risk_explanations"
spark = SparkSession.builder.appName("ICU-Kafka-to-HDFS-Continuous").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
risk_schema = StructType([
    StructField("schema_version", StringType()),
    StructField("event_time", StringType()),
    StructField("status", StringType()),
    StructField("stay_id", LongType()),
    StructField("window_id", IntegerType()),
    StructField("charttime", StringType()),
    StructField("window_start", StringType()),
    StructField("window_end", StringType()),
    StructField("sequence_length", IntegerType()),
    StructField("prediction", IntegerType()),
    StructField("threshold", DoubleType()),
    StructField("xgb_raw_probability", DoubleType()),
    StructField("xgb_probability", DoubleType()),
    StructField("lstm_raw_probability", DoubleType()),
    StructField("lstm_probability", DoubleType()),
    StructField("ensemble_probability", DoubleType()),
    StructField("model_version", StringType()),
    StructField("latency_ms", DoubleType()),
    StructField("features", ArrayType(DoubleType())),
    StructField("shap_status", StringType()),
    StructField("top_features", ArrayType(StructType([
        StructField("feature", StringType()),
        StructField("value", DoubleType()),
        StructField("shap_value", DoubleType()),
        StructField("abs_shap_value", DoubleType())
    ])))
])
explanation_schema = StructType([
    StructField("schema_version", StringType()),
    StructField("event_time", StringType()),
    StructField("status", StringType()),
    StructField("stay_id", LongType()),
    StructField("window_id", IntegerType()),
    StructField("charttime", StringType()),
    StructField("window_start", StringType()),
    StructField("window_end", StringType()),
    StructField("sequence_length", IntegerType()),
    StructField("prediction", IntegerType()),
    StructField("threshold", DoubleType()),
    StructField("xgb_raw_probability", DoubleType()),
    StructField("xgb_probability", DoubleType()),
    StructField("lstm_raw_probability", DoubleType()),
    StructField("lstm_probability", DoubleType()),
    StructField("ensemble_probability", DoubleType()),
    StructField("model_version", StringType()),
    StructField("latency_ms", DoubleType()),
    StructField("features", ArrayType(DoubleType())),
    StructField("shap_status", StringType()),
    StructField("top_features", ArrayType(StructType([
        StructField("feature", StringType()),
        StructField("value", DoubleType()),
        StructField("shap_value", DoubleType()),
        StructField("abs_shap_value", DoubleType())
    ])))
])
risk_stream = spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP).option("subscribe", RISK_TOPIC).option("startingOffsets", "latest").option("failOnDataLoss", "false").load()
risk_data = risk_stream.select(from_json(col("value").cast("string"), risk_schema).alias("data")).select("data.*")
risk_query = risk_data.writeStream.format("parquet").option("path", RISK_OUTPUT).option("checkpointLocation", RISK_CHECKPOINT).outputMode("append").trigger(processingTime="5 seconds").start()
explanation_stream = spark.readStream.format("kafka").option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP).option("subscribe", EXPLANATION_TOPIC).option("startingOffsets", "latest").option("failOnDataLoss", "false").load()
explanation_data = explanation_stream.select(from_json(col("value").cast("string"), explanation_schema).alias("data")).select("data.*")
explanation_query = explanation_data.writeStream.format("parquet").option("path", EXPLANATION_OUTPUT).option("checkpointLocation", EXPLANATION_CHECKPOINT).outputMode("append").trigger(processingTime="5 seconds").start()
spark.streams.awaitAnyTermination()
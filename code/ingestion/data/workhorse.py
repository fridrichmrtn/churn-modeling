# Databricks notebook source
### NOTE: ADD DOCSTRINGS

#
##
### REES46

def _rees46_load(data_path):
    import pyspark.sql.functions as f
    from pyspark.sql.types import StructType, IntegerType, DoubleType, TimestampType, StringType
    # load raw data
    events_schema = StructType()\
        .add("event_time",TimestampType(), True)\
        .add("event_type", StringType(), True)\
        .add("product_id", IntegerType(), True)\
        .add("category_id", StringType(), True)\
        .add("category_code", StringType(), True)\
        .add("brand", StringType(), True)\
        .add("price", DoubleType(), True)\
        .add("user_id", IntegerType(), True)\
        .add("user_session", StringType(), True)
    target_files = [tf for tf in dbutils.fs.ls(data_path)\
        if "csv.gz" in tf.path]
    for tf in target_files:
        if "events" not in locals():
            events = spark.read.csv(tf.path, schema=events_schema, header=True)
        else:
            events = events.union(spark.read.csv(tf.path, schema=events_schema, header=True))
    return events.repartition(200)

# FIX TIMESTAMP, STANDARDIZE AND RENAME COLUMNS, POSSIBLY ADD ID-REMAPP
def _rees46_fix(events):
    import pyspark.sql.functions as f
    # timestamp and names
    events = (events
        .withColumn("event_time", f.col("event_time")+f.expr("INTERVAL 6 HOURS"))
        .withColumnRenamed("user_session", "user_session_id")
        .withColumnRenamed("event_type", "event_type_name")) 
    # add a few useful columns
    events = (events.withColumn("view", (f.col("event_type_name")=="view").cast("int"))
        .withColumn("cart", (f.col("event_type_name")=="cart").cast("int"))
        .withColumn("purchase", (f.col("event_type_name")=="purchase").cast("int"))
        .withColumn("revenue", f.col("purchase")*f.col("price")))
    return events

# FILTER USERS    
def _rees46_filter(events):
    import pyspark.sql.functions as f
    user_filter = (events.where(f.col("event_type_name")=="purchase")
         .groupBy("user_id")
             .agg(f.countDistinct(f.col("user_session_id")).alias("purchase_count"))
         .where(f.col("purchase_count")>=2)
         .select("user_id"))
    return events.join(user_filter, on=["user_id"], how="inner")


# COMMAND ----------


#
##
### LOADING RETAIL ROCKET


#
##
### LOADING OLIST

# COMMAND ----------

#
##
### CONSTRUCT EVENTS

load_transform_dict = {"rees46":{"load":_rees46_load, "fix":_rees46_fix, "filter":_rees46_filter}}

def construct_events(data_path, dataset_name):
    # unpack
    load = load_transform_dict[dataset_name]["load"]
    fix = load_transform_dict[dataset_name]["fix"]
    filter = load_transform_dict[dataset_name]["filter"]
    
    # load, fix, and filter
    events = load(data_path)
    events = fix(events)
    events = filter(events)
    return events

#
##
### SAVE TO DELTA

def save_events(events, data_path):
    # do the repartitioning
    (events
         .write.format("delta")#.partitionBy("user_id")
         .mode("overwrite").option("overwriteSchema", "true")
         .save(data_path))

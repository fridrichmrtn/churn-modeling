# Databricks notebook source
### NOTE: ADD DOCSTRINGS

#
##
### REES46

def _rees46_load(dataInPath):
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
    target_files = [tf for tf in dbutils.fs.ls(dataInPath)\
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
         .where(f.col("purchase_count")>=10)
         .select("user_id"))
    return events.join(user_filter, on=["user_id"], how="inner")

# PUTTING EVERYTHING TOGETHER
def rees46_load_transform(dataInPath):
    # load, fix, and filter
    events = _rees46_load(dataInPath)
    events = _rees46_fix(events)
    events = _rees46_filter(events)
    return events

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
### SAVE TO DELTA

def save_events(events, dataOutPath):
    # do the repartitioning
    (events
         .write.format("delta")#.partitionBy("user_id")
         .mode("overwrite").option("overwriteSchema", "true")
         .save(dataOutPath))

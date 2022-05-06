# Databricks notebook source
# MAGIC %md
# MAGIC # User churn model
# MAGIC 
# MAGIC ### Data loading

# COMMAND ----------

import pyspark.sql.functions as f

# load raw data
DATA_IN = "dbfs:/mnt/rees46/raw/concatenated/sample/"
categories = spark.read.parquet(DATA_IN+"categories")
event_types = spark.read.parquet(DATA_IN+"event_types")
events = spark.read.parquet(DATA_IN+"events")
products = spark.read.parquet(DATA_IN+"products")

# denorm
events = events.join(event_types, on="event_type_id", how="inner")\
    .join(products, on="product_id", how="inner")\
    .join(categories, on="category_id", how="left")\
    .withColumn("view", (f.col("event_type_name")=="view").cast("int"))\
    .withColumn("cart", (f.col("event_type_name")=="cart").cast("int"))\
    .withColumn("purchase", (f.col("event_type_name")=="purchase").cast("int"))\
    .withColumn("revenue", f.col("purchase")*f.col("price"))
events = events.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate sessions

# COMMAND ----------

# sessions agg
sessions = events.groupBy("user_session_id", "user_id").agg(    
    # time chars
    f.min("event_time").alias("session_start"), f.max("event_time").alias("session_end"),
    ((f.max("event_time").alias("session_end")-f.min("event_time").alias("session_end")).cast("long")/60).alias("session_length"),
    # clicks
    f.sum("view").alias("view_count"), f.sum("cart").alias("cart_count"), f.sum("purchase").alias("purchase_count"),
    f.count("user_id").alias("click_count"),
    # revenue
    f.sum((f.col("view")*f.col("price"))).alias("view_revenue"), f.sum((f.col("cart")*f.col("price"))).alias("cart_revenue"),
    f.sum((f.col("purchase")*f.col("price"))).alias("purchase_revenue"))

from pyspark.sql.window import Window
ws = Window().partitionBy("user_id").orderBy("session_start")
last_session_start = sessions.agg(f.max(f.col("session_start")).alias("lsd"))\
    .collect()[0].__getitem__("lsd")

sessions = sessions\
    .withColumn("session_number",f.row_number().over(ws))\
    .withColumn("inter_session_time", (f.col("session_start")-f.lag("session_start",1).over(ws)).cast("long")/(3600*24))\
    .withColumn("session_recency", ((last_session_start-f.col("session_start")).cast("long")/(3600*24)))

sessions.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate users

# COMMAND ----------

# get last session date


wr = Window().partitionBy("user_id").orderBy("session_number")

sessions.groupBy("user_id").agg(
    # recency
        (f.min("session_recency")).alias("session_recency"),
        f.mean("inter_session_time").alias("session_recency_avg"),
        f.stddev_samp("inter_session_time").alias("session_recency_sd"),
        (f.stddev_samp("inter_session_time")/f.mean("inter_session_time")).alias("session_recency_cv"),
        (f.max("session_recency")).alias("user_maturity"),
    # frequency
        f.max("session_number").alias("session_count"),
        (f.max("session_number")/f.max("session_recency").alias("session_count_ratio"),
        f.sum("click_count").alias("click_count"),
        (f.sum("click_count")/f.max("session_number")).alias("click_count_ratio")
        # dont forget transactions and conversions 

).show()

# COMMAND ----------

sessions.where(f.col("user_id")==368).show()

# COMMAND ----------

sessions.printSchema()

# COMMAND ----------

last_session_date

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construct preferences

# COMMAND ----------

# dont forget to do the data split

# rectify the timezone

# implement rfm on both ses and tran

# implement preferences on product basis through reco
    # als
    # w2v
    # others?

# what is others
    # hour
    # day of month
    # month

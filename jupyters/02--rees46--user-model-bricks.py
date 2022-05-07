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
        # overall session properties
        f.min("event_time").alias("start"), f.max("event_time").alias("end"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/60).alias("length"),
        # date-time components
        # try 
    
        #
        f.make_date(f.year(f.min("event_time")),f.month(f.min("event_time")),f.lit(1)).alias("start_monthgroup"),
        f.year(f.min("event_time")).alias("start_year"), f.dayofyear(f.min("event_time")).alias("start_yearday"), 
        f.month(f.min("event_time")).alias("start_month"), f.dayofmonth(f.min("event_time")).alias("start_monthday"),
        f.weekofyear(f.min("event_time")).alias("start_week"), f.dayofweek(f.min("event_time")).alias("start_weekday"),
        (f.when(f.dayofweek(f.min("event_time"))==1,1).when(f.dayofweek(f.min("event_time"))==7,1).otherwise(0)).alias("start_isweekend"),
        f.hour(f.min("event_time")).alias("start_hour"),
    
    # events
        # clicks
        f.max(f.when(f.col("event_type_name")=="purchase",1).otherwise(0)).alias("haspurchase"),
        f.count("user_id").alias("click_count"), f.sum("view").alias("view_count"), 
        f.sum("cart").alias("cart_count"), f.sum("purchase").alias("purchase_count"),    
        # time to action
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.count("user_id"))).alias("time_to_click"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("view"))).alias("time_to_view"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("cart"))).alias("time_to_cart"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("purchase"))).alias("time_to_purchase"),    
    
    # revenue
        # sums
        f.sum((f.col("view")*f.col("price"))).alias("view_revenue"), f.sum((f.col("cart")*f.col("price"))).alias("cart_revenue"),
        f.sum((f.col("purchase")*f.col("price"))).alias("purchase_revenue"),
        # time to revenue
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("price"))).alias("time_to_click_revenue"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60* f.sum((f.col("view")*f.col("price"))))).alias("time_to_view_revenue"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60* f.sum((f.col("cart")*f.col("price"))))).alias("time_to_cart_revenue"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60* f.sum((f.col("purchase")*f.col("price"))))).alias("time_to_purchase_revenue"))

# windowing
from pyspark.sql.window import Window
ws = Window().partitionBy("user_id").orderBy("start")
last_session_start = sessions.agg(f.max(f.col("start")).alias("lsd"))\
    .collect()[0].__getitem__("lsd")

sessions = sessions\
    .withColumn("session_number",f.row_number().over(ws))\
    .withColumn("inter_session_time", (f.col("start")-f.lag("start",1).over(ws)).cast("long")/(3600*24))\
    .withColumn("session_recency", ((last_session_start-f.col("start")).cast("long")/(3600*24)))
purchases = sessions.where(f.col("haspurchase")==1)\
    .withColumn("purchase_number",f.row_number().over(ws))\
    .withColumn("inter_purchase_time", (f.col("start")-f.lag("start",1).over(ws)).cast("long")/(3600*24))\
    .withColumn("purchase_recency", ((last_session_start-f.col("start")).cast("long")/(3600*24)))\
    .select("user_session_id", "purchase_number", "inter_purchase_time", "purchase_recency")

sessions = sessions.join(purchases,["user_session_id"], "left")
sessions.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### WHAT TO DO TOMORROW
# MAGIC * expand the cols
# MAGIC * add handcrafted interactions
# MAGIC * add grouping, incl full lhs
# MAGIC * add lags for revenue, sessions, transactions, consider smoothing
# MAGIC 
# MAGIC ### WHAT TO DO THE DAY AFTER TOMORROW
# MAGIC * pick preference representation
# MAGIC * construct preference vec for each user
# MAGIC 
# MAGIC ### NEXT?
# MAGIC * TARGET VECTORS
# MAGIC * GENERAL FILTERS
# MAGIC * ML PIPE

# COMMAND ----------

# test the aggs
cols = ["time_to_view_revenue", "time_to_view", "view_count"]
agg_funcs = [f.mean, f.sum, f.min, f.max, f.stddev_samp]
agg_exp = [f(c).alias(c+"_"+str(f.__name__).split("_")[0])for f in agg_funcs for c in cols]
sessions.groupBy("user_id").agg(*agg_exp).show()

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
    (f.max("session_number")/f.max("session_recency")).alias("session_count_daily_ratio"),
    f.sum("click_count").alias("click_count"),
    (f.sum("click_count")/f.max("session_number")).alias("click_count_ratio"),
    f.sum(f.when(f.col("purchase_count")>0,1).otherwise(0)).alias("transaction_count"),
    (f.sum(f.when(f.col("purchase_count")>0,1).otherwise(0))/f.max("session_number")).alias("transaction_count_ratio"),
    # monetary
    (f.sum("view_revenue")).alias("session_viewed_revenue"),
    (f.sum("cart_revenue")).alias("session_cart_revenue"),
    f.sum("purchase_revenue").alias("session_purchase_revenue"),
    # others
        # date-time NOTE: push part of the calculations to the sessions level
    (f.mean(f.hour(f.col("session_start")))).alias("session_hour_avg"),
    (f.stddev_samp(f.hour(f.col("session_start")))).alias("session_hour_sd"),
    (f.mean(f.dayofweek(f.col("session_start")))).alias("session_wday_avg"),
    (f.stddev_samp(f.dayofweek(f.col("session_start")))).alias("session_wday_sd"),
    f.mean(f.when(f.dayofweek(f.col("session_start"))==1,1)\
           .when(f.dayofweek(f.col("session_start"))==7,1).otherwise(0)\
          ).alias("session_weekend_ratio"),
        # intermeasures
    
        # session length
    f.mean("session_length").alias("session_length_avg")
    
).select("user_id", "session_weekend_ratio").show()

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

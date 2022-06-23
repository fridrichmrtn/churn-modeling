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
         .where(f.col("purchase_count")>=10)
         .select("user_id"))
    return events.join(user_filter, on=["user_id"], how="inner")

# COMMAND ----------


#
##
### RETAIL ROCKET

def _retailrocket_load(data_path):
    import pyspark.sql.functions as f
    from pyspark.sql.types import StructType, IntegerType, StringType, LongType 

    # load raw data
    events_schema = StructType()\
        .add("timestamp", LongType(), False)\
        .add("visitorid", IntegerType(), False)\
        .add("event", StringType(), False)\
        .add("itemid", IntegerType(), False)\
        .add("transactionid", IntegerType(), True)

    properties_schema = StructType()\
        .add("timestamp", LongType(), False)\
        .add("itemid", IntegerType(), False)\
        .add("property", StringType(), True)\
        .add("value", StringType(), True)

    events = spark.read.csv(data_path+"events.csv",
        schema=events_schema, header=True).repartition(200)
    item_properties = spark.read.csv(data_path+"item_properties*",
        schema=properties_schema, header=True)
    return {"events":events, "item_properties":item_properties}

# FIX
def _retailrocket_fix(events_dict):
    import numpy as np
    import pyspark.sql.functions as f
    from pyspark.sql.window import Window
    import pyspark.pandas as ps
    
    events = events_dict["events"]
    item_properties = events_dict["item_properties"]
    del events_dict

    max_timestamp =  int(np.max([item_properties.select("timestamp")\
            .agg(f.max(f.col("timestamp")).alias("mts")).collect()[0]["mts"],
        events.select("timestamp")\
             .agg(f.max(f.col("timestamp")).alias("mts")).collect()[0]["mts"]]))

    # item-prop transforms
    item_properties = item_properties.where((f.col("property").isin(["categoryid","790"])))\
        .join(events.select("itemid").dropDuplicates(), on=["itemid"], how="inner")
    item_properties = item_properties.replace({"categoryid":"categoryid", "790":"price"}, subset=["property"])\
        .withColumn("value", f.regexp_replace("value", "n", "").cast("float"))

    w = Window.partitionBy(["itemid","property"]).orderBy("timestamp")
    item_properties = item_properties\
        .withColumn("lag_value", f.lag(f.col("value")).over(w))\
        .where(f.col("lag_value").isNull()|(f.col("lag_value")!=f.col("value")))\
        .withColumn("lead_timestamp", f.lead(f.col("timestamp")).over(w))\
        .na.fill(value=max_timestamp, subset=["lead_timestamp"])\
        .select(f.col("timestamp").alias("valid_start"), f.col("lead_timestamp").alias("valid_end"),
            f.col("itemid"), f.col("property"), f.col("value"),
                (f.col("lead_timestamp")-f.col("timestamp")).alias("time_valid"))\
        .persist()
    
    # events
    events = events.join(item_properties.select("itemid").dropDuplicates(),
        on=["itemid"], how="inner")
    events = events.join(item_properties, on=["itemid"], how="inner").\
        where((f.col("timestamp")>=f.col("valid_start")) & (f.col("timestamp")<f.col("valid_end")))
    events = events.groupBy(["timestamp", "itemid", "visitorid", "event", "transactionid"])\
        .pivot("property").agg(f.first("value"))\
        .withColumn("categoryid", f.col("categoryid").cast("int"))\
        .withColumn("timestamp", f.to_timestamp(f.col("timestamp").cast("Long")/1000-7*60*60)).persist()
    
    # fillna - not very nice
    top_item_properties = ps.DataFrame(item_properties)\
        .groupby(["itemid","property","value"],
            as_index=False).time_valid.sum().sort_values("time_valid")\
                .groupby(["itemid","property"]).tail(1)
    top_item_properties = top_item_properties.pivot_table(index=["itemid"],
        columns="property", values="value").reset_index()
    events = ps.DataFrame(events)
    events = events.merge(top_item_properties, on="itemid")
    events.loc[events.categoryid_x.isna(),
        ["categoryid_x"]] = events["categoryid_y"]
    events.loc[events.price_x.isna(),
        ["price_x"]] = events["price_y"]
    events.rename({"categoryid_x":"categoryid", "price_x":"price"},
        axis=1, inplace=True)
    events = events.loc[:,["timestamp","visitorid","itemid",
        "event", "categoryid", "price"]].to_spark()

    # sessionid
    w = Window().partitionBy("visitorid").orderBy("timestamp")
    events = (events.withColumn("row_num", f.row_number().over(w))
        .withColumn("time_diff", (f.col("timestamp")>f.lag("timestamp",1).over(w)
            +f.expr("INTERVAL 30 MINUTE"))|f.lag("timestamp",1).over(w).isNull())
        .withColumn("vis_diff", f.lead("visitorid",1).over(w).isNull())
        .withColumn("is_start", (f.col("timestamp")>f.lag("timestamp",1).over(w)
            +f.expr("INTERVAL 30 MINUTE"))|f.lag("timestamp",1).over(w).isNull()
                |f.lead("visitorid",1).over(w).isNull())
        .withColumn("is_end", f.lead("is_start",1).over(w)
            |f.lead("is_start",1).over(w).isNull())
        .withColumn("is_of_interest", f.col("is_start")|f.col("is_end")))

    gw = Window().partitionBy("visitorid").orderBy("row_num")
    lw = Window.orderBy(f.lit("m"))
    groups = (events.filter(f.col("is_start"))
        .withColumn("row_end", f.coalesce(f.lead("row_num",1).over(gw), f.col("row_num")))
        .withColumn("diff", f.col("row_end")-f.col("row_num")).filter(f.col("is_start"))
        .withColumn("sessionid", f.row_number().over(lw))
        .select("visitorid", "sessionid", f.col("row_num").alias("row_start"), "row_end"))

    events = (events.alias("e").join(groups.alias("g"),
        (f.col("e.visitorid")==f.col("g.visitorid")) & 
        (f.col("e.row_num")>=f.col("g.row_start")) & 
        (f.col("e.row_num")<f.col("g.row_end")),how="inner")
            .select(
                f.col("e.visitorid").alias("user_id"), f.col("e.timestamp").alias("event_time"),
                f.col("g.sessionid").alias("user_session_id"), f.col("e.event").alias("event_type_name"),
                f.col("e.itemid").alias("product_id"), f.col("e.categoryid").alias("category_id"),
                f.col("e.price").cast("double")))
    
    remap_dict = {"addtocart":"cart", "view":"view", "transaction":"purchase"}
    events = (events.replace(to_replace=remap_dict, subset=["event_type_name"])
        .withColumn("view", (f.col("event_type_name")=="view").cast("int"))
        .withColumn("cart", (f.col("event_type_name")=="cart").cast("int"))
        .withColumn("purchase", (f.col("event_type_name")=="purchase").cast("int"))
        .withColumn("revenue", f.col("purchase")*f.col("price")))
    return events

def _retailrocket_filter(events):
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
### CONSTRUCT EVENTS

load_transform_config = {
    "rees46":{"load":_rees46_load, "fix":_rees46_fix, "filter":_rees46_filter, "data":"dbfs:/mnt/rees46/"},
    "retailrocket":{"load":_retailrocket_load, "fix":_retailrocket_fix, "filter":_retailrocket_filter, "data":"dbfs:/mnt/retailrocket/"}}


    
def construct_events(dataset_name):
    # unpack
    data_path = load_transform_config[dataset_name]["data"]+"raw/"
    load = load_transform_config[dataset_name]["load"]
    fix = load_transform_config[dataset_name]["fix"]
    filt = load_transform_config[dataset_name]["filter"]
    
    # load, fix, and filter
    events = load(data_path)
    events = fix(events)
    events = filt(events)
    return events

#
##
### SAVE TO DELTA

def save_events(events, dataset_name):
    # do the repartitioning
    data_path = load_transform_config[dataset_name]["data"]
    data_path += "delta/events"
    #_rm_dir(data_path)
    (events
         .write.format("delta")#.partitionBy("user_id")
         .mode("overwrite")#.option("overwriteSchema", "true")
         .save(data_path))

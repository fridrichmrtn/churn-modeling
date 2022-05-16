# Databricks notebook source
# MAGIC %md
# MAGIC # Preliminary data munging

# COMMAND ----------

mnts = dbutils.fs.ls("/mnt/")
i=0
while i<len(mnts):
    if "rees46" not in mnts[0].path:
        print("Mounting rees46...")
        dbutils.fs.mount(
          source = "wasbs://rees46@churndatastore.blob.core.windows.net",
          mount_point = "/mnt/rees46",
          extra_configs = {"fs.azure.account.key.churndatastore.blob.core.windows.net":\
              dbutils.secrets.get(scope="scp", key = "churndata-key")})
        break
    else:
        i+=1
if i==len(mnts):
    print("Rees46 already mounted.")

# COMMAND ----------

from pyspark.sql.types import StructType, IntegerType, DoubleType, TimestampType, StringType

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

# load raw data
target_files = [tf for tf in dbutils.fs.ls("dbfs:/mnt/rees46/raw")\
    if "csv.gz" in tf.path]
for tf in target_files:
    if "events" not in locals():
        events = spark.read.csv(tf.path, schema=events_schema, header=True)
    else:
        events = events.union(spark.read.csv(tf.path, schema=events_schema, header=True))
events.write.mode('overwrite').parquet("dbfs:/mnt/rees46/raw/_")
events.printSchema()    

# COMMAND ----------

import pyspark.sql.functions as f
events = spark.read.parquet("dbfs:/mnt/rees46/raw/_")\
    .repartition(200).persist()#sample(fraction=1.0, seed=202205)
events.rdd.getNumPartitions()

# COMMAND ----------

# drop na user sessions
events = events.na.drop(subset=["user_session"]) # drop rswous
# construct id cols
def add_ind(df, colmn):
    from pyspark.sql.functions import dense_rank
    from pyspark.sql.window import Window
    w = Window.orderBy(colmn)
    return df.withColumn("new_"+colmn, dense_rank().over(w))

col_to_map = ["event_type", "product_id", "category_id", "user_id", "user_session"]
for c in col_to_map:
    events = add_ind(events, c)
# rename
events = events.select("event_time", "event_type", "category_code", "brand", "price",
     "new_event_type", "new_product_id", "new_category_id", "new_user_id",
         "new_user_session")\
    .withColumnRenamed("new_event_type", "event_type_id")\
    .withColumnRenamed("new_product_id", "product_id")\
    .withColumnRenamed("new_category_id", "category_id")\
    .withColumnRenamed("new_user_id", "user_id")\
    .withColumnRenamed("new_user_session", "user_session_id")

#from pyspark.storagelevel import StorageLevel as sl
events = events.persist()
events.show(3)    

# COMMAND ----------

# pushdown tables
def save_tables(df, location):
    from pyspark.sql.window import Window
    import pyspark.sql.functions as f
    # event types
    df.select(f.col("event_type_id"),
        f.col("event_type").alias("event_type_name")).dropDuplicates()\
        .write.mode("overwrite").parquet(location+"event_types")
    # products
    wp = Window.partitionBy("product_id").orderBy("event_time")\
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df.withColumn("last_brand", f.last("brand", True).over(wp))\
        .withColumn("last_category_id", f.last("category_id", True).over(wp))\
        .select(f.col("product_id"), f.col("last_category_id").alias("category_id"),
               f.col("last_brand").alias("brand")).dropDuplicates()\
        .write.mode("overwrite").parquet(location+"products")
    # categories
    wc = Window.partitionBy("category_id").orderBy("event_time")\
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    df.withColumn("last_category_code", f.last("category_code", True).over(wc))\
        .select(f.col("category_id"), f.col("last_category_code").alias("category_code")\
                   ).dropDuplicates()\
        .write.mode("overwrite").parquet(location+"categories")
    # events
    df.select(["event_time", "user_id", "product_id", "event_type_id", "price",
               "user_session_id"])\
        .write.mode("overwrite").parquet(location+"events")

DATA_OUT = "dbfs:/mnt/rees46/raw/concatenated/"
events.write.parquet(DATA_OUT+"_", mode="overwrite")
save_tables(events, DATA_OUT+"full/")
save_tables(events.sample(fraction=.01, seed=202205), DATA_OUT+"sample/") 

# COMMAND ----------

# cleanup
#dbutils.fs.rm("dbfs:/mnt/rees46/temp/events", True)
#dbutils.fs.rm("dbfs:/mnt/rees46/temp", True)
#del events

# COMMAND ----------

# consider registering to the delta

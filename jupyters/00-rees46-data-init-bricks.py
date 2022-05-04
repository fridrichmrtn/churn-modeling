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

# load raw data
target_files = [tf for tf in dbutils.fs.ls("dbfs:/mnt/rees46/raw")\
    if "csv.gz" in tf.path]
for tf in target_files:
    if "events" not in locals():
        events = spark.read.csv(tf.path, header=True)
    else:
        events = events.union(spark.read.csv(tf.path, header=True))
events.printSchema()        

# COMMAND ----------

# convert dtypes, possibly move to read part
import pyspark.sql.functions as f
events = events.withColumn("event_time", f.to_timestamp(f.col("event_time")))\
    .withColumn("product_id", f.col("product_id").cast("integer"))\
    .withColumn("category_id", f.col("category_id"))\
    .withColumn("price", f.col("price").cast("double"))\
    .withColumn("user_id", f.col("user_id").cast("integer"))
events.printSchema()


# COMMAND ----------

events.write.parquet("dbfs:/mnt/rees46/temp")
events = spark.read.parquet("dbfs:/mnt/rees46/temp")
events = events.persist()
events.show() # instantiate

# COMMAND ----------

# construct id cols
def add_ind(df, colmn):
    from pyspark.sql.functions import dense_rank
    from pyspark.sql.window import Window
    w = Window.orderBy(colmn)
    return df.withColumn("new_"+colmn, dense_rank().over(w))

col_to_map = ["event_type", "product_id", "category_id", "user_id", "user_session"]
for c in col_to_map:
    events = add_ind(events, c)
events = events.na.drop(subset=["user_session"]) # remove rows wo user_session   
#events.show(3)    

# COMMAND ----------

events = events.select("event_time", "event_type", "category_code", "brand", "price",
     "new_event_type", "new_product_id", "new_category_id", "new_user_id",
         "new_user_session")\
    .withColumnRenamed("new_event_type", "event_type_id")\
    .withColumnRenamed("new_product_id", "product_id")\
    .withColumnRenamed("new_category_id", "category_id")\
    .withColumnRenamed("new_user_id", "user_id")\
    .withColumnRenamed("new_user_session", "user_session_id")
events = events.persist()    
events.show(3)  

# COMMAND ----------

# pushdown tables
def save_tables(df, location):
    # carve-out the tabs
    event_types = df.select(f.col("event_type_id"),
        f.col("event_type").alias("event_type_name")).dropDuplicates()
    # carve-out remaining tabs based on the last observed vals
    products = df.groupBy("product_id").agg(f.max("event_time").alias("event_time"))\
        .join(df, on=["product_id", "event_time"], how="inner")\
            .select("product_id", "category_id", "brand").dropDuplicates()
    categories = df.groupBy("category_id").agg(f.max("event_time").alias("event_time"))\
        .join(df, on=["category_id", "event_time"], how="inner")\
            .select("category_id", "category_code").dropDuplicates()
    events = df.select(["event_time", "user_id", "product_id", "event_type_id", "price", "user_session_id"])
    # push down
    events.write.parquet(location+"events", mode="overwrite")
    products.write.parquet(location+"products", mode="overwrite")
    categories.write.parquet(location+"categories", mode="overwrite")
    event_types.write.parquet(location+"event_types", mode="overwrite")

DATA_OUT = "dbfs:/mnt/rees46/raw/concatenated/"
events.write.parquet(DATA_OUT+"_", mode="overwrite")
save_tables(events, DATA_OUT+"full/")
save_tables(events.sample(fraction=.05, seed=202205), DATA_OUT+"sample/") 

# COMMAND ----------

# cleanup
#dbutils.fs.rm("dbfs:/mnt/rees46/temp/events", True)
#dbutils.fs.rm("dbfs:/mnt/rees46/temp", True)
#del events

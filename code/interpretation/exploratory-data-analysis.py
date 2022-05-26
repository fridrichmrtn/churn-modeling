# Databricks notebook source
# load the delta tab

# get original dataset stats - DONE
    # start date, end date
    # shape
    # no of customers,no of interactions, no of transactions
    # revenue

# filter

# get 10 most correlated features

# do individual feature stats

# multicol

# link to target?

# COMMAND ----------

dataset_name = "rees46"
data_path = f"dbfs:/mnt/{dataset_name}/delta/"
events = spark.read.format("delta").load(data_path+"events")

# COMMAND ----------

events.show()

# COMMAND ----------

# basic stats
import pyspark.sql.functions as f
events.agg(f.min(f.col("event_time")).alias("min_time"), f.max(f.col("event_time")).alias("max_time"),
    f.countDistinct(f.col("user_id")).alias("customers"),
    f.count("user_id").alias("interactions"),
    f.sum("view").alias("views"),
    f.sum("cart").alias("carts"),           
    f.sum("purchase").alias("purchases"),
    f.sum(f.when(f.col("purchase")==1, f.col("price")).otherwise(0)).alias("revenue")).show()

# add simulated profit

# COMMAND ----------

test = spark.table("churndb.rees46_customer_model")#.where(f.col("week_step")==4)#.toPandas()

# COMMAND ----------

_impute_customer_model(test).toPandas().isnull().sum().sort_values()

# COMMAND ----------



# COMMAND ----------



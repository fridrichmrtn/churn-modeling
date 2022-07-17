# Databricks notebook source
# CREATE TARGET DB 
#spark.sql("CREATE SCHEMA IF NOT EXISTS churndb LOCATION \"dbfs:/mnt/churndb\"")

# COMMAND ----------

# MAGIC %run "./load-transform/workhorse"

# COMMAND ----------

# MAGIC %run "./customer-model/workhorse"

# COMMAND ----------

#
##
### RETAIL ROCKET

dataset_name = "retailrocket"

# LOAD TRANSFORM
#events = construct_events(dataset_name)
#save_events(events, dataset_name)

# CUSTOMER MODEL
customer_model = construct_customer_model(
    dataset_name, week_steps=11, week_target=4, overwrite=True)
save_customer_model(
    customer_model, dataset_name, overwrite=True)

# COMMAND ----------

#import pyspark.sql.functions as f
#customer_model = spark.table("churndb.retailrocket_customer_model")
#customer_model.where(f.col("user_id")==23076).toPandas().sort_values("week_step")

# COMMAND ----------

#
##
### REES46

#dataset_name = "rees46"
# LOAD TRANSFORM
#events = construct_events(dataset_name)
#save_events(events, dataset_name)
# CUSTOMER MODEL
#split_save_customer_model(dataset_name, week_steps=11,
#    week_target=4, overwrite=True)

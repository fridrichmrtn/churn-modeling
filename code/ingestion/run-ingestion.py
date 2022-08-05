# Databricks notebook source
# CREATE TARGET DB 
#spark.sql("CREATE SCHEMA IF NOT EXISTS churndb LOCATION \"dbfs:/mnt/churndb\"")

# COMMAND ----------

# MAGIC %run "./load-transform/workhorse"

# COMMAND ----------

# MAGIC %run "./customer-model/workhorse"

# COMMAND ----------

# #
# ##
# ### RETAIL ROCKET

# dataset_name = "retailrocket"

# # LOAD TRANSFORM
# events = construct_events(dataset_name)
# save_events(events, dataset_name)

# # CUSTOMER MODEL
# _prerun_optimize_recom(dataset_name)
# customer_model = construct_customer_model(
#      dataset_name, time_steps=4, week_target=4)
# save_customer_model(
#      customer_model, dataset_name, overwrite=True)

# COMMAND ----------


#
## REES46

dataset_name = "rees46"
 
# LOAD TRANSFORM
#events = construct_events(dataset_name)
#save_events(events, dataset_name)
 
# CUSTOMER MODEL
_prerun_optimize_recom(dataset_name)
customer_model = construct_customer_model(
   dataset_name, time_steps=7, week_target=4)
save_customer_model(
   customer_model, dataset_name, overwrite=True)

# COMMAND ----------

#spark.table("churndb.rees46_customer_model").write.format("parquet").save("dbfs:/mnt/rees46/raw/customer_model/")

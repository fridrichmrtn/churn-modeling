# Databricks notebook source
# MAGIC %run ./pipelines/workhorse

# COMMAND ----------

#glue_pipeline("rees46", range(1,9))

# COMMAND ----------

#spark.sql("DELETE FROM churndb.retailrocket_predictions WHERE pipe='dt'")
glue_pipeline("retailrocket", range(1,9), False)

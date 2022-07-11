# Databricks notebook source
# MAGIC %run ./pipelines/workhorse

# COMMAND ----------

#glue_pipeline("rees46", range(1,9))

# COMMAND ----------

spark.sql("DELETE FROM churndb.retailrocket_predictions WHERE pipe='combinet' AND week_step IN (1,2,4)")
glue_pipeline(dataset_name="retailrocket", week_range=[1,2,4], drop_predictions=False)

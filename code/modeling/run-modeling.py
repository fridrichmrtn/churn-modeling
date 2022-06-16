# Databricks notebook source
# MAGIC %run ./pipelines/workhorse

# COMMAND ----------

#glue_pipeline("rees46", range(1,11))

# COMMAND ----------

glue_pipeline("retailrocket", range(1,11))

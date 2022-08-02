# Databricks notebook source
# MAGIC %run "./pipelines/workhorse"

# COMMAND ----------

glue_pipeline(dataset_name="retailrocket", time_range=range(3), overwrite=True)

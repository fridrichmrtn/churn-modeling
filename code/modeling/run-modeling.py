# Databricks notebook source
# MAGIC %run "./pipelines/workhorse"

# COMMAND ----------

glue_pipeline(dataset_name="retailrocket", week_range=range(9), overwrite=True)

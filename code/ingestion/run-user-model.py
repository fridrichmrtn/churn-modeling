# Databricks notebook source
# MAGIC %run "./model/utils"

# COMMAND ----------

import pandas as pd
import pyspark.sql.functions as f

# data load
dataInPath = "dbfs:/mnt/rees46/delta/events"
split_date = pd.to_datetime("2020-03-01")

# data split
events = spark.read.format("delta").load(dataInPath).sample(fraction=.01)
user_model = get_user_model(events, split_date)

# COMMAND ----------

# TRY TO REGISTER DATASETS AS OUTPUT
# HOW TO RUN ON MULTIPLE PARAMS
# PARAMS - LOAD LOC, SPLIT, TARGET LOC (TABLE?)
# RUN ON FULL REES46 DATA

# Databricks notebook source
# MAGIC %run "./data/workhorse"

# COMMAND ----------



#
##
### REES46

data_path = "dbfs:/mnt/rees46/"
events = construct_events(data_path+"raw", "rees46")
save_events(events, data_path+"concatenated/test_run_events")

# Databricks notebook source
# MAGIC %run "./load-transform/workhorse"

# COMMAND ----------

#
##
### REES46

data_path = "dbfs:/mnt/rees46/raw/"
events = construct_events(data_path, "rees46")
save_events(events, data_path+"concatenated/events/")

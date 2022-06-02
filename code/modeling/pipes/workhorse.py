# Databricks notebook source
# MAGIC %run ./preprocessing

# COMMAND ----------

# MAGIC %run ./modeling

# COMMAND ----------

#
##
### PIPELINES AND SPACES

pipelines_spaces = {k:{"pipeline":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
    "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"])}
     for k, v in models.items()}

def get_pipeline(name):
    return pipelines_spaces[name]["pipeline"]
def get_space(name):
    return pipelines_spaces[name]["space"]

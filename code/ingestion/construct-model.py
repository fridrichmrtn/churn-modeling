# Databricks notebook source


# COMMAND ----------

# load util notebooks
ntbs_to_load = [
  #("./pref-model"),
  #("./base-model"),
  ("./utils")]

for n in ntbs_to_load:
    dbutils.notebook.run(n, 15)
    print('Finished loading notebook ' + n)

# COMMAND ----------

import pyspark.sql.functions as f

# data load
dataInPath = "dbfs:/mnt/rees46/delta/events"
split_date = f.to_date("2020-03-01")

# data split
events = spark.read.format("delta").load(dataInPath).sample(fraction=.01)
user_target = get_target(events, split_date)
user_events = get_feature_events(events, split_date)

# COMMAND ----------

# BASE MODEL
user_base= get_base_features(user_events)

# COMMAND ----------

# PREFERENCE MODELS
import mlflow
transaction_recommendation_model = optimize_recom(
    user_events.where(f.col("event_type_name")=="purchase"),"transactions", 42)
#transaction_recommendation_model =  mlflow.spark.load_model("models:/refit_recom_transactions/None").stages[0]
user_transaction_preference = get_user_factors(transaction_recommendation_model)

view_recommendation_model = optimize_recom(
    user_events.where(f.col("event_type_name")=="view"), "views", 42)
#transaction_recommendation_model =  mlflow.spark.load_model("models:/refit_recom_transactions/None").stages[0]
user_nontransaction_preference = get_user_factors(view_recommendation_model)

# COMMAND ----------

# CHECK SIZE

# PUT IT TOGETHER

# CONSIDER PARAMS

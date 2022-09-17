# Databricks notebook source
# MAGIC %run "./pipelines/workhorse"

# COMMAND ----------

# dataset_name = "retailrocket"
# #spark.sql(f"DELETE FROM churndb.{dataset_name}_predictions WHERE pipe LIKE 'mlp%'")
# glue_pipeline(dataset_name=dataset_name,
#     time_range=range(3), overwrite=True)
# save_evaluation(dataset_name=dataset_name,
#     evaluation=evaluate_predictions(dataset_name=dataset_name))

# COMMAND ----------

dataset_name = "rees46"
#spark.sql(f"DELETE FROM churndb.{dataset_name}_predictions WHERE pipe LIKE 'mlp%'")
glue_pipeline(dataset_name=dataset_name,
     time_range=range(4), overwrite=False)
save_evaluation(dataset_name=dataset_name,
  evaluation=evaluate_predictions(dataset_name=dataset_name))

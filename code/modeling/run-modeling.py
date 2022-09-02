# Databricks notebook source
# MAGIC %run "./pipelines/workhorse"

# COMMAND ----------

#dataset_name = "retailrocket"
#spark.sql(f"DELETE FROM churndb.{dataset_name}_predictions WHERE pipe LIKE 'gbm%'")
# glue_pipeline(dataset_name=dataset_name,
#     time_range=range(3), overwrite=True)
# save_evaluation(dataset_name=dataset_name,
#     evaluation=evaluate_predictions(dataset_name=dataset_name))

# COMMAND ----------

dataset_name = "rees46"
# glue_pipeline(dataset_name=dataset_name,
#     time_range=range(4), overwrite=False)
save_evaluation(dataset_name=dataset_name,
  evaluation=evaluate_predictions(dataset_name=dataset_name))

# NOTE:
    # TRAINING AND PREDICTION IS DONE FOR ALL MODELS EXCEPT MLP
    # COPY PREDICTIONS TO NEW TABLE - DONE
    # TRY TO JUST ADD MLP PREDICTIONS - DONE
    # SAVE EVALUATION ON BEEFY MACHINE

# COMMAND ----------

#spark.sql("SELECT * FROM churndb.rees46_predictions").write.mode("overwrite").saveAsTable("churndb.rees46_predictions_backup")
#spark.sql("SELECT * FROM churndb.rees46_predictions WHERE pipe LIKE 'mlp%'").show()
#spark.sql("DELETE FROM churndb.rees46_predictions WHERE pipe LIKE 'mlp%'")

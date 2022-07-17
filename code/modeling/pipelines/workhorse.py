# Databricks notebook source
# MAGIC %run "./utils"

# COMMAND ----------

# MAGIC %run "./steps/step-spaces"

# COMMAND ----------

# MAGIC %run "./hyperopt"

# COMMAND ----------

# MAGIC %run "./evaluation"

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import pyspark.sql.functions as f
from sklearn.base import clone
from sklearn.model_selection import train_test_split

# COMMAND ----------

def _get_cols(train):
    class_y = ["target_event"]
    reg_y = ["target_actual_profit"]
    target_cols = [c for c in train.columns if "target_" in c]
    helper_cols = ["user_id", "row_id", "week_step"] # NOTE: week_step?
    cust_val_cols = [c for c in train.columns if "customer_value" in c]
    class_set = set(target_cols+helper_cols+cust_val_cols)
    reg_set = set(target_cols+helper_cols)
    class_X = [c for c in train.columns if c not in class_set]
    reg_X = [c for c in train.columns if c not in reg_set]
    return {"classification":{"X":class_X,"y":class_y},
        "regression":{"X":reg_X,"y":reg_y}}

def _get_dataset(dataset_name, week_step):
    data = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>=week_step).toPandas()
    train = data[data.week_step>week_step]
    test = data[data.week_step==week_step]			
    columns = _get_cols(train)
    return {
      "train":
            {"raw":optimize_numeric_dtypes(train),
             "week_step":week_step,
             "columns":columns,
             "name":f"{dataset_name}_{week_step}"},
      "test":
            {"raw":optimize_numeric_dtypes(test),
             "week_step":week_step,
             "columns":columns,
             "name":f"{dataset_name}_{week_step}"}}      
    
def _fit_calibrated_pipeline(data, pipe):
    X, y = get_Xy(data, pipe)    
    exp_name = "{}_{}_refit".format(data["name"],pipe["name"])
    exp_id = get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        pipe["fitted"] = pipe["calibration"](pipe["steps"]).fit(X, y)
        mlflow.sklearn.log_model(pipe["fitted"],
             exp_name, registered_model_name=exp_name)
    return pipe

def _get_predictions(data, pipe):
    predictions = []
    for temp_type, temp_data in data.items():
        X, y = get_Xy(temp_data, pipe)    
        predictions.append(pd.DataFrame.from_dict({
            "pipe":pipe["name"],
            "task":pipe["task"],
            "set_type":temp_type,
            "week_step":temp_data["week_step"],
            "user_id":temp_data["raw"]["user_id"],
            "row_id":temp_data["raw"]["row_id"],
            "predictions": pipe["fitted"].predict(X)}))
    return optimize_numeric_dtypes(pd.concat(predictions))

def _save_predictions(dataset_name, predictions):
    spark.createDataFrame(predictions)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_predictions")
    return None
    
def glue_pipeline(dataset_name, week_range, overwrite=True):
    if overwrite:
        spark.sql(f"DROP TABLE IF EXISTS churndb.{dataset_name}_predictions;")
        spark.sql(f"DROP TABLE IF EXISTS churndb.{dataset_name}_evaluation;")
    pipelines = construct_pipelines()
    for week_step in week_range:
        data = _get_dataset(dataset_name, week_step)
        for pipe_name, pipe in pipelines.items():
            pipe = optimize_pipeline(data["train"], pipe)
            pipe = _fit_calibrated_pipeline(data["train"], pipe)
            _save_predictions(dataset_name, _get_predictions(data, pipe))
    #save_evaluation(dataset_name, evaluate_pipeline(dataset_name))
    return None

# COMMAND ----------



# Databricks notebook source
# MAGIC %run "./utils"

# COMMAND ----------

# MAGIC %run "./steps/step-spaces"

# COMMAND ----------

# MAGIC %run "./hyperopt"

# COMMAND ----------

# MAGIC %run "./evaluation"

# COMMAND ----------

import os
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
    helper_cols = ["user_id", "row_id", "time_step"]
    cust_val_cols = [c for c in train.columns if "customer_value" in c ]
    class_set = set(target_cols+helper_cols)
    reg_set = set(target_cols+helper_cols)
    class_X = [c for c in train.columns if c not in class_set]
    reg_X = [c for c in train.columns if c not in reg_set]
    return {"classification":{"X":class_X,"y":class_y},
        "regression":{"X":reg_X,"y":reg_y}}

def _get_dataset(dataset_name, time_step):
    data = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("time_step")>=time_step).toPandas()
    train = data[data.time_step>time_step]
    test = data[data.time_step==time_step]			
    columns = _get_cols(train)
    return {
      "train":
            {"raw":optimize_numeric_dtypes(train),
             "time_step":time_step,
             "columns":columns,
             "name":dataset_name},
      "test":
            {"raw":optimize_numeric_dtypes(test),
             "time_step":time_step,
             "columns":columns,
             "name":dataset_name}}      
    
def _fit_calibrated_pipeline(data, pipe):
    exp_name = "/{}/modeling/refit/{}_{}".format(data["name"],pipe["name"],data["time_step"])
    exp_id = get_exp_id(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        X, y = get_Xy(data, pipe)
        pipe["fitted"] = pipe["calibration"](base_estimator=pipe["steps"]).fit(X, y)
        mlflow.sklearn.log_model(pipe["fitted"],
             os.path.relpath(exp_name, "/"),
             registered_model_name="{}_{}_{}".format(data["name"],pipe["name"],data["time_step"]))
    return pipe

def _get_predictions(data, pipe):
    predictions = []
    for temp_type, temp_data in data.items():
        X, y = get_Xy(temp_data, pipe)    
        predictions.append(pd.DataFrame.from_dict({
            "pipe":pipe["name"],
            "task":pipe["task"],
            "set_type":temp_type,
            "time_step":temp_data["time_step"],
            "user_id":temp_data["raw"]["user_id"],
            "row_id":temp_data["raw"]["row_id"],
            "predictions": pipe["fitted"].predict(X).reshape(-1)}))
    return optimize_numeric_dtypes(pd.concat(predictions))

def _save_predictions(dataset_name, predictions):
    spark.createDataFrame(predictions)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_predictions")
    return None
    
def glue_pipeline(dataset_name, time_range, overwrite=True):
    if overwrite:
        spark.sql(f"DROP TABLE IF EXISTS churndb.{dataset_name}_predictions;")
        
    pipelines = construct_pipelines()
    for time_step in time_range:
        data = _get_dataset(dataset_name, time_step)
        for pipe_name, pipe in pipelines.items():
            pipe = optimize_pipeline(data["train"], pipe, force=True)
            pipe = _fit_calibrated_pipeline(data["train"], pipe)
            _save_predictions(dataset_name, _get_predictions(data, pipe))
    return None

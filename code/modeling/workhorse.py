# Databricks notebook source
# MAGIC %run ./hyperopt

# COMMAND ----------

def _get_data(dataset_name, week_step):
    # NOTE: customer-model/workhorse.py/split_save_customer_model
    # NOTE: refactor this for own model/profit measures!
    import pyspark.sql.functions as f
    train = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>week_step).toPandas()
    test = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")==week_step).toPandas()
    out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
    feat_cols = [c for c in train.columns if c not in set(out_cols)]
    return {"train":{"X":train.loc[:,feat_cols],"y":train["target_event"]},
        "test":{"X":test.loc[:,feat_cols],"y":test["target_event"]},
        "name":f"{dataset_name}_{week_step}"}

def glue_pipeline(pipe, space, data, refit=True):
    import mlflow
    from sklearn.calibration import CalibratedClassifierCV
    
    data_pipe_name = data["name"]+"_"+pipe_name
    if refit:
        # run mlflow from here
        exp_name = f"{data_pipe_name}_hyperopt"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:    
            optimized_pipeline = _optimize_pipeline(
                data["train"]["X"], data["train"]["y"],
                    pipe, space)
        exp_name = f"{data_pipe_name}_model"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:
            calibrated_pipeline = CalibratedClassifierCV(optimized_pipeline,
                method="isotonic").fit( 
                    data["train"]["X"], data["train"]["y"])
            mlflow.sklearn.log_model(calibrated_pipeline,
                exp_name, registered_model_name=exp_name)
    else:
        exp_name = f"models:/{data_pipe_name}_model/None"
        calibrated_pipeline = mlflow.sklearn.load_model(exp_name)
        
    # evaluate & push into the delta
    results = pd.DataFrame([dict(_get_performance(
        calibrated_pipeline, v["X"], v["y"]),
            **{"type":k,"week_step":week_step,"pipe":pipe_name})
                for k,v in data.items()])
    spark.createDataFrame(results)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance_evaluation")
    return results

# COMMAND ----------

# NOTE: broadcast this across the workers?
for week_step in range(2,4):
    for pipe_name in ["lr","dt"]:
        data = _get_data("rees46", week_step)
        pipe = get_pipe(pipe_name)
        space = get_space(pipe_name)
        glue_pipeline(pipe, space, data, True)

# COMMAND ----------

# import pandas as pd
# from itertools import product
# import pyspark.sql.functions as f
# from pyspark.sql.types import *

# schema = StructType([
#     StructField("accuracy_score", FloatType(), True),
#     StructField("precision_score", FloatType(), True),
#     StructField("recall_score", FloatType(), True),
#     StructField("f1_score", FloatType(), True),
#     StructField("roc_auc_score", FloatType(), True),
#     StructField("type", StringType(), True),
#     StructField("week_step", IntegerType(), True),
#     StructField("pipe", StringType(), True)])  


# pipe_steps = pd.DataFrame(product(["lr", "dt"],range(2,4)),
#     columns=["pipe_name","week_step"])

# @f.pandas_udf(schema, f.PandasUDFType.GROUPED_MAP)
# def udf_glue_pipeline(row):
#     pipe_name = row["pipe_name"]
#     week_step = row["week_step"]
#     pipe = get_pipe(pipe_name)
#     space = get_space(pipe_name)
#     data = _get_data("rees46", week_step)
#     return glue_pipeline(pipe, space, data, True)
# results = pipe_steps.groupby("week_step").apply(udf_glue_pipeline)

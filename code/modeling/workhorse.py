# Databricks notebook source
# MAGIC %run ./hyperopt

# COMMAND ----------

def _optimize_numeric_dtypes(df):
    import pandas as pd
    float_cols = df.select_dtypes("float").columns
    int_cols = df.select_dtypes("integer").columns
    df[float_cols] = df[float_cols].\
        apply(pd.to_numeric, downcast="float")
    df[int_cols] = df[int_cols].\
        apply(pd.to_numeric, downcast="integer")
    return df

# data class?
def _get_data(dataset_name, week_step):
    # NOTE: customer-model/workhorse.py/split_save_customer_model
    # NOTE: refactor this for own model/profit measures
    import pyspark.sql.functions as f
    train = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>week_step).toPandas()
    test = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")==week_step).toPandas()
    out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
    feat_cols = [c for c in train.columns if c not in set(out_cols)]
    return {
      "train":
            {"X":_optimize_numeric_dtypes(train.loc[:,feat_cols]),
             "y":train["target_event"],
             "week_step":week_step,
             "name":f"{dataset_name}_{week_step}"},
      "test":
            {"X":_optimize_numeric_dtypes(test.loc[:,feat_cols]),
             "y":test["target_event"],
             "week_step":week_step,
             "name":f"{dataset_name}_{week_step}"}}
    
def _fit_calibrated_pipeline(data, pipe):
    import mlflow
    from sklearn.calibration import CalibratedClassifierCV
    
    exp_name = "{}_{}_refit".format(data["name"],pipe["name"])
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        pipe["fitted"] = CalibratedClassifierCV(
            pipe["steps"], cv=5, method="isotonic").fit(data["X"], data["y"])
        mlflow.sklearn.log_model(pipe["fitted"],
             exp_name, registered_model_name=exp_name)
    return pipe    

# SEPARATE PIPELINE FITTING AND EVALUATION
def _evaluate_pipeline(data, pipe):
    # evaluate & push into the delta
    dataset_name = data["train"]["name"].split("_")[0]
    pipe["results"] = pd.DataFrame([dict(_get_performance(
        pipe["fitted"], v["X"], v["y"]),
            **{"type":k,"week_step":v["week_step"],"pipe":pipe["name"]})
                for k,v in data.items()]) # NOT NICE
    spark.createDataFrame(pipe["results"])\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance")
    return pipe

def glue_pipeline(dataset_name, week_range):
    for week_step in week_range:
        data = _get_data(dataset_name, week_step)
        for pipe_name, pipe in pipelines.items():
            pipe = _optimize_pipeline(data["train"], pipe)
            pipe = _fit_calibrated_pipeline(data["train"], pipe)
            pipe = _evaluate_pipeline(data, pipe)
    return None

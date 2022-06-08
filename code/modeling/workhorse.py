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

def _get_data(dataset_name, week_step):
    # NOTE: customer-model/workhorse.py/split_save_customer_model
    # NOTE: refactor this for own model/profit measures!
    import pyspark.sql.functions as f
    train = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>week_step).toPandas()
    test = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")==week_step).toPandas()
    train =  _optimize_numeric_dtypes(train)
    test = _optimize_numeric_dtypes(test)
    out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
    feat_cols = [c for c in train.columns if c not in set(out_cols)]
    return {"train":{"X":_optimize_numeric_dtypes(train.loc[:,feat_cols]), "y":train["target_event"]},
        "test":{"X":_optimize_numeric_dtypes(test.loc[:,feat_cols]),"y":test["target_event"]},
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

# NOTE: streamline this, concat steps and spaces
import datetime

pipe_name="lr"
data = _get_data("rees46", 2)
pipe = get_pipe(pipe_name)
space = get_space(pipe_name)
st = datetime.datetime.now()
optimized_pipeline = _optimize_pipeline(data["train"]["X"], data["train"]["y"], pipe, space)
datetime.datetime.now()-stb

# COMMAND ----------

# NOTE: broadcast this across the workers?
for week_step in range(2,4):
    for pipe_name in ["lr","dt"]:
        data = _get_data("rees46", week_step)
        pipe = get_pipe(pipe_name)
        space = get_space(pipe_name)
        glue_pipeline(pipe, space, data, True)

# COMMAND ----------

# RUNTIME ESTIMATES/ fast algos
# OPT - 2 mins a fit, 10 fits within an opt, 10 opt across weeks = 200 mins opt
# CAL - 3 mins a fit, 5 fits per cal, 10 cals across weeks = 150 mins cal
# SUBTOTAL 6 h
## ALL ALGOS >= 48h wo distributed backends

# STRATEGY 1
# paralelize on dataset slices

# STRATEGY 2
# paralelize within steps - OPT, CAL

# STRATEGY 3
# rewrite pipelines to spark pipes

# STRATEGY 4
# redesign pipeline and experiment steps

# STRATEGY 5
# compress dataset, eliminate prev steps

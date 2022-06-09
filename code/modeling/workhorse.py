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
    out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
    feat_cols = [c for c in train.columns if c not in set(out_cols)]
    return {
      "train":
            {"X":_optimize_numeric_dtypes(train.loc[:,feat_cols]),
             "y":train["target_event"]},
      "test":
            {"X":_optimize_numeric_dtypes(test.loc[:,feat_cols]),
             "y":test["target_event"]},
      "name":
            f"{dataset_name}_{week_step}"}

def glue_pipeline(pipe, data, refit=True):
    import mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    
    dataset_name = data["name"]
    week_step = dataset_name.split("_")[1]
    data_pipe_name = dataset_name+"_"+pipe["name"]
   
    if refit:
        exp_name = f"{data_pipe_name}_hyperopt"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:    
            optimized_pipeline = _optimize_pipeline(
                data["train"]["X"], data["train"]["y"],
                    pipe["steps"], pipe["space"])
        exp_name = f"{data_pipe_name}_model"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:
            X, Xc, y, yc = train_test_split(data["train"]["X"], data["train"]["y"],
                test_size=.2, stratify=data["train"]["y"])
            optimized_pipeline.fit(X, y)
            calibrated_pipeline = CalibratedClassifierCV(optimized_pipeline,
                method="isotonic", cv="prefit").fit(Xc, yc)
            mlflow.sklearn.log_model(calibrated_pipeline,
                exp_name, registered_model_name=exp_name)
    else:
        exp_name = f"models:/{data_pipe_name}_model/None"
        calibrated_pipeline = mlflow.sklearn.load_model(exp_name)
        
    # evaluate & push into the delta
    results = pd.DataFrame([dict(_get_performance(
        calibrated_pipeline, v["X"], v["y"]),
            **{"type":k,"week_step":week_step,"pipe":pipe_name})
                for k,v in data.items() if k in ["train", "test"]]) # NOT NICE
    spark.createDataFrame(results)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance_evaluation")
    return results

# COMMAND ----------

# try it with much smaller dataset
from sklearn.datasets import load_breast_cancer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import mlflow


steps = get_pipeline("dt")["steps"]
space = get_pipeline("dt")["space"]

#data = _get_data("rees46", 5)

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data["data"],data["target"], test_size=.4)

with mlflow.start_run() as run:  
    results = _optimize_pipeline(
        X_train, y_train,
            steps, space)

# COMMAND ----------


    

# COMMAND ----------

# ASYNC HYPEROPT AND FIT/CALIBRATION, CLEAN CODE
# REIMPLEMENT CALIBRATION
# TRY WET RUN ON 2 WEEKS AND CLFS

# COMMAND ----------

# NOTE: broadcast this across the workers?
for week_step in range(2,4):
    data = _get_data("rees46", week_step)
    for pipe_name in ["hgb","dt"]:
        pipe = get_pipes(pipe_name)
        glue_pipeline(pipe, data, True)

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

# COMMAND ----------



# COMMAND ----------

Scaler()

# COMMAND ----------

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline

from hyperopt import hp, tpe, fmin, SparkTrials, Trials, STATUS_OK, space_eval
from functools import partial
from imblearn.base import FunctionSampler



pipe = Pipeline([("sa", Sampler(sampling_strategy="under_sampling")),
    ("sc", Scaler(scaling_strategy="robust")),
    ("lr", LogisticRegression())])

data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data["data"],data["target"], test_size=.4)

def run_lr(params, pipe): 
    #alpha = params["alpha"]
    pipe.set_params(**params)
    pipe.fit(X_train,y_train)
    obj_metric = pipe.score(X_test, y_test)
    return {"loss": obj_metric, "status": STATUS_OK}

space = {
  "lr__C": hp.uniform("lr__C", 10**-3, 10),
  "sa__sampling_strategy": hp.choice("sa__sampling_strategy",["under_sampling","over_sampling"]),
  "sc__scaling_strategy":hp.choice("sa__scaling_strategy",["power","quantile", "robust"])
}

spark_trials =  SparkTrials(parallelism=8)
#spark_trials = Trials()
with mlflow.start_run():
    best_hyperparam = fmin(fn=partial(run_lr, pipe=pipe), 
    space=space, 
    algo=tpe.suggest, 
    max_evals=16, 
    trials=spark_trials)

# COMMAND ----------



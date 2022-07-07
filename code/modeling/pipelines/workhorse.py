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

# NOTE: CLASSED
def _get_cols(train):
    out_multi_cols = ["user_id", "target_event",
        "target_revenue", "week_step", "target_cap"]
    out_standard_cols = ["cap_month_lag0", "cap_month_lag1",
        "cap_month_lag2", "cap_month_lag3", "cap_month_ma3"]
    multi_cols = [c for c in train.columns if c not in set(out_multi_cols)]
    standard_cols = [c for c in multi_cols if c not in set(out_standard_cols)]
    return (standard_cols, multi_cols)

def _get_data(dataset_name, week_step):
    # NOTE: customer-model/workhorse.py/split_save_customer_model
    import pyspark.sql.functions as f
    
    data = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>=week_step).toPandas()
    train = data[data.week_step>week_step]
    test = data[data.week_step==week_step]			
    standard_cols, multi_cols = _get_cols(train)
    return {
      "train":
            {"raw":_optimize_numeric_dtypes(train),
             #"y":train["target_event"],
             #"cap":train.loc[:,"target_cap"],
             #"user_id":train.loc[:,"user_id"],
             "week_step":week_step,
             "columns":{"standard":standard_cols, "multi-output":multi_cols},
             "name":f"{dataset_name}_{week_step}"},
      "test":
            {"raw":_optimize_numeric_dtypes(test),
             #"y":test["target_event"],
             #"cap":test.loc[:,"target_cap"],
             #"user_id":test.loc[:,"user_id"],
             "week_step":week_step,
             "columns":{"standard":standard_cols, "multi-output":multi_cols},
             "name":f"{dataset_name}_{week_step}"}}      
    
def _fit_calibrated_pipeline(data, pipe):
    import mlflow
    from sklearn.model_selection import train_test_split
    
    X, y = _get_Xy(data, pipe)    
    exp_name = "{}_{}_refit".format(data["name"],pipe["name"])
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        #prefit_model = pipe["steps"].fit(X_train, y_train)
        pipe["fitted"] = pipe["calibration"](pipe["steps"],
            cv=3, method="sigmoid").fit(X, y)
        mlflow.sklearn.log_model(pipe["fitted"],
             exp_name, registered_model_name=exp_name)
    return pipe    

# SEPARATE PIPELINE FITTING AND EVALUATION
def _evaluate_pipeline(data, pipe):
    # evaluate & push into the delta
    X, y = _get_Xy(data, pipe) 
    dataset_name = data["train"]["name"].split("_")[0]
    pipe["results"] = pd.DataFrame([dict(_get_performance(
        pipe["fitted"], v["X"], v["y"]),
            **{"type":k,"week_step":v["week_step"],"pipe":pipe["name"]})
                for k,v in data.items()]) # NOT NICE
    spark.createDataFrame(pipe["results"])\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance")
    return pipe

def _get_predictions(data, pipe):
    #import numpy as np
    import pandas as pd
    predictions = []
    for temp_type, temp_data in data.items():
        
        X, y = _get_Xy(temp_data, pipe)
        y_pred_event = pipe["fitted"].predict(X)
        y_pred_event_proba = pipe["fitted"].predict_proba(X)[:,1]
        if pipe["type"]=="multi-output":
            y_pred_cap = pipe["fitted"].predict(X, scope="regression")
        else:
            y_pred_cap = None
            
        predictions.append(pd.DataFrame.from_dict({
            "pipe":pipe["name"],
            "type":temp_type,
            "week_step":temp_data["week_step"],
            "user_id":temp_data["raw"]["user_id"],
            "y_event":temp_data["raw"]["target_event"],
            "y_cap":temp_data["raw"]["target_cap"],
            "y_pred_event":y_pred_event,
            "y_pred_event_proba":y_pred_event_proba,
            "y_pred_cap":y_pred_cap}))
        
    return pd.concat(predictions)

def _save_predictions(dataset_name, predictions):
    spark.createDataFrame(predictions)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_predictions")
    return None
    
def glue_pipeline(dataset_name, week_range, drop_predictions=True):
    if drop_predictions:
        spark.sql(f"DROP TABLE IF EXISTS churndb.{dataset_name}_predictions;")
    for week_step in week_range:
        data = _get_data(dataset_name, week_step)
        for pipe_name, pipe in pipelines.items():
            pipe = _optimize_pipeline(data["train"], pipe)
            pipe = _fit_calibrated_pipeline(data["train"], pipe)
            _save_predictions(dataset_name, _get_predictions(data, pipe))
    return None

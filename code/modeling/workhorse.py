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
            "name":f"{dataset_name}_{week_step}"},
      "test":
            {"X":_optimize_numeric_dtypes(test.loc[:,feat_cols]),
             "y":test["target_event"],
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
# 
def glue_pipeline(pipe, data, refit=True):
    
    import mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    
    dataset_name = data["name"]
    # we can shift this into the optimize and calibrate funcs
    week_step = dataset_name.split("_")[1]
    data_pipe_name = dataset_name+"_"+pipe["name"]
   
    if refit:
        # encapsulate this entirely to the _optimize pipeline func
        # pass the dataset and pipe dicts directly unpack
        exp_name = f"{data_pipe_name}_hyperopt"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:    
            optimized_pipeline = _optimize_pipeline(
                data["train"]["X"], data["train"]["y"],
                    pipe["steps"], pipe["space"])
            
        # encapsulate this in calibrate_fit_pipeline func
        # consider different naming?
        # clean up the calibration separation?
        # probably just ignore parallelization at this point wrt compute?
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
        # 
        exp_name = f"models:/{data_pipe_name}_model/None"
        calibrated_pipeline = mlflow.sklearn.load_model(exp_name)
        
    # evaluate & push into the delta
    results = pd.DataFrame([dict(_get_performance(
        calibrated_pipeline, v["X"], v["y"]),
            **{"type":k,"week_step":week_step,"pipe":pipe_name})
                for k,v in data.items()]) # NOT NICE
    spark.createDataFrame(results)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance_evaluation")
    return results

# COMMAND ----------

for week_step in range(2, 3):
    data = _get_data("retailrocket", week_step)
    meh = ["lr", "svm_lin", "svm_rbf", "mlp", "dt", "rf", "hgb"]
    for pipe_name in meh:
        pipe = pipelines[pipe_name]
        pipe = _optimize_pipeline(data["train"], pipe)
        pipe = _fit_calibrated_pipeline(data["train"], pipe)

# COMMAND ----------

# MODELING
# ENCAPSULATE EVALUATE PIPE FROM GLUE - NEXT
# RE-RUN FULL PIPELINES ON RR DATASET - NEXT

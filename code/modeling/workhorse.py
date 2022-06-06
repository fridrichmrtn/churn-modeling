# Databricks notebook source
# MAGIC %run ./hyperopt

# COMMAND ----------

def _get_data(dataset_name, week_step):
    # NOTE: customer-model/workhorse.py/split_save_customer_model
    # NOTE: refactor this for own model/profit measures!
    import pyspark.sql.functions as f
    train = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")==week_step).toPandas()
    test = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")==(week_step-1)).toPandas()
    out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
    feat_cols = [c for c in train.columns if c not in set(out_cols)]
    return {"train":{"X":train.loc[:,feat_cols],"y":train["target_event"]},
        "test":{"X":test.loc[:,feat_cols],"y":test["target_event"]}}

def glue_pipeline(pipe_name, dataset_name, week_step, refit=True):
    # NOTE: move model names out of the conditions
    # NOTE: factor out the hyperopt and fitting for async
    
    import mlflow
    from sklearn.calibration import CalibratedClassifierCV
        
    data = _get_data(dataset_name, week_step)
    pipe = get_pipe(pipe_name)
    space = get_space(pipe_name)
    if refit:
        # run mlflow from here
        exp_name = f"{dataset_name}_{week_step}_{pipe_name}_hyperopt"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:    
            optimized_pipeline = _optimize_pipeline(
                data["train"]["X"], data["train"]["y"],
                    pipe, space)
        exp_name = f"{dataset_name}_{week_step}_{pipe_name}_model"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        # NOTE: consider removing n_jobs
        with mlflow.start_run() as run:
            calibrated_pipeline = CalibratedClassifierCV(optimized_pipeline,
                method="isotonic", n_jobs=5).fit( 
                    data["train"]["X"], data["train"]["y"])
            # log model
            mlflow.sklearn.log_model(calibrated_pipeline,
                exp_name, registered_model_name=exp_name)
    else:
        exp_name = f"models:/{dataset_name}_{week_step}_{pipe_name}_model/None"
        calibrated_pipeline = mlflow.sklearn.load_model(exp_name)
        
    # evaluate & push into the delta
    results = pd.DataFrame([dict(_get_performance(
        calibrated_pipeline, v["X"], v["y"]),
            **{"type":k,"week_step":week_step,"pipe":pipe_name})
                for k,v in data.items()])
    spark.createDataFrame(results)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance_evaluation")
    return None

# COMMAND ----------

# NOTE: broadcast this across the workers?
# NOTE: lol, factor out the data extraction, you moron
for pipe_name in ["lr","dt"]:
    for week_step in range(2,12):
        glue_pipeline(pipe_name, "rees46", week_step, True)

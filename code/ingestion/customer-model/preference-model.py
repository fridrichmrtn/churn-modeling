# Databricks notebook source
import os
from hyperopt import hp
preference_config = {
    "event_types":["view","purchase"],
    "space":{"rank": hp.randint("rank", 5, 50),
             "maxIter": hp.randint("maxIter", 5, 100),
             "regParam": hp.loguniform("regParam", -5, 3),
             "alpha": hp.uniform("alpha", 25, 350)},
    "max_evals":25,
    "seed":42}

# COMMAND ----------

#
##
### HELPERS

def _get_exp_id(exp_path):
    import mlflow
    try:
        exp_id = mlflow.get_experiment_by_name(exp_path).experiment_id
    except:
        exp_id = mlflow.create_experiment(exp_path)
    return exp_id

def _get_imp_feed(events):
    import pyspark.sql.functions as f
    # form the dataset
    return (events.groupBy("user_id", "product_id")
        .agg(f.count(f.col("user_session_id")).alias("implicit_feedback")))

#
##
### HYPEROPT SEARCH 

def _set_recom(params, seed):
    from pyspark.ml.recommendation import ALS
    
    # unpack params
    rank = int(params["rank"]) # int
    maxiter = int(params["maxIter"]) # int
    regparam = float(params["regParam"]) # float
    alpha = float(params["alpha"]) # float    
    # init
    als = ALS(userCol="user_id", itemCol="product_id", ratingCol="implicit_feedback",
        implicitPrefs=True, coldStartStrategy="drop", nonnegative=True,
        seed=seed, rank=rank, maxIter=maxiter, alpha=alpha, regParam=regparam)
    return als

def _eval_recom(params, implicit_feedback, seed):
    import mlflow
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
    from hyperopt import STATUS_OK
    
    # factor split ratio out?
    train_data, val_data = implicit_feedback.randomSplit([.6,.4])
    
    # setup
    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(params)
        model = _set_recom(params, seed).fit(train_data)
        train_pred = model.transform(train_data)
        val_pred = model.transform(val_data)
        metrics_dict = {}
        for metric in ["rmse", "r2", "var"]:
            evaluator = RegressionEvaluator(metricName=metric,
                labelCol="implicit_feedback", predictionCol="prediction")
            metrics_dict["train_"+metric] = evaluator.evaluate(train_pred)
            metrics_dict["val_"+metric] = evaluator.evaluate(val_pred)
        mlflow.log_metrics(metrics_dict)   
    return {"loss":metrics_dict["val_rmse"], "status":STATUS_OK}

def _optimize_recom(events, dataset_name, event_type, seed):
    from hyperopt import fmin, tpe
    from functools import partial
    import mlflow
    
    # optimization
    exp_name = f"/{dataset_name}/ingestion/hyperopt/{event_type}_reco"
    exp_id = _get_exp_id(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:    
        optimized_params = fmin(
            fn=partial(_eval_recom,
                implicit_feedback=_get_imp_feed(events), seed=seed),
            max_evals=preference_config["max_evals"],
            space=preference_config["space"], algo=tpe.suggest)
        return optimized_params

#
##
### REFIT

def _get_hyperopt_run(dataset_name, event_type):
    import mlflow
    run_id = mlflow.search_runs(experiment_ids=_get_exp_id(
        f"/{dataset_name}/ingestion/hyperopt/{event_type}_reco"),
            search_all_experiments=True, order_by=["metrics.val_rmse ASC"],
                max_results=1)["run_id"][0]
    return run_id

def _fit_recom(events, event_type, dataset_name, run_id=None):
    import mlflow
    
    if run_id is None:
        run_id = _get_hyperopt_run(dataset_name, event_type)        
    run_params = mlflow.get_run(run_id=run_id).data.params
    
    exp_name = f"/{dataset_name}/ingestion/refit/{event_type}_reco"
    exp_id = _get_exp_id(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:     
        mlflow.log_params(run_params)
        imp_feed = _get_imp_feed(events)
        model = _set_recom(run_params, seed=preference_config["seed"]).fit(imp_feed)
        mlflow.spark.log_model(model,
            os.path.relpath(exp_name, "/"),
            registered_model_name="{}_{}_reco".format(dataset_name,event_type))
        return model
      
#
##
### PREFERENCE FROM MODEL

def _get_user_factors(recom_model, event_type):
    import pyspark.sql.functions as f
    
    dims = range(recom_model.rank)
    user_factors = (recom_model.userFactors
        .select(f.col("id").alias("user_id"),
            *[f.col("features").getItem(d).alias(f"{event_type}_latent_factor{d}")
                for d in dims]))
    return user_factors

#
##
### PRE-RUN THE OPTIMIZATION ON FULL DATA

def _prerun_optimize_recom(dataset_name):
    import pyspark.sql.functions as f
    
    data_path = f"dbfs:/mnt/{dataset_name}/delta/events"
    events = spark.read.format("delta").load(data_path)
    for event_type in  preference_config["event_types"]:
        _ = _optimize_recom(
            events.where(f.col("event_type_name")==event_type),
            dataset_name, event_type, preference_config["seed"])
    
#
##
### GET PREFERENCE MODEL

def get_pref_model(events, dataset_name):
    import pyspark.sql.functions as f
    import mlflow
    
    dfs = []
    for event_type in  preference_config["event_types"]:
        recom = _fit_recom(events.where(f.col("event_type_name")==event_type),
            event_type, dataset_name)
        dfs += [_get_user_factors(recom, event_type)]
    return dfs[0].join(dfs[1], on=["user_id"]) 

#
##
### PLOTTING
# NOTE: consider scaling metrics to same axis

def _plot_hyperopt(parent_run_id, labels):
    import re
    import plotly.express as px

    search_data = mlflow.search_runs(filter_string=f"tags.mlflow.rootRunId=\"{parent_run_id}\"",
        search_all_experiments=True).dropna()
    search_data.columns  = [re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower()\
        if "params" in c else c for c in search_data.columns]
    search_data = search_data[list(labels.keys())]
    for c in search_data.columns: # assuming all numeric
        search_data[c] = pd.to_numeric(search_data[c])
        
    color = [k for k in labels.keys() if "metrics.val" in k][0]
    fig = px.parallel_coordinates(
        search_data, color = color,
        dimensions=list(labels.keys()), labels = labels,
        color_continuous_scale=px.colors.sequential.Jet,
        color_continuous_midpoint=None)
    fig.show()    

# COMMAND ----------

#
##
### test runs

# test run
#dataset_name = "rees46"
#data_path = f"dbfs:/mnt/{dataset_name}/delta/"
#events = spark.read.format("delta").load(data_path+"events").sample(fraction=.001)
#test = get_pref_features(events, "rees46", refit=False)

# test run
#_prerun_optimize_recom("rees46")

# test run
#from collections import OrderedDict
#parent_run_id = "016c545a89474ecf9747d6f06dc2affd"
#labels = OrderedDict([("params.alpha","alpha"), ("params.rank","rank"),
#    ("params.max_iter","iterations"), ("params.reg_param","regularization strength"),
#    ("metrics.train_rmse","training RMSE"), ("metrics.val_rmse","validation RMSE")])

#_plot_hyperopt(parent_run_id, labels)

# COMMAND ----------

# run this before cust model build
#_prerun_optimize_recom("rees46")

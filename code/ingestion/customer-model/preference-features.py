# Databricks notebook source
### NOTE: ADD DOCSTRINGS, FACTOR OUT HYPEROPT SPACE AND ITERNO?

#
##
### SETUP
from hyperopt import hp

setup_dict = {
    "rees46":{
        
        # optimization conf
        "hyperopt":{"space":{"rank": hp.randint("rank", 5, 50),
                             "maxIter": hp.randint("maxIter", 5, 100),
                             "regParam": hp.loguniform("regParam", -5, 3),
                             "alpha": hp.uniform("alpha", 25, 350)},
                    
                    "max_evals":50,
            
                     # optimized runs
                    "runs":{"purchase":"d7711ccc392e4c01844ffeca197660e0",
                        "view":"db0941699fd94fb08b1c23d3350a3a2c"},
                    # TBD
                    "models":{"purchase":"a4b7daa633574c518aadb33855d61c1f",
                         "view":"a1c9a2b68f0d4c848258d5c5a2c5ee60"}},
        
        # utils
        #"reco_types":["purchase", "view"],
        "seed":42,}}

# COMMAND ----------

#
##
### PREFERENCE

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
    implicit_feedback = (events.groupBy("user_id", "product_id")
        .agg(f.count(f.col("user_session_id")).alias("implicit_feedback")))
    return implicit_feedback

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
    
    # setup
    space = setup_dict[dataset_name]["hyperopt"]["space"]
    max_evals = setup_dict[dataset_name]["hyperopt"]["max_evals"]
    implicit_feedback = _get_imp_feed(events)
    
    # optimization
    exp_name = f"{dataset_name}_optimize_recom_{event_type}"
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:    
        optimized_params = fmin(
            fn=partial(_eval_recom,
                implicit_feedback=implicit_feedback, seed=seed),
            max_evals=max_evals, space=space, algo=tpe.suggest)
        return optimized_params

#
##
### REFIT
    
def _fit_optimized_recom(events, event_type, dataset_name, run_id=None):
    import mlflow
    
    if run_id is None:
        run_id = setup_dict[dataset_name]["hyperopt"]["runs"][event_type]
            
    seed = setup_dict[dataset_name]["seed"]
    run_params = mlflow.get_run(run_id=run_id).data.params   
    exp_name = f"{dataset_name}_refit_recom_{event_type}"
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:     
        mlflow.log_params(run_params)
        imp_feed = _get_imp_feed(events)
        model = _set_recom(run_params, seed=seed).fit(imp_feed)
        mlflow.spark.log_model(model, exp_name, registered_model_name=exp_name)
        return model
      
#
##
### PREFERENCE FROM MODEL

def _get_user_factors(recom_model, event_type):
    import pyspark.sql.functions as f
    
    dims = range(recom_model.rank)
    user_factors = (recom_model.userFactors
        .select(f.col("id").alias("user_id"),
            *[f.col("features").getItem(d).alias(event_type+"_latent_factor"+str(d))
                for d in dims]))
    return user_factors

#
##
### GET PREFERENCE MODEL

def get_pref_features(events, dataset_name, refit=True):
    import pyspark.sql.functions as f
    import mlflow
    
    dfs = []
    if refit:
        for k, v in  setup_dict[dataset_name]["hyperopt"]["runs"].items():
            recom = _fit_optimized_recom(events.where(f.col("event_type_name")==k),
                k, dataset_name, v)
            dfs += [_get_user_factors(recom, k)]
            
    else:
        for k, v in  setup_dict[dataset_name]["hyperopt"]["models"].items():
            recom = mlflow.spark.load_model(f"models:/{dataset_name}_refit_recom_{k}/None")\
                .stages[0]
            dfs += [_get_user_factors(recom, k)]
    
    # crude solution for now
    return dfs[0].join(dfs[1], on=["user_id"])

#
##
### PRE-RUN THE OPTIMIZATION

def _prerun_optimize_recom(dataset_name):
    import pyspark.sql.functions as f
    
    seed = setup_dict[dataset_name]["seed"]
    
    events = spark.read.format("delta").load(f"dbfs:/mnt/{dataset_name}/delta/events")#.sample(fraction=.001)
    transactions = events.where(f.col("event_type_name")=="purchase")
    _ = _optimize_recom(transactions, dataset_name, "transactions", seed)
    views = events.where(f.col("event_type_name")=="view")
    _ = _optimize_recom(views, dataset_name, "views", seed)

# COMMAND ----------


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

# test run
#from collections import OrderedDict
#parent_run_id = "016c545a89474ecf9747d6f06dc2affd"
#labels = OrderedDict([("params.alpha","alpha"), ("params.rank","rank"),
#    ("params.max_iter","iterations"), ("params.reg_param","regularization strength"),
#    ("metrics.train_rmse","training RMSE"), ("metrics.val_rmse","validation RMSE")])

#_plot_hyperopt(parent_run_id, labels)

# COMMAND ----------

# test run
#dataset_name = "rees46"
#data_path = f"dbfs:/mnt/{dataset_name}/delta/"
#events = spark.read.format("delta").load(data_path+"events").sample(fraction=.001)
#test = get_pref_features(events, "rees46", refit=False)

# test run
_prerun_optimize_recom("rees46")

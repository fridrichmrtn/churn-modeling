# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

import numpy as np
from functools import partial
import mlflow
from hyperopt import SparkTrials, tpe, fmin, space_eval, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from scipy.stats import hmean
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, r2_score,
    mean_absolute_error, mean_squared_error)
    
hyperopt_config = {
    "max_evals":25,
    "trials":SparkTrials,
    "algo":tpe.suggest,
    "seed":20220602}

# COMMAND ----------

#
##
### PIPELINE HYPEROPT

def _train_test_dict(task, X, y, test_size, seed):
    strat_y = y
    if task=="regression":
        strat_y = np.digitize(y, bins=np.percentile(y, q=[0,25,50,75]))
        #strat_y = None
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, stratify=strat_y, random_state=seed) 
    return {"train":{"X":X_train, "y":y_train},
        "test":{"X":X_test, "y":y_test}}
    
def _get_class_perf(model, X, y):
    metrics = {m.__name__:m for m in\
        [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    predicted = model.predict(X)   
    results ={}
    for n,f in metrics.items():
        if "roc_auc" in n:
            results[n] = np.nan
            if hasattr(model, "predict_proba"):
                predict_proba = model.predict_proba(X)[:,1]
                results[n]=f(y, predict_proba)
        else:
            results[n] = f(y, predicted)
    results["loss"] = results["f1_score"]
    return results

def _get_reg_perf(model, X, y):
    metrics = {m.__name__:m for m in\
            [r2_score, mean_absolute_error, mean_squared_error]}
    predicted = model.predict(X)
    results = {n:f(y, predicted) for n,f in metrics.items()}
    results["loss"] = results["r2_score"]
    return results

def _get_performance(task, model, X, y):
    f = _get_class_perf
    if task=="regression":
        f = _get_reg_perf
    return f(model, X, y)

def _evaluate_hyperopt(params, task, model, X, y, seed):
    data_dict = _train_test_dict(task, X, y, .4, seed)
    model.set_params(**params)
    model.fit(data_dict["train"]["X"], data_dict["train"]["y"])
    metrics = {n+"_"+m:v for n, data in data_dict.items()
        for m,v in _get_performance(task, model, data["X"], data["y"]).items()}
    return {"loss": -metrics["test_loss"], "status":STATUS_OK}

def optimize_pipeline(data, pipe):
    X, y = get_Xy(data, pipe)
    
    exp_name = "/{}/modeling/hyperopt/{}_{}".format(data["name"],pipe["name"],data["time_step"])
    exp_id = get_exp_id(exp_name)
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        space_optimized = fmin(
            fn=partial(_evaluate_hyperopt,
                X=X, y=y, task=pipe["task"], model=pipe["steps"],
                    seed=hyperopt_config["seed"]),
            space=pipe["space"], max_evals=hyperopt_config["max_evals"], 
            trials=hyperopt_config["trials"](parallelism=5), algo=hyperopt_config["algo"])
    pipe["steps"] =  clone(pipe["steps"]).set_params(
        **space_eval(pipe["space"], space_optimized))
    return pipe

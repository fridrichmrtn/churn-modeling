# Databricks notebook source
import numpy as np
from functools import partial
import mlflow
from hyperopt import SparkTrials, tpe, fmin, space_eval, STATUS_OK
from sklearn.model_selection import train_test_split
from scipy.stats import hmean
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, r2_score,
    mean_squared_error)
    
hyperopt_config = {
    "max_evals":1,
    "trials":SparkTrials,
    "algo":tpe.suggest,
    "seed":20220602}

# COMMAND ----------

#
##
### PIPELINE HYPEROPT

def _train_test_dict(X, y, test_size, seed):
    y_strata = y
    if len(y.shape)>1:
        y_strata = y[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, stratify=y_strata, random_state=seed) 
    return {"train":{"X":X_train, "y":y_train},
        "test":{"X":X_test, "y":y_test}}
    
def _get_class_perf(model, X, y):
    metrics = {m.__name__:m for m in\
        [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    predicted = model.predict(X)
    if len(y.shape)>1:
        y = y[:,0]    
    results ={}
    for n,f in metrics.items():
        if "roc_auc" in n:
            results[n] = np.nan
            if hasattr(model, "predict_proba"):
                predict_proba = model.predict_proba(X)[:,1]
                results[n]=f(y, predict_proba)
        else:
            results[n] = f(y, predicted)
    return results

def _get_reg_perf(model, X, y):
    metrics = {m.__name__:m for m in\
            [r2_score, mean_squared_error]}
    if len(y.shape)>1:
        y = y[:,1]
        predicted = model.predict(X, scope="regression")
        results = {n:f(y, predicted) for n,f in metrics.items()}
    else:
        results ={n:np.nan for n,f in metrics.items()}
    return results

def _get_performance(model, X, y):
    class_metrics = _get_class_perf(model, X, y)
    reg_metrics = _get_reg_perf(model, X, y)
    return dict(**class_metrics, **reg_metrics)

def _evaluate_hyperopt(params, model, X, y, seed):
    data_dict = _train_test_dict(X, y, .4, seed)
    model.set_params(**params)
    model.fit(data_dict["train"]["X"], data_dict["train"]["y"])
    metrics = {n+"_"+m:v for n, data in data_dict.items()
        for m,v in _get_performance(model, data["X"], data["y"]).items()}
    
    loss = np.maximum(metrics["test_roc_auc_score"],10**-5)
    if len(y.shape)>1:
        test_r2_score = np.maximum(metrics["test_r2_score"],10**-5)
        loss = hmean([loss, test_r2_score])
    return {"loss":-loss, "status":STATUS_OK}

def optimize_pipeline(data, pipe):
    X, y = get_Xy(data, pipe)
    exp_name = "{}_{}_hyperopt".format(data["name"],pipe["name"])
    exp_id = get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        space_optimized = fmin(
            fn=partial(_evaluate_hyperopt,
                X=X, y=y, model=pipe["steps"],\
                    seed=hyperopt_config["seed"]),
            space=pipe["space"], max_evals=hyperopt_config["max_evals"], 
            trials=hyperopt_config["trials"](parallelism=4), algo=hyperopt_config["algo"])
    pipe["steps"] =  pipe["steps"].set_params(
        **space_eval(pipe["space"], space_optimized))
    return pipe

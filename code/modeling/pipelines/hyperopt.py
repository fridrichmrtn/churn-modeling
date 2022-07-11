# Databricks notebook source
# MAGIC %run ./step-spaces

# COMMAND ----------

from hyperopt import SparkTrials, tpe
hyperopt_config = {
    "max_evals":50,
    "trials":SparkTrials,
    "algo":tpe.suggest,
    "seed":20220602}
#
##
### PIPELINE HYPEROPT

def _get_Xy(data, pipe):
    X = data["raw"].loc[:,data["columns"][pipe["type"]]]
    if pipe["type"]=="standard":
        y = data["raw"]["target_event"]
    else:
        y = data["raw"][["target_event", "target_cap"]]
    return (X.values, y.values)

def _get_exp_id(exp_path):
    import mlflow
    try:
        exp_id = mlflow.get_experiment_by_name(exp_path).experiment_id
    except:
        exp_id = mlflow.create_experiment(exp_path)
    return exp_id

def _train_test_dict(X, y, test_size, seed):
    from sklearn.model_selection import train_test_split
    
    if len(y.shape)>1:
        y_strata = y[:,0]
    else:
        y_strata = y
        
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, stratify=y_strata, random_state=seed) 
    return {"train":{"X":X_train, "y":y_train},
        "test":{"X":X_test, "y":y_test}}
    
def _get_class_perf(model, X, y):
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score)

    metrics = {m.__name__:m for m in\
        [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    predicted = model.predict(X)
    if len(y.shape)>1:
        y = y[:,0]    
    results ={}
    for n,f in metrics.items():
        if "roc_auc" in n:
            if hasattr(model, "predict_proba"):
                predicted_proba = model.predict_proba(X)[:,1]
                results[n] = f(y, predicted_proba)
            else:
                results[n] = np.nan
        else:
            results[n] = f(y, predicted)
    return results

def _get_reg_perf(model, X, y):
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error
    
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
    return dict(class_metrics, **reg_metrics)

def _evaluate_hyperopt(params, model, X, y, seed):
    from hyperopt import STATUS_OK
    from sklearn.model_selection import train_test_split
    from scipy.stats import hmean
  
    data_dict = _train_test_dict(X, y, .4, seed)
    model.set_params(**params)
    model.fit(data_dict["train"]["X"], data_dict["train"]["y"])
    metrics = {n+"_"+m:v for n, data in data_dict.items()
        for m,v in _get_performance(model, data["X"], data["y"]).items()}
    
    if len(y.shape)>1:
        test_r2_score = np.maximum(metrics["test_r2_score"],10**-6)
        loss = hmean([metrics["test_f1_score"], test_r2_score])
    else:
        loss = metrics["test_f1_score"]
    return {"loss":-loss, "status":STATUS_OK}

def _optimize_pipeline(data, pipe):
    import mlflow
    from hyperopt import fmin, space_eval
    from functools import partial
        
    # set the experiment logging
    X, y = _get_Xy(data, pipe)
    exp_name = "{}_{}_hyperopt".format(data["name"],pipe["name"])
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        space_optimized = fmin(
            fn=partial(_evaluate_hyperopt,
                X=X, y=y, model=pipe["steps"],\
                    seed=hyperopt_config["seed"]),
            space=pipe["space"], max_evals=hyperopt_config["max_evals"], 
            trials=hyperopt_config["trials"](parallelism=6), algo=hyperopt_config["algo"])
    pipe["steps"] =  pipe["steps"].set_params(
        **space_eval(pipe["space"], space_optimized))
    return pipe

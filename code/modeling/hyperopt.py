# Databricks notebook source
# MAGIC %run ./step-spaces

# COMMAND ----------

#
##
### PIPELINE HYPEROPT

hyperopt_config = {
    "max_evals":25,
    "seed":20220602}

def _get_exp_id(exp_path):
    import mlflow
    try:
        exp_id = mlflow.get_experiment_by_name(exp_path).experiment_id
    except:
        exp_id = mlflow.create_experiment(exp_path)
    return exp_id

def _train_test_dict(X, y, test_size, seed):
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=test_size, stratify=y, random_state=seed) 
    return {"train":{"X":X_train, "y":y_train},
        "test":{"X":X_test, "y":y_test}}

def _get_performance(model, X, y):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {m.__name__:m for m in\
        [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    predicted = model.predict(X)
    predicted_proba = model.predict_proba(X)[:,1]
    results ={}
    for n,f in metrics.items():
        if "roc_auc" in n:
            results[n] = f(y, predicted_proba)
        else:
            results[n] = f(y, predicted)
    return results

def _evaluate_pipeline(params, pipe, X, y, seed):
    from hyperopt import STATUS_OK
    from sklearn.model_selection import train_test_split
    import mlflow
    
    data_dict = _train_test_dict(X, y, .4, seed)
    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(params)
        pipe.set_params(**params)
        pipe.fit(data_dict["train"]["X"], data_dict["train"]["y"])
        metrics = {n+"_"+m:v for n, data in data_dict.items()
            for m,v in _get_performance(pipe, data["X"], data["y"]).items()}
        mlflow.log_metrics(metrics)
    return {"loss":-metrics["test_f1_score"], "status":STATUS_OK}


def _optimize_pipeline(X, y, pipe, space):

    max_evals = hyperopt_config["max_evals"]
    seed = hyperopt_config["seed"]
    
    import mlflow
    from hyperopt import fmin, tpe, space_eval
    from functools import partial 
    # run optimization & staff

    space_optimized = fmin(
        fn=partial(_evaluate_pipeline,
            pipe=pipe, X=X, y=y, seed=seed),
        max_evals=max_evals, space=space, algo=tpe.suggest)
    return pipe.set_params(**space_eval(space, space_optimized))



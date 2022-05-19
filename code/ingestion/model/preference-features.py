# Databricks notebook source
### NOTE: ADD DOCSTRINGS, FACTOR OUT HYPEROPT SPACE AND ITERNO?

#
##
### PREFERENCE

#
##
### HELPERS

def _set_recom(params, seed):
    from pyspark.ml.recommendation import ALS
    
    # unpack params
    rank = int(params["rank"]) # int
    maxiter = int(params["maxIter"]) # int
    regparam = params["regParam"] # float
    alpha = params["alpha"] # float    
    # init
    als = ALS(userCol="user_id", itemCol="product_id", ratingCol="implicit_feedback",
        implicitPrefs=True, coldStartStrategy="drop", nonnegative=True,
        seed=seed, rank=rank, maxIter=maxiter, alpha=alpha, regParam=regparam)
    return als

def _eval_recom(params, data, event_type, run_name, seed):
    import mlflow
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from hyperopt import STATUS_OK
    
    # data distri for larger datasets?
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model = _set_recom(params, seed)
        evaluator = RegressionEvaluator(metricName="rmse",
            labelCol="implicit_feedback", predictionCol="prediction")
        cv = CrossValidator(estimator=model, estimatorParamMaps=ParamGridBuilder().build(),
            evaluator=evaluator, seed=seed).fit(data)
        cv_rmse = cv.avgMetrics[0]
        mlflow.log_metric("cv_rmse", cv_rmse)     
        return {"loss":cv_rmse, "status":STATUS_OK}
    
#
##
### DO HYPEROPT SEARCH 

def optimize_recom(events, event_type, seed):
    from functools import partial
    from hyperopt import fmin, tpe, hp, Trials, space_eval
    import mlflow
    
    # form the dataset
    implicit_feedback = (events.groupBy("user_id", "product_id")
        .agg(f.count(f.col("user_session_id")).alias("implicit_feedback")))
    
    # optimization
    # NOTE: MOVE SPACE AND OPTPAR OUTSIDE THE FUNC
    space = {
        "rank": hp.randint("rank", 5, 30),
        "maxIter": hp.randint("maxIter", 5, 40),
        "regParam": hp.loguniform("regParam", -5, 3),
        "alpha": hp.uniform("alpha", 25, 350)}
    run_name = "optimize_recom_"+event_type
    optimized_params = fmin(
        fn=partial(_eval_recom,
            data=implicit_feedback, event_type=event_type,
            run_name=run_name, seed=seed),
        max_evals=1, space=space, algo=tpe.suggest)
    
    # refit
    run_name = "refit_recom_"+event_type
    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()        
        mlflow.log_params(optimized_params)        
        optimized_model = _set_recom(optimized_params, seed=seed)
        optimized_model = optimized_model.fit(implicit_feedback)
        mlflow.spark.log_model(optimized_model, run_name)
        path = f"runs:/{run.info.run_id}/"+run_name
        mlflow.register_model(path, run_name)
        print(f"Optimized recommendation engine stored in \"{path}\".")
        return optimized_model
      
#
##
### GET ENCODED PREFERENCE

def get_user_factors(recom_model, event_type):
    import pyspark.sql.functions as f
    dims = range(recom_model.rank)
    user_factors = recom_model.userFactors\
        .select(f.col("id").alias("user_id"),
            *[f.col("features").getItem(d).alias(event_type+"_latent_factor"+str(d)) for d in dims])
    return user_factors

# COMMAND ----------



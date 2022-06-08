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

def _get_data(dataset_name, week_step):
    # NOTE: customer-model/workhorse.py/split_save_customer_model
    # NOTE: refactor this for own model/profit measures!
    import pyspark.sql.functions as f
    train = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>week_step).toPandas()
    test = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")==week_step).toPandas()
    out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
    feat_cols = [c for c in train.columns if c not in set(out_cols)]
    return {"train":
                {"X":_optimize_numeric_dtypes(train.loc[:,feat_cols]),
                 "y":train["target_event"]},
            "test":
                {"X":_optimize_numeric_dtypes(test.loc[:,feat_cols]),
                 "y":test["target_event"]},
            "name":f"{dataset_name}_{week_step}"}

def glue_pipeline(pipe, data, refit=True):
    import mlflow
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    
    dataset_name = data["name"]
    week_step = dataset_name.split("_")[1]
    data_pipe_name = dataset_name+"_"+pipe["name"]
   
    if refit:
        exp_name = f"{data_pipe_name}_hyperopt"
        exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
        mlflow.set_experiment(experiment_id=exp_id)
        with mlflow.start_run() as run:    
            optimized_pipeline = _optimize_pipeline(
                data["train"]["X"], data["train"]["y"],
                    pipe["steps"], pipe["space"])
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
        exp_name = f"models:/{data_pipe_name}_model/None"
        calibrated_pipeline = mlflow.sklearn.load_model(exp_name)
        
    # evaluate & push into the delta
    results = pd.DataFrame([dict(_get_performance(
        calibrated_pipeline, v["X"], v["y"]),
            **{"type":k,"week_step":week_step,"pipe":pipe_name})
                for k,v in data.items() if k in ["train", "test"]])
    spark.createDataFrame(results)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance_evaluation")
    return results

# COMMAND ----------

# NOTE: streamline this, push pipe spaces to optimization directly
# NOTE: 

import datetime
import mlflow

pipe_name = "dt"
data = _get_data("rees46", 2)
pipe = get_pipeline(pipe_name)
steps = pipe["steps"]
space = pipe["space"]

b_data = sc.broadcast(data)
b_pipe = sc.broadcast(pipe)



# COMMAND ----------

with mlflow.start_run() as run:  
    results = _optimize_pipeline(
        b_data.value["train"]["X"], b_data.value["train"]["y"],
            b_pipe.value["steps"], b_pipe.value["space"])

# COMMAND ----------

# NOTE: broadcast this across the workers?
for week_step in range(2,4):
    data = _get_data("rees46", week_step)
    for pipe_name in ["hgb","dt"]:
        pipe = get_pipes(pipe_name)
        glue_pipeline(pipe, data, True)

# COMMAND ----------

# RUNTIME ESTIMATES/ fast algos
# OPT - 2 mins a fit, 10 fits within an opt, 10 opt across weeks = 200 mins opt
# CAL - 3 mins a fit, 5 fits per cal, 10 cals across weeks = 150 mins cal
# SUBTOTAL 6 h
## ALL ALGOS >= 48h wo distributed backends

# STRATEGY 1
# paralelize on dataset slices

# STRATEGY 2
# paralelize within steps - OPT, CAL

# STRATEGY 3
# rewrite pipelines to spark pipes

# STRATEGY 4
# redesign pipeline and experiment steps

# STRATEGY 5
# compress dataset, eliminate prev steps

# COMMAND ----------

import numpy as np
 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
 
import mlflow
X, y = fetch_california_housing(return_X_y=True)

from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
X = scaler.fit_transform(X)
y_discrete = np.where(y < np.median(y), 0, 1)


def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    else:
        return 0
    accuracy = cross_val_score(clf, X, y_discrete).mean()
    
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}

search_space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'max_depth': hp.randint('max_depth', 2, 5),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
    {
        'type': 'logreg',
        'C': hp.lognormal('LR_C', 0, 1.0),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    },
])

algo=tpe.suggest
spark_trials = SparkTrials(parallelism=4)
#spark_trials=Trials()
data_pipe_name = "lulw"
exp_name = f"{data_pipe_name}_hyperopt"
exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
mlflow.set_experiment(experiment_id=exp_id)
with mlflow.start_run():
    best_result = fmin(
        fn=objective, 
        space=search_space,
        algo=algo,
        max_evals=32,
        trials=spark_trials)
    mlflow.log_params(best_result)

# COMMAND ----------

# Databricks notebook source
# MAGIC 
# MAGIC %md
# MAGIC # Hyperopt
# MAGIC 
# MAGIC The [Hyperopt library](https://github.com/hyperopt/hyperopt) allows for parallel hyperparameter tuning using either random search or Tree of Parzen Estimators (TPE). With MLflow, we can record the hyperparameters and corresponding metrics for each hyperparameter combination. You can read more on [SparkTrials w/ Hyperopt](https://github.com/hyperopt/hyperopt/blob/master/docs/templates/scaleout/spark.md).
# MAGIC 
# MAGIC For this example, we will parallelize the hyperparameter search for a tf.keras model with the california housing dataset.

# COMMAND ----------

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cal_housing = fetch_california_housing()

# split 80/20 train-test
X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2,
                                                    random_state=1)
# Feature-wise standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC ## Keras Model
# MAGIC 
# MAGIC We will define our NN in Keras and use the hyperparameters given by HyperOpt.
# MAGIC 
# MAGIC <img alt="Side Note" title="Side Note" style="vertical-align: text-bottom; position: relative; height:1.75em; top:0.05em; transform:rotate(15deg)" src="https://files.training.databricks.com/static/images/icon-note.webp"/> We need to import `tensorflow` within the function due to a pickling issue.  <a href="https://docs.databricks.com/applications/deep-learning/single-node-training/tensorflow.html#tensorflow-2-known-issues" target="_blank">See known issues here.</a>

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import mlflow
import mlflow.keras
tf.random.set_seed(42)

def create_model(hpo):
    model = Sequential()
    model.add(Dense(int(hpo["dense_l1"]), input_dim=8, activation="relu"))
    model.add(Dense(int(hpo["dense_l2"]), activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model

# COMMAND ----------

from hyperopt import fmin, hp, tpe, STATUS_OK, SparkTrials

def runNN(hpo):
  # Need to include the TF import due to serialization issues
    import tensorflow as tf
  
    model = create_model(hpo)

    # Select Optimizer
    optimizer_call = getattr(tf.keras.optimizers, hpo["optimizer"])
    optimizer = optimizer_call(learning_rate=hpo["learning_rate"])

  # Compile model
    model.compile(loss="mse",
                optimizer=optimizer,
                metrics=["mse"])

    history = model.fit(X_train, y_train, validation_split=.2, epochs=10, verbose=2)

    # Evaluate our model
    score = model.evaluate(X_test, y_test, verbose=0)
    obj_metric = score[0]  
    return {"loss": obj_metric, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup hyperparameter space and training
# MAGIC 
# MAGIC We need to create a search space for HyperOpt and set up SparkTrials to allow HyperOpt to run in parallel using Spark worker nodes. We can also start a MLflow run to automatically track the results of HyperOpt's tuning trials.

# COMMAND ----------

space = {
  "dense_l1": hp.quniform("dense_l1", 10, 30, 1),
  "dense_l2": hp.quniform("dense_l2", 10, 30, 1),
  "learning_rate": hp.loguniform("learning_rate", -5, 0),
  "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"])
 }

spark_trials = SparkTrials(parallelism=4)

with mlflow.start_run():
    best_hyperparam = fmin(fn=runNN, 
                         space=space, 
                         algo=tpe.suggest, 
                         max_evals=16, 
                         trials=spark_trials)

best_hyperparam

# COMMAND ----------

# MAGIC %md
# MAGIC To view the MLflow experiment associated with the notebook, click the Runs icon in the notebook context bar on the upper right. There, you can view all runs. You can also bring up the full MLflow UI by clicking the button on the upper right that reads View Experiment UI when you hover over it.
# MAGIC 
# MAGIC To understand the effect of tuning a hyperparameter:
# MAGIC 
# MAGIC 0. Select the resulting runs and click Compare.
# MAGIC 0. In the Scatter Plot, select a hyperparameter for the X-axis and loss for the Y-axis.

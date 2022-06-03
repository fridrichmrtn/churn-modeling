# Databricks notebook source
# MAGIC %run ./pipelines

# COMMAND ----------

#
##
### PIPELINE HYPEROPT

hyperopt_config = {
    "max_evals":10,
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


def _optimize_pipeline(X, y, dataset_name, pipe_name):

    pipe = get_pipe(pipe_name)
    space = get_space(pipe_name)
    max_evals = hyperopt_config["max_evals"]
    seed = hyperopt_config["seed"]
    
    import mlflow
    from hyperopt import fmin, tpe, space_eval
    from functools import partial

    # run optimization & staff
    exp_name = f"{dataset_name}_optimize_pipe_{pipe_name}"
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        space_optimized = fmin(
            fn=partial(_evaluate_pipeline,
                pipe=pipe, X=X, y=y, seed=seed),
            max_evals=max_evals, space=space, algo=tpe.suggest)
        return space_eval(space, space_optimized)


# COMMAND ----------

# lets roll
import pyspark.sql.functions as f
dfp = spark.table("churndb.rees46_customer_model").where(f.col("week_step")==0).toPandas()
out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
feat_cols = [c for c in dfp.columns if c not in set(out_cols)]

X = dfp.loc[:, feat_cols].copy()
y = dfp["target_event"]

# COMMAND ----------

dfp = spark.table("churndb.rees46_customer_model")#.toPandas()

# COMMAND ----------

dfp.write.parquet("dbfs:/mnt/rees46/customer_model")

# COMMAND ----------

dataset_name = "test"
pipe_name = "svm_rbf"
_optimize_pipeline(X, y, dataset_name, pipe_name)

# COMMAND ----------

dataset_name = "test"
pipe_name = "svm_rbf"
_optimize_pipeline(X, y, dataset_name, pipe_name)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)

# COMMAND ----------

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization

def cm():
    model = Sequential()
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# COMMAND ----------

try:
    import scikeras
except ImportError:
    !python -m pip install scikeras

# COMMAND ----------

from tensorflow import keras


def get_clf(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size))
        model.add(keras.layers.LeakyReLU(alpha=0.3))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    return model

from scikeras.wrappers import KerasClassifier


clf = KerasClassifier(
    model=get_clf,
    loss="binary_crossentropy",
    hidden_layer_sizes=(100,),
    dropout=0.5,
    optimizer__learning_rate=0.0005,
    optimizer="adam",
    epochs=10,
    batch_size=5
)

# COMMAND ----------



# COMMAND ----------

from sklearn.calibration import CalibratedClassifierCV
from scikeras.wrappers import KerasClassifier

model = CalibratedClassifierCV(clf)
model.fit(X_train, y_train)

# COMMAND ----------

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(X_train, y_train, epochs=200, batch_size=10, validation_data=(X_val, y_val), callbacks=[es])

# COMMAND ----------

from sklearn.metrics import f1_score, accuracy_score
print(f1_score(y_test, model.predict(X_test)))

# COMMAND ----------

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
mlp = HistGradientBoostingClassifier()
mlp.fit(X_train, y_train)
print(f1_score(y_test, mlp.predict(X_test)))

# COMMAND ----------



# COMMAND ----------

# CONSIDER REWIRING PRUNING TO ONE PARAM - TREES

# dataset name mapping

# all previous - hyperopt and training
# log model
# evaluate - training, testing
# log metrics

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

# NOTE: CLASSED
def _get_cols(train):
    out_multi_cols = ["user_id", "row_id", "target_event",
        "target_revenue", "week_step", "target_cap"]
    out_standard_cols = ["cap", "cap_month_lag0", "cap_month_lag1",
        "cap_month_lag2", "cap_month_lag3", "cap_month_ma3"]
    multi_cols = [c for c in train.columns if c not in set(out_multi_cols)]
    standard_cols = [c for c in multi_cols if c not in set(out_standard_cols)]
    return (standard_cols, multi_cols)

def _get_dataset(dataset_name, week_step):
    import pyspark.sql.functions as f
    
    data = spark.table(f"churndb.{dataset_name}_customer_model")\
        .where(f.col("week_step")>=week_step).toPandas()
    train = data[data.week_step>week_step]
    test = data[data.week_step==week_step]			
    standard_cols, multi_cols = _get_cols(train)
    return {
      "train":
            {"raw":_optimize_numeric_dtypes(train),
             "week_step":week_step,
             "columns":{"standard":standard_cols, "multi-output":multi_cols},
             "name":f"{dataset_name}_{week_step}"},
      "test":
            {"raw":_optimize_numeric_dtypes(test),
             "week_step":week_step,
             "columns":{"standard":standard_cols, "multi-output":multi_cols},
             "name":f"{dataset_name}_{week_step}"}}      
    
def _fit_calibrated_pipeline(data, pipe):
    import mlflow
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    
    X, y = _get_Xy(data, pipe)    
    exp_name = "{}_{}_refit".format(data["name"],pipe["name"])
    exp_id = _get_exp_id(f"/Shared/dev/{exp_name}")
    mlflow.set_experiment(experiment_id=exp_id)
    with mlflow.start_run() as run:
        #prefit_model = pipe["steps"].fit(X_train, y_train)
        pipe["fitted"] = pipe["calibration"](pipe["steps"],
            cv=3, method="sigmoid").fit(X, y)
        
        #pipe["fitted"] = clone(pipe["steps"]).fit(X,y)
        mlflow.sklearn.log_model(pipe["fitted"],
             exp_name, registered_model_name=exp_name)
    return pipe    

def _evaluate_pipeline(data, pipe):
    # evaluate & push into the delta
    X, y = _get_Xy(data, pipe) 
    dataset_name = data["train"]["name"].split("_")[0]
    pipe["results"] = pd.DataFrame([dict(_get_performance(
        pipe["fitted"], v["X"], v["y"]),
            **{"type":k,"week_step":v["week_step"],"pipe":pipe["name"]})
                for k,v in data.items()]) # NOT NICE
    spark.createDataFrame(pipe["results"])\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_performance")
    return pipe

def _get_predictions(data, pipe):
    import numpy as np
    import pandas as pd
    predictions = []
    for temp_type, temp_data in data.items():
        
        X, y = _get_Xy(temp_data, pipe)
        y_pred_event = pipe["fitted"].predict(X)
        y_pred_event_proba = pipe["fitted"].predict_proba(X)[:,1]
        if pipe["type"]=="multi-output":
            y_pred_cap = pipe["fitted"].predict(X, scope="regression")
        else:
            y_pred_cap = np.nan
            
        predictions.append(pd.DataFrame.from_dict({
            "pipe":pipe["name"],
            "type":temp_type,
            "week_step":temp_data["week_step"],
            "user_id":temp_data["raw"]["user_id"],
            "row_id":temp_data["raw"]["row_id"],
            "y_pred_event":y_pred_event.astype("int"),
            "y_pred_event_proba":y_pred_event_proba,
            "y_pred_cap":y_pred_cap}))
    return _optimize_numeric_dtypes(pd.concat(predictions))

def _save_predictions(dataset_name, predictions):
    spark.createDataFrame(predictions)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_predictions")
    return None
    
def glue_pipeline(dataset_name, week_range, drop_predictions=True):
    if drop_predictions:
        spark.sql(f"DROP TABLE IF EXISTS churndb.{dataset_name}_predictions;")
    for week_step in week_range:
        data = _get_dataset(dataset_name, week_step)
        for pipe_name, pipe in pipelines.items():
            #pipe = _optimize_pipeline(data["train"], pipe)
            pipe = _fit_calibrated_pipeline(data["train"], pipe)
            _save_predictions(dataset_name, _get_predictions(data, pipe))
    return None

# COMMAND ----------

import numpy as np
import pandas as pd
#import copy
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import  _SigmoidCalibration
from sklearn.isotonic import IsotonicRegression

class MultiOutputCalibrationCV(BaseEstimator):
    def __init__(self, base_estimator, method="sigmoid", cv=3):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        calibrated_pairs = []
        
        scv = StratifiedKFold(n_splits=self.cv)
        for train_index, test_index in scv.split(X, y[:,0]):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # fit combinet
            base_estimator = clone(self.base_estimator)
            base_estimator.fit(X_train, y_train)
            y_pred = base_estimator.predict_proba(X_test)
        
            # fit calibrator
            if self.method=="isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
            if self.method=="sigmoid":
                calibrator = _SigmoidCalibration()
            calibrator.fit(y_pred[:,1].T, y_test[:,0])
            calibrated_pairs.append((base_estimator, calibrator))
        self.calibrated_pairs = calibrated_pairs
        return self

    def predict_proba(self, X):
        # calibrated positive class
        calibrated_class = np.zeros(shape=(X.shape[0], len(self.calibrated_pairs)))
        for i, calibrated_pair in enumerate(self.calibrated_pairs):
            raw_prediction = calibrated_pair[0].predict_proba(X)[:,1]
            #calibrated_class[:,i] = raw_prediction
            calibrated_class[:,i] = calibrated_pair[1].predict(raw_prediction.T)
        calibrated_class = np.mean(calibrated_class, axis=1)
        return np.column_stack([1-calibrated_class, calibrated_class])

    def predict_reg(self, X):
        calibrated_reg = np.zeros(shape=(X.shape[0], len(self.calibrated_pairs)))
        for i, calibrated_pair in enumerate(self.calibrated_pairs):
            calibrated_reg[:,i] = calibrated_pair[0].predict(X, scope="regression")
        return np.mean(calibrated_reg, axis=1)
    
    def predict_full(self, X):
        return np.column_stack([(self.predict_proba(X)[:,1]>0.5).astype("int"),
            self.predict_reg(X)])
    
    def predict(self, X, scope="classification"):

        if scope=="classification":
            return (self.predict_proba(X)[:,1]>0.5).astype("int")
        if scope=="regression":
            return self.predict_reg(X)
        if scope=="full":
            return self.predict_full(X)

# COMMAND ----------

from sklearn.metrics import f1_score

data = _get_dataset("retailrocket", 1)
pipe = pipelines["combinet"]
X_train, y_train = _get_Xy(data["train"], pipe)
X_test, y_test = _get_Xy(data["test"], pipe)
pipe["steps"].fit(X_train, y_train)

print(f1_score(y_test[:,0], pipe["steps"].predict(X_test)))

# COMMAND ----------

mocv = MultiOutputCalibrationCV(base_estimator=pipe["steps"], method="isotonic")
mocv.fit(X_train, y_train)
print(f1_score(y_test[:,0], mocv.predict(X_test)))

# COMMAND ----------

from sklearn.metrics import brier_score_loss
brier_score_loss(y_test[:,0], mocv.predict(X_test))

# COMMAND ----------

from sklearn.calibration import calibration_curve
x, y = calibration_curve(y_test[:,0], mocv.predict(X_test))

# COMMAND ----------

ze = np.zeros_like(y_test)
for i, c in enumerate(mocv.calibrated_pairs):
    ze += c[0].predict_proba(X_test)
ze = ze/3
ze[:,1] = (ze[:,1]>0.5).astype("int")
print(f1_score(y_test[:,0],ze[:,1]))
    

# COMMAND ----------

X, y = X_train, y_train

scv = StratifiedKFold(n_splits=10)
train_index, test_index = list(scv.split(X, y[:,0]))[0]
X0, X1 = X[train_index], X[test_index]
y0, y1 = y[train_index], y[test_index]

# fit combinet
base_estimator = clone(pipe["steps"])
base_estimator.fit(X0, y0)
y_pred = base_estimator.predict_proba(X1)
calibrator = IsotonicRegression()
calibrator.fit(y_pred[:,1].reshape(-1,1), y1[:,0])
    

# COMMAND ----------

y_proba = base_estimator.predict_proba(X_test)[:,1].reshape(-1,1)
y_prediction = (calibrator.predict(y_proba)>.5).astype("int")

# COMMAND ----------

f1_score(y_test[:,0], y_prediction)

# COMMAND ----------



# Databricks notebook source
# load the delta tab

# get original dataset stats - DONE
    # start date, end date
    # shape
    # no of customers,no of interactions, no of transactions
    # revenue

# filter

# get 10 most correlated features

# do individual feature stats

# multicol

# link to target?

# COMMAND ----------

dataset_name = "rees46"
data_path = f"dbfs:/mnt/{dataset_name}/delta/"
events = spark.read.format("delta").load(data_path+"events")
events.show(3)

# COMMAND ----------

# basic stats
import pyspark.sql.functions as f
events.agg(f.min(f.col("event_time")).alias("min_time"), f.max(f.col("event_time")).alias("max_time"),
    f.countDistinct(f.col("user_id")).alias("customers"),
    f.count("user_id").alias("interactions"),
    f.sum("view").alias("views"),
    f.sum("cart").alias("carts"),           
    f.sum("purchase").alias("purchases"),
    f.sum(f.when(f.col("purchase")==1, f.col("price")).otherwise(0)).alias("revenue")).show()

# add simulated profit, this should be probably part of the load-transform

# COMMAND ----------

import pyspark.sql.functions as f
df = spark.table("churndb.rees46_customer_model").where(f.col("week_step")==0)#.toPandas()

# COMMAND ----------

import pyspark.sql.functions as f
dfp = spark.table("churndb.rees46_customer_model").where(f.col("week_step")<5).toPandas()
out_cols = ["user_id", "target_event", "target_revenue", "week_step"]
feat_cols = [c for c in dfp.columns if c not in set(out_cols)]

# COMMAND ----------



# COMMAND ----------

#
##
### WRAP OPTIMIZATION AROUND THIS

#
##
### CONSIDER MULTIPLE PROCESSING BRANCHES

#
##
### "LINEAR" AND TREES

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X_train, X_test, y_train, y_test = train_test_split(dfp.loc[:, feat_cols],
    dfp["target_event"], test_size=0.4, stratify=dfp["target_event"], random_state=1)

X_train_1 = dfp.loc[dfp.week_step>1, feat_cols]
y_train_1 = dfp.loc[dfp.week_step>1, "target_event"]

X_train_2 = dfp.loc[dfp.week_step==1, feat_cols]
y_train_2 = dfp.loc[dfp.week_step==1, "target_event"]

X_test = dfp.loc[dfp.week_step==0, feat_cols]
y_test = dfp.loc[dfp.week_step==0, "target_event"]

lr = Pipeline([
    ("vt", VarianceThreshold(threshold=0.1)),
    ("qt", QuantileTransformer()),
    ("hfs", HierarchicalFeatureSelector(n_features=100)),
    ("ss", RandomUnderSampler()),
    ("lr", LogisticRegression())])

lr.fit(X_train_1, y_train_1)

print(accuracy_score(y_test,lr.predict(X_test)))
print(f1_score(y_test,lr.predict(X_test)))
lr_test_pred = lr.predict_proba(X_test)

# COMMAND ----------

#
##
### PIPELINE HYPEROPT

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


# COMMAND ----------

def _get_exp_id(exp_path):
    import mlflow
    try:
        exp_id = mlflow.get_experiment_by_name(exp_path).experiment_id
    except:
        exp_id = mlflow.create_experiment(exp_path)
    return exp_id

# COMMAND ----------

#
##
### LOGISTIC REGRESSION TESTING BED





# OPT SPACE
from hyperopt import hp


# COMMAND ----------

#
##
### GET OPTIMIZED MODEL/AND OR OPTIMIZE MODEL DO THIS ACC TO THE PREFERENCE MODELS


# COMMAND ----------

# NOTE: consider loading params from the pipe conf
def _optimize_pipeline(dataset_name="test", pipe_name="lr"):
    from imblearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    


    # unpack params from conf
    pipe = Pipeline([
        ("variance_filter", VarianceThreshold(threshold=0.1)),
        ("data_scaler", PowerTransformer()),
        ("feature_selector", HierarchicalFeatureSelector(n_features=100)),
        ("data_sampler", RandomUnderSampler()),
        ("model", LogisticRegression())])

    space = {
        "variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-2, 10**0),
        "data_scaler":hp.choice("data_scaler", scalers), # here we have to try multiple transforms
        "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
        "data_sampler":hp.choice("data_sampler", samplers),
        "model__C":hp.uniform("model__C", 10**-5, 10**0),
        "model__max_iter":hp.randint("model__max_iter",10**2,5*10**3)}

    seed = 0
    X, y = X_train_1, y_train_1
    max_evals = 25
    
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

mehe = _optimize_pipeline()

# COMMAND ----------

# start to generalize this so we can roll it out to multiple pipes
# add spark trials - later

# COMMAND ----------


from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

mlp = Pipeline([
    ("vt", VarianceThreshold(threshold=0.1)),
    ("qt", QuantileTransformer()),
    ("hfs", HierarchicalFeatureSelector(n_features=100)),
    ("ss", RandomUnderSampler()),
    ("lr", CalibratedClassifierCV(MLPClassifier(hidden_layer_sizes=(32, 16, ))))])

mlp.fit(X_train_1, y_train_1)

print(accuracy_score(y_test,mlp.predict(X_test)))
print(f1_score(y_test,mlp.predict(X_test)))
mlp_test_pred = mlp.predict_proba(X_test)

hgb = Pipeline([
    #("vt", VarianceThreshold(threshold=0.1)),
    #("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_clusters=100)),
    #("ss", RandomOverSampler()),
    ("hgb", CalibratedClassifierCV(HistGradientBoostingClassifier()))])

hgb.fit(X_train_1, y_train_1)

print(accuracy_score(y_test,hgb.predict(X_test)))
print(f1_score(y_test,hgb.predict(X_test)))
hgbc_test_pred = hgb.predict_proba(X_test)



lgm = Pipeline([
    #("vt", VarianceThreshold(threshold=0.1)),
    #("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_clusters=100)),
    #("ss", RandomOverSampler()),
    ("lgm", CalibratedClassifierCV(LGBMClassifier()))])

lgm.fit(X_train_1, y_train_1)

print(accuracy_score(y_test,lgm.predict(X_test)))
print(f1_score(y_test,lgm.predict(X_test)))
lgbm_test_pred = lgm.predict_proba(X_test)

xgb = Pipeline([
    #("vt", VarianceThreshold(threshold=0.1)),
    #("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_clusters=100)),
    #("ss", RandomOverSampler()),
    ("lr", CalibratedClassifierCV(XGBClassifier()))])

xgb.fit(X_train_1, y_train_1)

print(accuracy_score(y_test,xgb.predict(X_test)))
print(f1_score(y_test,xgb.predict(X_test)))
xgb_test_pred = xgb.predict_proba(X_test)

# COMMAND ----------

!pip install deslib

# COMMAND ----------

#importing DCS techniques from DESlib
from deslib.dcs.ola import OLA
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB

#import DES techniques from DESlib
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES

class_pool = [lr, mlp, hgb, lgm, xgb]

apri = APriori(class_pool)
apri.fit(X_train_2, y_train_2)
apri_pred = apri.predict(X_test)
print(accuracy_score(y_test, apri_pred))
print(f1_score(y_test, apri_pred))

# COMMAND ----------

# just simply voting
print(accuracy_score(y_test,[0 if r[0] > r[1] else 1 for r in (lr_test_pred+lgbm_test_pred+hgbc_test_pred+xgb_test_pred+mlp_test_pred)/5]))
print(f1_score(y_test,[0 if r[0] > r[1] else 1 for r in (lr_test_pred+lgbm_test_pred+hgbc_test_pred+xgb_test_pred+mlp_test_pred)/5]))

# COMMAND ----------

# simulation

# for each category, draw initial margin
    # for each product in category draw margin from normal distribution with category margin as exp val
        # for each product and day draw margin from normal distribution with exp val from previous day
        
# align this with 

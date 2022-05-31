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

# imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import AgglomerativeClustering

class TransposeDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.copy().T
    
class HierarchicalFeatureSelector(SelectorMixin, BaseEstimator):
    
    def __init__(self, n_features):
        self.n_features = n_features
    
    def get_cluster_assignments_(self, data):
        data = data.loc[:,self.results_.feature.values]
        pipe = Pipeline([("rotate",TransposeDataFrame()),
            ("cluster", AgglomerativeClustering(n_clusters=self.n_features))])
        return pipe.fit_predict(data)
    
    def get_correlations_(self, X, y):
        data = X.copy()
        data["target"] = y
        correlations = pd.DataFrame(data.corr().abs()["target"])
        correlations = correlations.reset_index()
        correlations.columns = ["feature", "abs_corr"]
        return correlations[correlations.feature!="target"]
    
    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        self.in_features_ =  X.columns
        self.results_ = self.get_correlations_(X, y)
        self.results_["cluster"] = self.get_cluster_assignments_(X)
        self.best_ = self.results_.merge(self.results_.groupby("cluster",
            as_index=False).abs_corr.max(), on=["cluster", "abs_corr"])\
                .drop_duplicates(["cluster", "abs_corr"]).dropna()
        return self
    
    def _get_support_mask(self):
        return np.array([c in set(self.best_.feature) for c in self.in_features_])

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

X_train = dfp.loc[dfp.week_step>0, feat_cols]
y_train = dfp.loc[dfp.week_step>0, "target_event"]

X_test = dfp.loc[dfp.week_step==0, feat_cols]
y_test = dfp.loc[dfp.week_step==0, "target_event"]

pipe = Pipeline([
    ("vt", VarianceThreshold(threshold=0.1)),
    ("qt", QuantileTransformer()),
    ("hfs", HierarchicalFeatureSelector(n_features=100)),
    ("ss", RandomUnderSampler()),
    ("lr", LogisticRegression())])

pipe.fit(X_train, y_train)

print(accuracy_score(y_test,pipe.predict(X_test)))
print(f1_score(y_test,pipe.predict(X_test)))

# COMMAND ----------

X_train = dfp.loc[dfp.week_step>0, feat_cols]
y_train = dfp.loc[dfp.week_step>0, "target_event"]

X_test = dfp.loc[dfp.week_step==0, feat_cols]
y_test = dfp.loc[dfp.week_step==0, "target_event"]

pipe = Pipeline([
    ("vt", VarianceThreshold(threshold=0.1)),
    ("qt", QuantileTransformer()),
    ("hfs", HierarchicalFeatureSelector(n_features=100)),
    ("ss", RandomUnderSampler()),
    ("lr", LogisticRegression())])

pipe.fit(X_train, y_train)

print(accuracy_score(y_test,pipe.predict(X_test)))
print(f1_score(y_test,pipe.predict(X_test)))
lr_test_pred = pipe.predict_proba(X_test)

# COMMAND ----------

from sklearn.neural_network import MLPClassifier

pipe = Pipeline([
    ("vt", VarianceThreshold(threshold=0.1)),
    ("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_features=100)),
    ("ss", RandomUnderSampler()),
    ("lr", MLPClassifier())])

pipe.fit(X_train, y_train)

print(accuracy_score(y_test,pipe.predict(X_test)))
print(f1_score(y_test,pipe.predict(X_test)))
mlp_test_pred = pipe.predict_proba(X_test)

# COMMAND ----------

from sklearn.ensemble import HistGradientBoostingClassifier

pipe = Pipeline([
    #("vt", VarianceThreshold(threshold=0.1)),
    #("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_clusters=100)),
    #("ss", RandomOverSampler()),
    ("lr", HistGradientBoostingClassifier())])

pipe = pipe
pipe.fit(X_train, y_train)

print(accuracy_score(y_test,pipe.predict(X_test)))
print(f1_score(y_test,pipe.predict(X_test)))
hgbc_test_pred = pipe.predict_proba(X_test)

# COMMAND ----------

from lightgbm import LGBMClassifier

pipe = Pipeline([
    #("vt", VarianceThreshold(threshold=0.1)),
    #("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_clusters=100)),
    #("ss", RandomOverSampler()),
    ("lr", LGBMClassifier())])

pipe = pipe
pipe.fit(X_train, y_train)

print(accuracy_score(y_test,pipe.predict(X_test)))
print(f1_score(y_test,pipe.predict(X_test)))
lgbm_test_pred = pipe.predict_proba(X_test)

# COMMAND ----------

from xgboost import XGBClassifier

pipe = Pipeline([
    #("vt", VarianceThreshold(threshold=0.1)),
    #("qt", QuantileTransformer()),
    #("hfs", HierarchicalFeatureSelector(n_clusters=100)),
    #("ss", RandomOverSampler()),
    ("lr", XGBClassifier())])

pipe = pipe
pipe.fit(X_train, y_train)

print(accuracy_score(y_test,pipe.predict(X_test)))
print(f1_score(y_test,pipe.predict(X_test)))
xgb_test_pred = pipe.predict_proba(X_test)

# COMMAND ----------

accuracy_score(y_test,[0 if r[0] > r[1] else 1 for r in (lr_test_pred+lgbm_test_pred+hgbc_test_pred+xgb_test_pred+mlp_test_pred)/5])

# COMMAND ----------

f1_score(y_test,[0 if r[0] > r[1] else 1 for r in (lr_test_pred+lgbm_test_pred+hgbc_test_pred+xgb_test_pred+mlp_test_pred)/5])

# COMMAND ----------

lr.predict(X_test)

# COMMAND ----------

y_test

# COMMAND ----------



# COMMAND ----------


lgbm = LGBMClassifier()
lgbm.fit(X_train.loc[:,rfe.support_],y_train)

# COMMAND ----------

np.sum(rfe.support_)

# COMMAND ----------



# COMMAND ----------

print(accuracy_score(y_test,lgbm.predict(X_test.loc[:,rfe.support_])))
print(f1_score(y_test,lgbm.predict(X_test.loc[:,rfe.support_])))

# COMMAND ----------

np.sum(pca.explained_variance_ratio_[:50])

# COMMAND ----------

# simulation

# for each category, draw initial margin
    # for each product in category draw margin from normal distribution with category margin as exp val
        # for each product and day draw margin from normal distribution with exp val from previous day
        
# align this with 

# COMMAND ----------



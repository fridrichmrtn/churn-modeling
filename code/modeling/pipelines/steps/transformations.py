# Databricks notebook source
# MAGIC %run "../utils"

# COMMAND ----------

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from imblearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import AgglomerativeClustering

# COMMAND ----------

#
##
### HIERARCHICAL FEATURE SELECTOR

class _DataFrameTransposer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.copy().T

class HierarchicalFeatureSelector(SelectorMixin, BaseEstimator):
  
    def __init__(self, n_features=75, alpha=0.001):
        self.n_features = n_features
        self.alpha = alpha
        self.y_=None   

    def _get_cluster_assignments(self, data):
        data = data.loc[:,self.results_.feature.values]
        n_components = data.shape[1]
        n_clusters = np.minimum(n_components, self.n_features)
        pipe = Pipeline([("rotate", _DataFrameTransposer()),
            ("grp", GaussianRandomProjection(n_components=n_components)),
            ("cluster", AgglomerativeClustering(n_clusters=n_clusters))])
        return pipe.fit_predict(data)
    
    @reduce_y
    def _get_correlations(self, X, y):
        tf_corr = [pearsonr(y, X[c]) for c in X.columns]
        correlations = pd.DataFrame(tf_corr, index=X.columns).reset_index()
        correlations.columns = ["feature", "r", "p"]
        correlations["abs_r"] = correlations.r.abs()
        correlations["sf"] = correlations.p<=self.alpha/X.shape[1]
        return correlations
    
    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.in_features_ =  X.columns
        self.results_ = self._get_correlations(X, y)
        
        if (np.sum(self.results_.sf)==0):
            raise Exception("Correlation tests fail for all features. Check input data.")
        else:
            self.results_["cluster"] = self._get_cluster_assignments(X)
            self.best_ = (self.results_[self.results_.sf]
                .merge(self.results_.groupby("cluster",
                    as_index=False).abs_r.max(), on=["cluster", "abs_r"])
                    .drop_duplicates(["cluster", "abs_r"]).dropna())            
        return self
    
    def _get_support_mask(self):
        return np.array([c in set(self.best_.feature) for c in self.in_features_])        

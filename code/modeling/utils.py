# Databricks notebook source
#
##
### CUSTOM PIECES OF CODE

# HIERARCHICAL FEATURE SELECTOR
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from imblearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering

class DataFrameTransposer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.copy().T
    
class HierarchicalFeatureSelector(SelectorMixin, BaseEstimator):
    def __init__(self, n_features=10):
        self.n_features = n_features
    def get_cluster_assignments_(self, data):
        data = data.loc[:,self.results_.feature.values]
        pipe = Pipeline([("rotate", DataFrameTransposer()),
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
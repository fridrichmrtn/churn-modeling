# Databricks notebook source
#
##
### CUSTOM PIECES OF CODE

# HIERARCHICAL FEATURE SELECTOR
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
    def __init__(self, n_features=10, alpha=0.001):
        self.n_features = n_features
        self.alpha = alpha
    def get_cluster_assignments_(self, data):
        data = data.loc[:,self.results_.feature.values]
        pipe = Pipeline([("rotate", DataFrameTransposer()),
            ("cluster", AgglomerativeClustering(n_clusters=self.n_features))])
        return pipe.fit_predict(data)
    def get_correlations_(self, X, y):
        tf_corr = [pearsonr(y, X[c]) for c in X.columns]
        correlations = pd.DataFrame(tf_corr, index=X.columns).reset_index()
        correlations.columns = ["feature", "r", "p"]
        correlations["abs_r"] = correlations.r.abs()
        correlations["sf"] = correlations.p<=self.alpha/X.shape[1]
        return correlations
    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.in_features_ =  X.columns
        self.results_ = self.get_correlations_(X, y)
        if np.sum(self.results_.sf)<= self.n_features:
            self.best_ = self.results_[self.results_.sf]
        else:
            self.results_["cluster"] = self.get_cluster_assignments_(X)
            self.best_ = self.results_[self.results_.sf]\
                .merge(self.results_.groupby("cluster",
                    as_index=False).abs_r.max(), on=["cluster", "abs_r"])\
                        .drop_duplicates(["cluster", "abs_r"]).dropna()
        return self
    def _get_support_mask(self):
        return np.array([c in set(self.best_.feature) for c in self.in_features_])
      
# MLPC
from tensorflow import keras
from scikeras.wrappers import KerasClassifier

class MLPClassifier(KerasClassifier):

    def __init__(
        self,
        n_layers=1,
        layer_size=8,
        activation="relu",
        optimizer="adam",
        optimizer__learning_rate=10**-3,
        epochs=50,
        verbose=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.activation = activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

    def _keras_build_fn(self, compile_kwargs):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        for i in range(self.n_layers):
            model.add(keras.layers.Dense(self.layer_size,
                activation=self.activation))
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer=compile_kwargs["optimizer"])
        return model

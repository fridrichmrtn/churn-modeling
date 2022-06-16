# Databricks notebook source
#
##
### CUSTOM STEPS

# FEATURE SELECTOR
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin
from imblearn.pipeline import Pipeline
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import AgglomerativeClustering
from sklearn.utils.validation import check_is_fitted

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
        n_components = data.shape[1]
        pipe = Pipeline([("rotate", DataFrameTransposer()),
            ("pca", GaussianRandomProjection(n_components=n_components)),
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

    def __init__(self, n_layers=1, layer_size=8,
        activation="relu", optimizer="adam",
        #optimizer__learning_rate=10**-3,
        epochs=50, verbose=0, **kwargs,):
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
        model.compile(loss="binary_crossentropy",
            optimizer=compile_kwargs["optimizer"])
        return model

# COMMAND ----------

#
##
### SETUP PREPROCESSING
from hyperopt import hp
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

scaling = [RobustScaler(), QuantileTransformer(), PowerTransformer()]
sampling = [RandomUnderSampler(), RandomOverSampler()]

preprocessing = {
    "smooth":
        {"steps":
            [("variance_filter", VarianceThreshold()),
            ("data_scaler", PowerTransformer()),
            ("feature_selector", HierarchicalFeatureSelector()),
            ("data_sampler", RandomUnderSampler())],
        "space":
            {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-1, 5*10**1),
            "data_scaler":hp.choice("data_scaler", scaling),
            "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
            "data_sampler":hp.choice("data_sampler", sampling)
            }},
    "tree":
         {"steps":
              [("variance_filter", VarianceThreshold()),
              ("data_sampler", RandomUnderSampler())],
         "space":
             {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-2, 10**0),
             "data_sampler":hp.choice("data_sampler", sampling)
             }}}

#
##
### SETUP MODELS
from hyperopt import hp
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier

models = {
    "lr":
          {"model":
               [("lr", LogisticRegression(solver="saga",
                   penalty="elasticnet", max_iter=200))],          
           "space":
               {"lr__C":hp.uniform("lr__C",10**-2,10**1),
                "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1)},
           "preprocessing":"smooth"},
    "svm_lin":
          {"model":
               [("svm_lin", LinearSVC(dual=False))],
           "space":
               {"svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1),
                "svm_lin__penalty":hp.choice("svm_lin__penalty",["l1","l2"])},
          "preprocessing":"smooth"},
    "svm_rbf":
          {"model":
               [("rbf", Nystroem()),
                ("svm_lin", LinearSVC(dual=False))],
           "space":
               {"rbf__n_components":hp.randint("rbf__n_components",20,100),
                "svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1),
                "svm_lin__penalty":hp.choice("svm_lin__penalty",["l1","l2"])},
          "preprocessing":"smooth"},
    
    "mlp":
          {"model":
               [("mlp", MLPClassifier())],
           "space":
               {"mlp__batch_size":hp.randint("mlp__batch_size",2**3,2**7),
                "mlp__epochs":hp.randint("mlp__epochs",10**2,10**3),
                "mlp__n_layers":hp.randint("mlp__n_layers",1,10),
                "mlp__layer_size":hp.randint("mlp__layer_size",2**2,2**8),
                "mlp__activation":hp.choice("mlp__activation",
                    ["tanh", "sigmoid", "relu", keras.layers.LeakyReLU()]),
                "mlp__optimizer__learning_rate":hp.uniform("mlp__optimizer__learning_rate", 10**-5,10**-3),
                "mlp__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])},
          "preprocessing":"smooth"},
    
    "dt":
          {"model":
               [("dt", DecisionTreeClassifier())],
           "space":
               {"dt__max_depth":hp.randint("dt__max_depth",2,30),
                "dt__min_samples_split":hp.randint("dt__min_samples_split",10**1,2*10**2),
                "dt__min_samples_leaf":hp.randint("dt__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__base_estimator__ccp_alpha",0,1),
                "dt__min_impurity_decrease":hp.uniform("dt__min_impurity_decrease",0,.05),
                "dt__min_weight_fraction_leaf":hp.uniform("dt__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree"},
    "rf":
          {"model":
               [("rf", RandomForestClassifier())],
           "space":
               {"rf__n_estimators":hp.randint("rf__n_estimators",25,1000),
                "rf__max_features":hp.uniform("rf__max_features",0.1,.7),
                "rf__max_depth":hp.randint("rf__max_depth",2,30),
                "rf__min_samples_split":hp.randint("rf__min_samples_split",10**1,2*10**2),
                "rf__min_samples_leaf":hp.randint("rf__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__ccp_alpha",0,0.05),
                "rf__min_impurity_decrease":hp.uniform("rf__min_impurity_decrease",0,.05),
                "rf__min_weight_fraction_leaf":hp.uniform("rf__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree"},
    
    "hgb":
          {"model":
               [("hgb", HistGradientBoostingClassifier())],
           "space":
               {
                "hgb__learning_rate":hp.uniform("hgb__learning_rate",0.01,.15),
                "hgb__max_iter":hp.randint("hgb__max_iter",25,1000),
                "hgb__max_leaf_nodes":hp.randint("hgb__max_leaf_nodes",5**2,5**3),
                "hgb__max_depth":hp.randint("hgb__max_depth",2,30),
                "hgb__min_samples_leaf":hp.randint("hgb__min_samples_leaf",1,100),
                "hgb__l2_regularization":hp.uniform("hgb__l2_regularization",0,10**2),                
        },
          "preprocessing":"tree"}
}

#
##
### SETUP PIPELINES
pipelines = {k:{"steps":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
    "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"]),
    "name":k} for k, v in models.items()}

def get_pipeline(name):
    return pipelines[name]

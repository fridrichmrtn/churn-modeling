# Databricks notebook source
# MAGIC %run ./utils

# COMMAND ----------

#
##
### PREPROCESSING

# SETUP PREPROCESSING
from hyperopt import hp
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

scalers = [RobustScaler(), QuantileTransformer(), PowerTransformer()]
samplers = [RandomUnderSampler(), RandomOverSampler()]

preprocessing = {
    "smooth":
        {"steps":
            [("variance_filter", VarianceThreshold()),
            ("data_scaler", PowerTransformer()),
            ("feature_selector", HierarchicalFeatureSelector()),
            ("data_sampler", RandomUnderSampler())],
        "space":
            {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-2, 10**0),
            "data_scaler":hp.choice("data_scaler", scalers),
            "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
            "data_sampler":hp.choice("data_sampler", samplers)}},
    "tree":
         {"steps":
              [("variance_filter", VarianceThreshold()),
              ("data_sampler", RandomUnderSampler())],
         "space":
             {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-2, 10**0),
             "data_sampler":hp.choice("data_sampler", samplers)}}}

#
##
### SETUP MODELS

from hyperopt import hp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem

models = {
    "lr":
          {"model":
               [("lr", LogisticRegression(solver="saga", penalty="elasticnet"))],
           "space":
               {"lr__C":hp.uniform("lr__C",10**-2,10**1),
                "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1)},
           "preprocessing":"smooth"},
    "svm_lin":
          {"model":
               [("svm_lin", CalibratedClassifierCV(LinearSVC(dual=False)))],
           "space":
               {"svm_lin__base_estimator__C":hp.uniform("svm_lin__base_estimator__C",10**-2,10**1),
                "svm_lin__base_estimator__penalty":hp.choice("svm_lin__base_estimator__penalty",["l1","l2"])},
          "preprocessing":"smooth"}}

#
##
### PUT IT TOGETHER

pipelines_spaces = {k:{"pipeline":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
    "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"])}
     for k, v in models.items()}

def get_pipeline(name):
    return pipelines_spaces[name]["pipeline"]
def get_space(name):
    return pipelines_spaces[name]["space"]

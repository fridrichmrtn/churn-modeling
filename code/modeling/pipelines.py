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
            {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-1, 5*10**1),
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

models = {
    "lr":
          {"model":
               [("lr", LogisticRegression(solver="saga", penalty="elasticnet", n_jobs=-1))],
           "space":
               {"lr__C":hp.uniform("lr__C",10**-2,10**1),
                "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1),
                "lr__max_iter":hp.uniform("lr__max_iter",5*10**2,5*10**3)},
           "preprocessing":"smooth"},
    "svm_lin":
          {"model":
               [("svm_lin", CalibratedClassifierCV(LinearSVC(dual=False)))],
           "space":
               {"svm_lin__base_estimator__C":hp.uniform("svm_lin__base_estimator__C",10**-2,10**1),
                "svm_lin__base_estimator__penalty":hp.choice("svm_lin__base_estimator__penalty",["l1","l2"])},
          "preprocessing":"smooth"},
    "svm_rbf":
          {"model":
               [("rbf", Nystroem()),
                ("svm_lin", CalibratedClassifierCV(LinearSVC(dual=False)))],
           "space":
               {"rbf__n_components":hp.randint("rbf__n_components",20,100),
                "svm_lin__base_estimator__C":hp.uniform("svm_lin__base_estimator__C",10**-2,10**1),
                "svm_lin__base_estimator__penalty":hp.choice("svm_lin__base_estimator__penalty",["l1","l2"])},
          "preprocessing":"smooth"},
    "dt":
          {"model":
               [("dt", CalibratedClassifierCV(DecisionTreeClassifier(), n_jobs=-1))],
           "space":
               {"dt__base_estimator__max_depth":hp.randint("dt__base_estimator__max_depth",2,30),
                "dt__base_estimator__min_samples_split":hp.randint("dt__base_estimator__min_samples_split",10**1,2*10**2),
                "dt__base_estimator__min_samples_leaf":hp.randint("dt__base_estimator__min_samples_leaf",1,100),
                #"dt__base_estimator__ccp_alpha":hp.uniform("dt__base_estimator__ccp_alpha",0,1),
                "dt__base_estimator__min_impurity_decrease":hp.uniform("dt__base_estimator__min_impurity_decrease",0,.05),
                "dt__base_estimator__min_weight_fraction_leaf":hp.uniform("dt__base_estimator__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree"},
    "rf":
          {"model":
               [("rf", CalibratedClassifierCV(RandomForestClassifier(n_jobs=-1)))],
           "space":
               {"rf__base_estimator__n_estimators":hp.randint("rf__base_estimator__n_estimators",25,1000),
                "rf__base_estimator__max_features":hp.uniform("rf__base_estimator__max_features",0.1,.7),
                "rf__base_estimator__max_depth":hp.randint("dt__base_estimator__max_depth",2,30),
                "rf__base_estimator__min_samples_split":hp.randint("dt__base_estimator__min_samples_split",10**1,2*10**2),
                "rf__base_estimator__min_samples_leaf":hp.randint("dt__base_estimator__min_samples_leaf",1,100),
                #"dt__base_estimator__ccp_alpha":hp.uniform("dt__base_estimator__ccp_alpha",0,0.05),
                "rf__base_estimator__min_impurity_decrease":hp.uniform("dt__base_estimator__min_impurity_decrease",0,.05),
                "rf__base_estimator__min_weight_fraction_leaf":hp.uniform("dt__base_estimator__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree"},
    
    "hgb":
          {"model":
               [("hgb", CalibratedClassifierCV(HistGradientBoostingClassifier(), n_jobs=-1))],
           "space":
               {
                "hgb__base_estimator__learning_rate":hp.uniform("hgb__base_estimator__learning_rate",0.01,.15),
                "hgb__base_estimator__max_iter":hp.randint("hgb__base_estimator__max_iter",25,1000),
                "hgb__base_estimator__max_leaf_nodes":hp.randint("hgb__base_estimator__max_leaf_nodes",5**2,5**3),
                "hgb__base_estimator__max_depth":hp.randint("hgb__base_estimator__max_depth",2,30),
                "hgb__base_estimator__min_samples_leaf":hp.randint("hgb__base_estimator__min_samples_leaf",1,100),
                "hgb__base_estimator__l2_regularization":hp.uniform("hgb__base_estimator__l2_regularization",0,10**2),                
        },
          "preprocessing":"tree"}        
    
}

#
##
### PUT IT TOGETHER

pipelines_spaces = {k:{"pipeline":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
    "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"])}
     for k, v in models.items()}

def get_pipe(name):
    return pipelines_spaces[name]["pipeline"]
def get_space(name):
    return pipelines_spaces[name]["space"]

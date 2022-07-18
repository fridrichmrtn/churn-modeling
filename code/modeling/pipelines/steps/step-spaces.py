# Databricks notebook source
# MAGIC %run "./transformations"

# COMMAND ----------

# MAGIC %run "./modeling"

# COMMAND ----------

from hyperopt import hp
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow.keras.layers import LeakyReLU
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from lightgbm import LGBMClassifier, LGBMRegressor

# COMMAND ----------

#
##
### SETUP PREPROCESSING

scaling = {
    "regression":
       [RobustScaler(), QuantileTransformer(), PowerTransformer()],
    "classification":
       [RobustScaler(), QuantileTransformer(), PowerTransformer()]}

sampling = {
    "regression":
           ["passthrough"],
      "classification":
           [RandomOverSampler(), RandomUnderSampler(), "passthrough"]}

preprocessing = {
    task:{"smooth":
        {"steps":
            [("variance_filter", VarianceThreshold()),
            ("data_scaler", PowerTransformer()),
            ("feature_selector", HierarchicalFeatureSelector()),
            ("data_sampler", "passthrough")],
        "space":
            {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-3, 5*10**1),
            "data_scaler":hp.choice("data_scaler", scaling[task]),
            "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
            "data_sampler":hp.choice("data_sampler", sampling[task])
            }},
    "tree":
         {"steps":
              [("variance_filter", VarianceThreshold()),
              ("data_sampler", "passthrough")],
         "space":
             {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-3, 5*10**1),
             "data_sampler":hp.choice("data_sampler", sampling[task])}}} for task in scaling.keys()}
        
        
#
##
### MODELS AND SPACES

models = {
    "lr":
          {"model":{
              "classification":
                  [("lr", LogisticRegression(solver="saga",
                      penalty="elasticnet", max_iter=2500))],
              "regression":
                  [("lr", ElasticNetC(max_iter=2500))]},          
           "space":{               
               "lr__C":hp.uniform("lr__C",10**-2,10**1),
               "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1)},
           "preprocessing":"smooth"},
    
    "svm_lin":{
            "model":{
              "classification":
                  [("svm_lin", LinearSVC(dual=False))],
              "regression":
                  [("svm_lin", LinearSVR(loss="squared_epsilon_insensitive", dual=False))]},
           "space":
               {"svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1)},
           "preprocessing":"smooth"},
    
    "svm_rbf":{
            "model":{
              "classification":
                  [("rbf", Nystroem()), ("svm_lin", LinearSVC(dual=False))],
              "regression":
                  [("rbf", Nystroem()), ("svm_lin", LinearSVR(loss="squared_epsilon_insensitive", dual=False))]},
           "space":{
               "rbf__n_components":hp.randint("rbf__n_components",20,100),
               "svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1)},
           "preprocessing":"smooth"},
    
    "mlp":{
           "model":{
              "classification":
                  [("mlp", MLPClassifier())],
              "regression":
                  [("mlp", MLPRegressor())]},
           "space":
               {"mlp__batch_size":hp.randint("mlp__batch_size",2**3,2**6),
                "mlp__epochs":hp.randint("mlp__epochs",10**2,10**3),
                "mlp__layers":hp.randint("mlp__layers",1,10),
                "mlp__units":hp.randint("mlp__units",2**2,2**8),
                "mlp__activation":hp.choice("mlp__activation",
                    ["tanh", "sigmoid", "relu", LeakyReLU()]),
                "mlp__optimizer__learning_rate":hp.uniform("mlp__optimizer__learning_rate", 10**-5,10**-3),
                "mlp__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])},
           "preprocessing":"smooth"},
    
    "dt":{
           "model":{
               "classification":
                   [("dt", DecisionTreeClassifier())],
               "regression":
                   [("dt", DecisionTreeRegressor())]},
           "space":
               {"dt__max_depth":hp.randint("dt__max_depth",2,30),
                "dt__min_samples_split":hp.randint("dt__min_samples_split",10**1,2*10**2),
                "dt__min_samples_leaf":hp.randint("dt__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__base_estimator__ccp_alpha",0,1),
                "dt__min_impurity_decrease":hp.uniform("dt__min_impurity_decrease",0,.05),
                "dt__min_weight_fraction_leaf":hp.uniform("dt__min_weight_fraction_leaf",0,.05)},
           "preprocessing":"tree"},
    
    "rf":{
           "model":{
               "classification":
                   [("rf", RandomForestClassifier())],
               "regression":
                   [("rf", RandomForestRegressor())]},
           "space":
               {"rf__n_estimators":hp.randint("rf__n_estimators",25,1000),
                "rf__max_features":hp.uniform("rf__max_features",0.1,.7),
                "rf__max_depth":hp.randint("rf__max_depth",2,30),
                "rf__min_samples_split":hp.randint("rf__min_samples_split",10**1,2*10**2),
                "rf__min_samples_leaf":hp.randint("rf__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__ccp_alpha",0,0.05),
                "rf__min_impurity_decrease":hp.uniform("rf__min_impurity_decrease",0,.05),
                "rf__min_weight_fraction_leaf":hp.uniform("rf__min_weight_fraction_leaf",0,.05)},
           "preprocessing":"tree"},
    
   "hgb":{ # NOTE: ADD SUBSAMPLING?
          "model":{
              "classification":
                  [("hgb", LGBMClassifier())],
              "regression":
                  [("hgb", LGBMRegressor())]},
          "space":
              {"hgb__learning_rate":hp.uniform("hgb__learning_rate",0.01,.5),
               "hgb__n_estimators":hp.randint("hgb__n_estimators",25,1000),
               "hgb__num_leaves":hp.randint("hgb__num_leaves",5**2,5**4),
               "hgb__max_depth":hp.randint("hgb__max_depth",2,30),
               "hgb__min_child_samples":hp.randint("hgb__min_child_samples",2,100),
               "hgb__reg_lambda":hp.uniform("hgb__reg_lambda",10**-2,10**2)},
          "preprocessing":"tree"}
}
    
calibration = {
    "classification": CalibratedClassifierCV,
    "regression": CalibratedPassthrough}
    
#
##
### SETUP PIPELINES

def construct_pipelines(preprocessing=preprocessing, models=models):
    pipelines = {}
    tasks = ["classification","regression"]
    for task, task_short in zip(tasks,["class","reg"]):
        temp_preprocessing = preprocessing[task]
        temp_calibration = calibration[task]
        for name, model_space in models.items():
            pipe = {}
            pipe["name"] = name+"_"+task_short
            pipe["task"] = task
            pipe["steps"] = Pipeline(temp_preprocessing[model_space["preprocessing"]]["steps"]+model_space["model"][task])
            pipe["space"] = dict(temp_preprocessing[model_space["preprocessing"]]["space"],**model_space["space"])
            pipe["calibration"] = temp_calibration
            pipelines[pipe["name"]] = pipe
    return pipelines

# pipelines = {k:{"steps":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
#     "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"]),
#     "calibration":calibration[v["type"]],
#    "type":v["type"],
#     "name":k} for k, v in models.items()}

# def get_pipeline(name):
#     return pipelines[name]

# def get_pipelines():
#     return pipelines

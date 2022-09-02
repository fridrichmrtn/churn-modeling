# Databricks notebook source
# MAGIC %run "./transformations"

# COMMAND ----------

# MAGIC %run "./modeling"

# COMMAND ----------

from hyperopt import hp
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tensorflow.keras.layers import LeakyReLU
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor, BaggingRegressor)
from lightgbm import LGBMClassifier, LGBMRegressor

# COMMAND ----------

#
##
### SETUP PREPROCESSING

scaling = {
    "regression":
       [RobustScaler(), QuantileTransformer(),PowerTransformer()],
    "classification":
       [RobustScaler(), QuantileTransformer(),PowerTransformer()]}

sampling = {
    "regression":
           ["passthrough"],
     "classification":
           [RandomOverSampler(), RandomUnderSampler(), "passthrough"]}

preprocessing = {
    task:{
        "smooth":
            {"steps":
                [("stability_scaler", MinMaxScaler()),
                ("data_scaler", PowerTransformer()),
                ("variance_filter", VarianceThreshold()),
                ("feature_selector", HierarchicalFeatureSelector()),
                ("data_sampler", "passthrough")],
            "space":
                {"data_scaler":hp.choice("data_scaler", scaling[task]),
                "variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-5, 10**-1),
                "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
                "data_sampler":hp.choice("data_sampler", sampling[task])
                }},
        "tree":
             {"steps":
                  [("stability_scaler", MinMaxScaler()),
                  ("data_scaler", PowerTransformer()),
                  ("variance_filter", VarianceThreshold()),
                  ("data_sampler", "passthrough")],
             "space":
                 {"data_scaler":hp.choice("data_scaler", scaling[task]),
                 "variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-5, 10**-1),
                 "data_sampler":hp.choice("data_sampler", sampling[task])}}} for task in scaling.keys()}

#
##
### TARGET TRANSFORMATIONS

target_trans = [RobustScaler(), QuantileTransformer(), PowerTransformer()]
     
#
##
### MODELS AND SPACES

models = {
#    "lr":
#           {"model":{
#               "classification":
#                   [("lr", LogisticRegression(solver="saga",
#                       penalty="elasticnet", max_iter=2500))],
#               "regression":
#                   [("lr", TransformedTargetRegressor(regressor=ElasticNetC(max_iter=2500),
#                       transformer=PowerTransformer()))]},          
#            "space":
#                {"classification":
#                    {"lr__C":hp.uniform("lr__C",10**-2,10**1),
#                    "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1)},
#                 "regression":
#                    {"lr__regressor__C":hp.uniform("lr__regression__C",10**-2,10**1),
#                    "lr__regressor__l1_ratio":hp.uniform("lr__regression__l1_ratio",0,1),
#                    "lr__transformer":hp.choice("lr__transformer",target_trans)}},
#            "preprocessing":"smooth"},
    
#     "svm_lin":{
#             "model":{
#               "classification":
#                   [("svm_lin", LinearSVC(dual=False))],
#               "regression":
#                   [("svm_lin", TransformedTargetRegressor(
#                           regressor=LinearSVR(loss="squared_epsilon_insensitive", dual=False),
#                       transformer=PowerTransformer()))]},
#            "space":
#                {"classification":
#                    {"svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1)},
#                "regression":
#                    {"svm_lin__regressor__C":hp.uniform("svm_lin__regressor__C",10**-2,10**1),
#                    "svm_lin__transformer":hp.choice("svm_lin__transformer",target_trans)}},
#            "preprocessing":"smooth"},
    
#     "svm_rbf":{
#             "model":{
#               "classification":
#                   [("rbf", Nystroem()), ("svm_lin", LinearSVC(dual=False))],
#               "regression":
#                   [("rbf", Nystroem()), ("svm_lin", TransformedTargetRegressor(
#                           regressor=LinearSVR(loss="squared_epsilon_insensitive", dual=False),
#                       transformer=PowerTransformer()))]},
#            "space":
#                {"classification":
#                    {"rbf__n_components":hp.randint("rbf__n_components",20,100),
#                    "svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**2)},
#                 "regression":
#                    {"rbf__n_components":hp.randint("rbf__n_components",20,100),
#                    "svm_lin__regressor__C":hp.uniform("svm_lin__C",10**-2,10**2),
#                    "svm_lin__transformer":hp.choice("svm_lin__transformer",target_trans)}},                
#            "preprocessing":"smooth"},
    
    "mlp":{
           "model":{
              "classification":
                  [("mlp", MLPClassifier())],
              "regression":
                  [("mlp", TransformedTargetRegressor(regressor=MLPRegressor(),
                      transformer=PowerTransformer()))]},
           "space":
               {"classification":
                   {"mlp__batch_size":hp.randint("mlp__batch_size",2**3,2**6),
                    "mlp__epochs":hp.randint("mlp__epochs",5*10, 5*10**2),
                    "mlp__layers":hp.randint("mlp__layers",1,5),
                    "mlp__units":hp.randint("mlp__units",2**2,2**7),
                    "mlp__activation":hp.choice("mlp__activation", ["elu", LeakyReLU()]),
                    "mlp__optimizer__learning_rate":hp.uniform("mlp__optimizer__learning_rate", 10**-5,10**-1),
                    "mlp__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])},
                "regression":
                   {"mlp__regressor__batch_size":hp.randint("mlp__regressor__batch_size",2**3,2**6),
                    "mlp__regressor__epochs":hp.randint("mlp__regressor__epochs",5*10, 5*10**2),
                    "mlp__regressor__layers":hp.randint("mlp__regressor__layers",1,5),
                    "mlp__regressor__units":hp.randint("mlp__regressor__units",2**2,2**7),
                    "mlp__regressor__activation":hp.choice("mlp__regressor__activation", ["elu", LeakyReLU()]),
                    "mlp__regressor__optimizer__learning_rate":hp.uniform("mlp__regressor__optimizer__learning_rate", 10**-5,10**-1),
                    "mlp__regressor__optimizer":hp.choice("mlp__regressor__optimizer",["sgd", "adam", "rmsprop"]),
                    "mlp__transformer":hp.choice("mlp__transformer",target_trans)}},
           "preprocessing":"smooth"},
    
#     "dt":{
#            "model":{
#                "classification":
#                    [("dt", DecisionTreeClassifier())],
#                "regression":
#                    [("dt", TransformedTargetRegressor(regressor=DecisionTreeRegressor(),
#                        transformer=PowerTransformer()))]},
#            "space":
#                {"classification":
#                    {"dt__max_depth":hp.randint("dt__max_depth",2,30),
#                     "dt__min_samples_split":hp.randint("dt__min_samples_split",10**1,2*10**2),
#                     "dt__min_samples_leaf":hp.randint("dt__min_samples_leaf",2,100),
#                     "dt__min_impurity_decrease":hp.uniform("dt__min_impurity_decrease",0,.1),
#                     "dt__min_weight_fraction_leaf":hp.uniform("dt__min_weight_fraction_leaf",0,.1)},
#                 "regression":
#                    {"dt__regressor__max_depth":hp.randint("dt__regressor__max_depth",2,30),
#                     "dt__regressor__min_samples_split":hp.randint("dt__regressor__min_samples_split",10**1,2*10**2),
#                     "dt__regressor__min_samples_leaf":hp.randint("dt__regressor__min_samples_leaf",2,100),
#                     "dt__regressor__min_impurity_decrease":hp.uniform("dt__regressor__min_impurity_decrease",0,.1),
#                     "dt__regressor__min_weight_fraction_leaf":hp.uniform("dt__regressor__min_weight_fraction_leaf",0,.1),
#                     "dt__transformer":hp.choice("dt__transformer",target_trans)}},
#            "preprocessing":"tree"},
    
#     "rf":{
#            "model":{
#                "classification":
#                    [("rf", RandomForestClassifier(n_jobs=-1))],
#                "regression":
#                    [("rf", TransformedTargetRegressor(regressor=RandomForestRegressor(n_jobs=-1),
#                        transformer=PowerTransformer()))]},
#            "space":
#                {"classification":
#                    {"rf__n_estimators":hp.randint("rf__n_estimators",50,1000),
#                    "rf__max_features":hp.uniform("rf__max_features",0.2,.8),
#                    "rf__max_samples":hp.uniform("rf__max_samples",0.2,.8),
#                    "rf__max_depth":hp.randint("rf__max_depth",2,30),
#                    "rf__min_samples_split":hp.randint("rf__min_samples_split",10**1,2*10**2),
#                    "rf__min_samples_leaf":hp.randint("rf__min_samples_leaf",2,100),
#                    "rf__min_impurity_decrease":hp.uniform("rf__min_impurity_decrease",0,.1),
#                    "rf__min_weight_fraction_leaf":hp.uniform("rf__min_weight_fraction_leaf",0,.1)},
#                "regression":
#                    {"rf__regressor__n_estimators":hp.randint("rf__regressor__n_estimators",50,1000),
#                    "rf__regressor__max_features":hp.uniform("rf__regressor__max_features",0.2,.8),
#                    "rf__regressor__max_samples":hp.uniform("rf__regressor__max_samples",0.2,.8),
#                    "rf__regressor__max_depth":hp.randint("rf__regressor__max_depth",2,30),
#                    "rf__regressor__min_samples_split":hp.randint("rf__regressor__min_samples_split",10**1,2*10**2),
#                    "rf__regressor__min_samples_leaf":hp.randint("rf__regressor__min_samples_leaf",2,100),
#                    "rf__regressor__min_impurity_decrease":hp.uniform("rf__regressor__min_impurity_decrease",0,.1),
#                    "rf__regressor__min_weight_fraction_leaf":hp.uniform("rf__regressor__min_weight_fraction_leaf",0,.1),
#                    "rf__transformer":hp.choice("rf__transformer",target_trans)}},
#            "preprocessing":"tree"},
    
#    "gbm":{
#           "model":{
#               "classification":
#                   [("gbm", LGBMClassifier(n_jobs=-1))],
#               "regression":
#                   [("gbm", TransformedTargetRegressor(regressor=LGBMRegressor(n_jobs=-1),
#                       transformer=PowerTransformer()))]},
#           "space":
#               {"classification":
#                   {"gbm__learning_rate":hp.uniform("gbm__learning_rate",0.005,.2),
#                   "gbm__n_estimators":hp.randint("gbm__n_estimators",50,1000),
#                   "gbm__num_leaves":hp.randint("gbm__num_leaves",5**2,5**4),
#                   "gbm__max_depth":hp.randint("gbm__max_depth",2,30),
#                   "gbm__min_child_samples":hp.randint("gbm__min_child_samples",2,100),
#                   "gbm__subsample_freq":hp.randint("gbm__subsample_freq",1,5),
#                   "gbm__subsample":hp.uniform("gbm__subsample",0.2,0.8),
#                   "gbm__colsample_bytree":hp.uniform("gbm__colsample_bytree",0.2,0.8),
#                   "gbm__reg_alpha":hp.uniform("gbm__reg_alpha",10**-2,10**2), 
#                   "gbm__reg_lambda":hp.uniform("gbm__reg_lambda",10**-2,10**2)},
#                "regression":
#                    {"gbm__regressor__learning_rate":hp.uniform("gbm__regressor__learning_rate",0.005,.2),
#                    "gbm__regressor__n_estimators":hp.randint("gbm__regressor__n_estimators",50,1000),
#                    "gbm__regressor__num_leaves":hp.randint("gbm__regressor__num_leaves",5**2,5**4),
#                    "gbm__regressor__max_depth":hp.randint("gbm__regressor__max_depth",2,30),
#                    "gbm__regressor__min_child_samples":hp.randint("gbm__regressor__min_child_samples",2,100),
#                    "gbm__regressor__subsample_freq":hp.randint("gbm__regressor__subsample_freq",1,5),
#                    "gbm__regressor__subsample":hp.uniform("gbm__regressor__subsample",0.2,0.8),
#                    "gbm__regressor__colsample_bytree":hp.uniform("gbm__regressor__colsample_bytree",0.2,0.8),
#                    "gbm__regressor__reg_alpha":hp.uniform("gbm__regressor__reg_alpha",10**-2,10**2),
#                    "gbm__regressor__reg_lambda":hp.uniform("gbm__regressor__reg_lambda",10**-2,10**2),
#                    "gbm__transformer":hp.choice("gbm__transformer",target_trans)}},
#            "preprocessing":"tree"}
}
    
calibration = {
    "classification": CalibratedClassifierCV,
    "regression": CalibratedRegression}
    
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
            pipe["space"] = dict(temp_preprocessing[model_space["preprocessing"]]["space"], **model_space["space"][task])
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

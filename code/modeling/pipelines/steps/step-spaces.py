# Databricks notebook source
# MAGIC %run "./transformations"

# COMMAND ----------

# MAGIC %run "./modeling"

# COMMAND ----------

from hyperopt import hp
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import LeakyReLU
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# COMMAND ----------

#
##
### SETUP PREPROCESSING

scaling = [RobustScaler(), QuantileTransformer(), PowerTransformer()]
sampling = [RandomOverSampler(), RandomUnderSampler(),"passthrough"]

preprocessing = {
    "smooth":
        {"steps":
            [("variance_filter", VarianceThreshold()),
            ("data_scaler", PowerTransformer()),
            ("feature_selector", HierarchicalFeatureSelector()),
            ("data_sampler", "passthrough")],
        "space":
            {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-1, 5*10**1),
            "data_scaler":hp.choice("data_scaler", scaling),
            "feature_selector__n_features":hp.randint("feature_selector__n_features", 5, 100),
            "data_sampler":hp.choice("data_sampler", sampling)
            }},
    "tree":
         {"steps":
              [("variance_filter", VarianceThreshold()),
              ("data_sampler", "passthrough")
              ],
         "space":
             {"variance_filter__threshold":hp.uniform("variance_filter__threshold", 10**-2, 5*10**1),
             "data_sampler":hp.choice("data_sampler", sampling)
             }}}

calibration = {
    "standard": CalibratedClassifierCV,
    "multi-output": MultiOutputCalibrationCV}

#
##
### SETUP MODELS

models = {
    "lr":
          {"model":
               [("lr", LogisticRegression(solver="saga",
                   penalty="elasticnet", max_iter=200))],          
           "space":
               {"lr__C":hp.uniform("lr__C",10**-2,10**1),
                "lr__l1_ratio":hp.uniform("lr__l1_ratio",0,1)},
           "preprocessing":"smooth",
          "type":"standard"},
    
    "svm_lin":
          {"model":
               [("svm_lin", LinearSVC(dual=False))],
           "space":
               {"svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1),
                "svm_lin__penalty":hp.choice("svm_lin__penalty",["l1","l2"])},
          "preprocessing":"smooth",
          "type":"standard"},
    
    "svm_rbf":
          {"model":
               [("rbf", Nystroem()),
                ("svm_lin", LinearSVC(dual=False))],
           "space":
               {"rbf__n_components":hp.randint("rbf__n_components",20,100),
                "svm_lin__C":hp.uniform("svm_lin__C",10**-2,10**1),
                "svm_lin__penalty":hp.choice("svm_lin__penalty",["l1","l2"])},
          "preprocessing":"smooth",
          "type":"standard"},
    
    "mlp":
          {"model":
               [("mlp", MLPClassifier())],
           "space":
               {"mlp__batch_size":hp.randint("mlp__batch_size",2**3,2**6),
                "mlp__epochs":hp.randint("mlp__epochs",10**2,10**3),
                "mlp__layers":hp.randint("mlp__layers",1,10),
                "mlp__units":hp.randint("mlp__units",2**2,2**8),
                "mlp__activation":hp.choice("mlp__activation",
                    ["tanh", "sigmoid", "relu", keras.layers.LeakyReLU()]),
                "mlp__optimizer__learning_rate":hp.uniform("mlp__optimizer__learning_rate", 10**-5,10**-3),
                "mlp__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])},
          "preprocessing":"smooth",
          "type":"standard"},
    
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
          "preprocessing":"tree",
          "type":"standard"},
    
    "rf":
          {"model":
               [("rf", RandomForestClassifier())],
           "space":
               {"rf__n_estimators":hp.randint("rf__n_estimators",25,500),
                "rf__max_features":hp.uniform("rf__max_features",0.1,.7),
                "rf__max_depth":hp.randint("rf__max_depth",2,30),
                "rf__min_samples_split":hp.randint("rf__min_samples_split",10**1,2*10**2),
                "rf__min_samples_leaf":hp.randint("rf__min_samples_leaf",1,100),
                #"dt__ccp_alpha":hp.uniform("dt__ccp_alpha",0,0.05),
                "rf__min_impurity_decrease":hp.uniform("rf__min_impurity_decrease",0,.05),
                "rf__min_weight_fraction_leaf":hp.uniform("rf__min_weight_fraction_leaf",0,.05),
               },
          "preprocessing":"tree",
          "type":"standard"},
    
   "hgb":
         {"model":
              [("hgb", HistGradientBoostingClassifier())],
          "space":
              {
               "hgb__learning_rate":hp.uniform("hgb__learning_rate",0.01,.15),
               "hgb__max_iter":hp.randint("hgb__max_iter",25,500),
               "hgb__max_leaf_nodes":hp.randint("hgb__max_leaf_nodes",5**2,5**3),
               "hgb__max_depth":hp.randint("hgb__max_depth",2,30),
               "hgb__min_samples_leaf":hp.randint("hgb__min_samples_leaf",1,100),
               "hgb__l2_regularization":hp.uniform("hgb__l2_regularization",0,10**2),                
       },
         "preprocessing":"tree",
         "type":"standard"},
    
#     "combinet":
#          {"model":
#               [("combinet", CombiNet())],
#            "space":
#                {
#                 "combinet__se_layers":hp.randint("combinet__se_layers",1,2),
#                 "combinet__se_units":hp.randint("combinet__se_units",2**5,2**9),
#                 "combinet__re_layers":hp.randint("combinet__re_layers",1,2),
#                 "combinet__re_units":hp.randint("combinet__re_units",2**4,2**8),                
#                 "combinet__ce_layers":hp.randint("combinet__ce_layers",1,4),
#                 "combinet__ce_units":hp.randint("combinet__ce_units",2**4,2**8), 
#                 "combinet__cc_units":hp.randint("combinet__cc_units",2**4,2**6), 
#                 "combinet__activation":hp.choice("combinet__activation",
#                     ["selu", LeakyReLU()]),
#                 "combinet__batch_size":hp.randint("combinet__batch_size",2**3,2**6),
#                 "combinet__epochs":hp.randint("combinet__epochs",10**2,10**3),   
#                 "combinet__optimizer__learning_rate":hp.uniform("combinet__optimizer__learning_rate", 10**-5,10**-3),
#                 "combinet__optimizer":hp.choice("mlp__optimizer",["sgd", "adam", "rmsprop"])
#                },
#          "preprocessing":"smooth",
#          "type":"multi-output"},
    
}

#
##
### SETUP PIPELINES

pipelines = {k:{"steps":Pipeline(preprocessing[v["preprocessing"]]["steps"]+v["model"]),
    "space":dict(preprocessing[v["preprocessing"]]["space"],**v["space"]),
    "calibration":calibration[v["type"]],
   "type":v["type"],
    "name":k} for k, v in models.items()}

def get_pipeline(name):
    return pipelines[name]

def get_pipelines():
    return pipelines

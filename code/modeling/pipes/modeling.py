# Databricks notebook source
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

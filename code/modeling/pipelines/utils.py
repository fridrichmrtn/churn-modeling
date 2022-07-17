# Databricks notebook source
import copy
import pandas as pd
import mlflow

# COMMAND ----------

#
##
### PIPELINE STEPS

def reduce_y(func):
    def actual_reduce(self, X, y,  *args, **kwargs):
        if self.y_ is None:
            self.y_ = copy.copy(y)
        if len(y.shape)>1:
            print("shit")
            y = y.iloc[:,0]
        return func(self, X, y, *args, **kwargs)
    return actual_reduce

def expand_y(func):
    def actual_expand(self, X, y, *args, **kwargs):
        #if len(self.y_.shape)>1:
        y = copy.copy(self.y_)
        return func(self, X, y, *args, **kwargs)
    return actual_expand

#
##
### DATA

def optimize_numeric_dtypes(df):
    float_cols = df.select_dtypes("float").columns
    int_cols = df.select_dtypes("integer").columns
    df[float_cols] = df[float_cols].\
        apply(pd.to_numeric, downcast="float")
    df[int_cols] = df[int_cols].\
        apply(pd.to_numeric, downcast="integer")
    return df

def get_Xy(data, pipe):
    X = data["raw"].loc[:, data["columns"][pipe["task"]]["X"]]                   
    y = data["raw"].loc[:, data["columns"][pipe["task"]]["y"]]                  
    return (X, y)

#
##
### MLFLOW

def get_exp_id(exp_path):
    try:
        exp_id = mlflow.get_experiment_by_name(exp_path).experiment_id
    except:
        exp_id = mlflow.create_experiment(exp_path)
    return exp_id

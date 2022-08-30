# Databricks notebook source
# just dump the model
from mlflow.sklearn import load_model, save_model
full_model = load_model("models:/retailrocket_gbm_reg_0/None")
save_model(full_model,"/dbfs/mnt/retailrocket/pipelines/gbm_reg")

full_model = load_model("models:/retailrocket_lr_class_0/None")
save_model(full_model,"/dbfs/mnt/retailrocket/pipelines/lr_class")

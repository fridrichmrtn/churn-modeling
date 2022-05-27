# Databricks notebook source
# CREATE TARGET DB 
#spark.sql("CREATE SCHEMA IF NOT EXISTS churndb LOCATION \"dbfs:/mnt/churndb\"")

# COMMAND ----------

# MAGIC %run "./load-transform/workhorse"

# COMMAND ----------

# MAGIC %run "./customer-model/workhorse"

# COMMAND ----------

#
##
### REES46

dataset_name = "rees46"

# LOAD TRANSFORM
events = construct_events(dataset_name)
save_events(events, dataset_name)

# CUSTOMER MODEL
split_save_customer_model(dataset_name, week_steps=11,
    week_target=4, overwrite=True)


# COMMAND ----------

# TODO


# EXPLORATION
# individual features
# multi features

# MODELING

# MODELING
# WRITE IT ON SMALL DATASET IN SKLEARN

# HYPEROPT

# PIPELINE
# IMPUTATION - DONE IN THE MODELING PHASE  
# FILTER - NZV, MULTICORR, TARGET CORR 
# UNDER/OVER SAMPLING
# SCALING - QUANTILE TRANSFORMER
# FIT

# EVALUATE
# REGRESSION - MAE, MSE
# CLASSIFICATION - ACC, PRE, REC, F1, AUC
# COST-BENEFIT METRICS - H-measure, MP, EMP, R-EMP

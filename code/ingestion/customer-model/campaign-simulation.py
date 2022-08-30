# Databricks notebook source
import numpy as np
import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.window import Window

simulation_config = {
    "retailrocket":{
        "gamma":{"alpha":2.042, "beta":202.116},
        "delta":7000, 
        "psi":{"alpha":6.12, "beta":3.15},
        "n_iter":1000},
    "rees46":{"gamma":{"alpha":2.042, "beta":202.116},
        "delta":20, 
        "psi":{"alpha":6.12, "beta":3.15},
        "n_iter":1000}}

# COMMAND ----------



def flush_dataframe(dataframe, dataset_name, table_name, overwrite=True):
    mode = "append"
    if overwrite:
        spark.sql("DROP TABLE IF EXISTS "\
            + f"churndb.{dataset_name}_{table_name}")   
        mode ="overwrite"
    dataframe.write.format("delta").mode(mode)\
        .saveAsTable(f"churndb.{dataset_name}_{table_name}")
    return None

#
##
###  CAMPAIGN SIMULATION

def get_campaign_params(customer_model, dataset_name):
    config = simulation_config[dataset_name]
    gamma = config["gamma"]
    delta = config["delta"]
    psi = config["psi"]
    n_iter=config["n_iter"]
    user_id = customer_model.select("user_id").distinct().toPandas().user_id
    n_users = len(user_id)
    # NOTE: consider generating randoms at once
    params = []
    for i in range(n_iter):
        np.random.seed(i+1)
        temp = pd.DataFrame.from_dict({
            "user_id":user_id,
            "seed":i,
            "gamma":np.random.beta(gamma["alpha"], gamma["beta"], size=n_users),
            "psi":np.random.beta(psi["alpha"], psi["beta"], size=n_users),
            "delta":delta})
        params.append(temp)
    return spark.createDataFrame(pd.concat(params))

def add_campaign_features(customer_model, campaign_params):
    # lagged customer value
    w = Window.partitionBy("user_id").orderBy(f.col("time_step").asc())
    customer_model = customer_model.withColumn("target_customer_value_lag1",
            f.lead("target_customer_value").over(w))\
        .fillna(0, subset=["target_customer_value_lag1"])
    # target actual profit
    campaign_profit = customer_model.join(campaign_params, on=["user_id"],
        how="inner").groupby("row_id").agg(f.mean(
            f.col("target_event")*f.col("gamma")
                *(f.col("target_customer_value")-f.col("delta"))
            - (1-f.col("target_event"))*f.col("psi")
                *f.col("delta")).alias("target_actual_profit"))
    return customer_model.join(campaign_profit, on=["row_id"], how="inner")
  
def save_campaign_simulation(dataset_name, overwrite=True):
    customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
    campaign_params = _get_campaign_params(customer_model, dataset_name)
    flush_dataframe(campaign_params,
        dataset_name, "campaign_params", True)
    customer_model = _add_campaign_features(customer_model, campaign_params)
    flush_dataframe(customer_model,
        dataset_name, "customer_model", True)
    return customer_model

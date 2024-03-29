# Databricks notebook source
# MAGIC %run "./base-model"

# COMMAND ----------

# MAGIC %run "./preference-model"

# COMMAND ----------

# MAGIC %run "./campaign-simulation"

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.window import Window
from dateutil.relativedelta import relativedelta
import re
import mlflow

# COMMAND ----------

### NOTE: ADD DOCSTRINGS
### NOTE: flust_dataframe to utils?

def flush_dataframe(dataframe, dataset_name, table_name,
    overwrite=True):
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
### TARGET AND FEATURES

def _get_target(events, split_time, week_target):
    n_weeks = events.agg((f.datediff(f.max("event_time"),
            f.min("event_time"))/7).alias("dr"))\
        .collect()[0]["dr"]
    
    target = events.select("user_id").distinct()\
        .join(
            (events.where(f.col("event_time")>split_time)
                .groupBy("user_id").agg(
                     f.when(f.sum("purchase")>0,0).otherwise(1).alias("target_event"),
                     f.sum(f.col("revenue")).alias("target_revenue"))),
            on=["user_id"], how="left")\
            .fillna(1, subset=["target_event"])\
            .fillna(0, subset=["target_revenue"])\
        .join(events.groupBy("user_id").agg(
                (week_target*f.sum("profit")/n_weeks).alias("target_customer_value")),
            on=["user_id"], how="left")\
            .fillna(0, subset=["target_customer_value"])
    return target

def _get_feature_events(events, split_time):
    return events.where(f.col("event_time")<=split_time)
  
#
##
### CUSTOMER MODEL

def _impute_customer_model(customer_model):   
    reg_pat = ["_lag[0-9]+$", "_ma[0-9]+$", "_stddev$"]
    zero_cols = [c for c in customer_model.columns for fc in reg_pat\
        if re.search(fc, c) is not None]
    customer_model = customer_model.fillna(0, subset=zero_cols)
    # MAX IMP
    max_cols = [c for c in customer_model.columns for fc in\
        ["^inter_", "^time_to_", "_cv$"] if re.search(fc, c) is not None]
    max_expr = [f.max(c).alias(c) for c in max_cols]
    max_vals = customer_model.agg(*max_expr).toPandas().transpose().to_dict()[0]
    customer_model = customer_model.fillna(max_vals)
    interactions = ["view", "cart", "purchase"]
    operations = ["sum", "mean", "min", "max"]
    zero_cols = [f"{i}_revenue_{o}" for o in operations for i in interactions]
    customer_model = customer_model.fillna(0, subset=zero_cols)
    
    return customer_model

def _construct_customer_model(dataset_name, events, split_time,
    time_step, week_target):
    cust_target = _get_target(events, split_time, week_target)
    cust_events = _get_feature_events(events, split_time).persist()    

    # BASE MODEL
    cust_base = get_base_model(cust_events, week_target)
    # PREFERENCE MODELS
    cust_pref = get_pref_model(cust_events, dataset_name)
    # ALL TOGETHER
    customer_model = (cust_base.join(cust_pref, on=["user_id"])
        .join(cust_target, on=["user_id"]))
    #customer_model = cust_base.join(cust_target, on=["user_id"])
    # IMPUTATION
    customer_model = _impute_customer_model(customer_model)
    customer_model = customer_model.withColumn("time_step", f.lit(time_step))
    return customer_model
    
#
##
###  CUSTOMER MODEL
        
def construct_customer_model(dataset_name, time_steps=3,
    week_target=4):
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"
    
    # WEEK STEPS
    events = spark.read.format("delta").load(data_path+"events").repartition(16)
    max_date = events.agg(f.to_date(f.next_day(f.max("event_time"),"Sun")
        -f.expr("INTERVAL 7 DAYS")).alias("mdt")).collect()[0]["mdt"]
    for time_step in range(time_steps):
        temp_max_date = max_date+relativedelta(days=-(7*time_step*week_target))
        temp_split_date = temp_max_date+relativedelta(days=-(7*week_target))
        temp_customer_model = _construct_customer_model(dataset_name,
                events.where(f.col("event_time")<=temp_max_date),
                    temp_split_date, time_step, week_target)
        if "customer_model" not in locals():
            customer_model = temp_customer_model
        else:
            customer_model = customer_model.union(temp_customer_model)
    # ROW ID
    customer_model = customer_model.withColumn("row_id",f.row_number()\
        .over(Window.orderBy(f.monotonically_increasing_id()))).persist()
    
    # CAMPAIGN SIMULATION
    campaign_params = get_campaign_params(customer_model, dataset_name)
    flush_dataframe(campaign_params,dataset_name,"campaign_params")
    customer_model = add_campaign_features(customer_model, campaign_params)
    
    return customer_model
    
def save_customer_model(customer_model, dataset_name, overwrite=True):
    # FLUSH
    flush_dataframe(customer_model,
        dataset_name, "customer_model", overwrite)
    return None

# COMMAND ----------

# import pyspark.sql.functions as f
# dataset_name = "retailrocket"
# data_path = f"dbfs:/mnt/{dataset_name}/delta/"
# events = spark.read.format("delta").load(data_path+"events").toPandas()
# #.where(f.col("user_id")==23076).toPandas()
# # check the price distribution
# events.sort_values("event_time")

# COMMAND ----------

# dataset_name = "rees46"
# customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
# customer_model.select("target_customer_value").describe().show()
# SET PARAMS IN THE SIMULATION AND RELOAD

# COMMAND ----------

# customer_model = customer_model.drop(["target_customer_value_lag1","target_actual_profit"])
# campaign_params = get_campaign_params(customer_model, dataset_name)
# flush_dataframe(campaign_params,dataset_name,"campaign_params")
# customer_model = add_campaign_features(customer_model, campaign_params)
# save_customer_model(customer_model, dataset_name)

# Databricks notebook source
# MAGIC %run "./base-model"

# COMMAND ----------

# MAGIC %run "./preference-model"

# COMMAND ----------

### NOTE: ADD DOCSTRINGS

#
##
### TARGET AND FEATURES

def _get_target(events, split_time, week_target):
    import pyspark.sql.functions as f
    
    n_weeks = events.agg((f.datediff(f.max("event_time"),
            f.min("event_time"))/7).alias("dr"))\
        .collect()[0]["dr"]
    target = events.select("user_id").distinct()\
        .join((events.where(f.col("event_time")>split_time)
            .groupBy("user_id").agg(
                f.when(f.sum("purchase")>0,0).otherwise(1).alias("target_event"),
                f.sum(f.col("revenue")).alias("target_revenue"))),
            on=["user_id"], how="left")\
        .join(events.groupBy("user_id").agg(
                (week_target*f.sum("profit")/n_weeks).alias("target_cap")),
            on=["user_id"], how="left").na.fill(0)
    return target

def _get_feature_events(events, split_time):
    import pyspark.sql.functions as f
    return events.where(f.col("event_time")<=split_time)
  
#
##
### CUSTOMER MODEL

def _impute_customer_model(customer_model):
    import pyspark.sql.functions as f
    import re
    
    # zero imputation
    reg_pat = ["_lag[0-9]+$", "_ma[0-9]+$", "_stddev$"]
    zero_cols = [c for c in customer_model.columns for fc in reg_pat\
        if re.search(fc, c) is not None]
    customer_model = customer_model.fillna(0, subset=zero_cols)

    # max imputation
    max_cols = [c for c in customer_model.columns for fc in\
        ["^inter_", "^time_to_", "_variation$"] if re.search(fc, c) is not None]
    max_expr = [f.max(c).alias(c) for c in max_cols]
    max_vals = customer_model.agg(*max_expr).toPandas().transpose().to_dict()[0]
    customer_model = customer_model.fillna(max_vals)
    
    # revenue imp
    interactions = ["view", "cart", "purchase"]
    operations = ["sum","mean", "min", "max"]
    zero_cols = [f"{i}_revenue_{o}" for o in operations for i in interactions]
    customer_model = customer_model.fillna(0, subset=zero_cols)
    
    return customer_model

def _construct_customer_model(dataset_name, events, split_time,
    week_step, week_target):
    import pyspark.sql.functions as f
    import mlflow    
    cust_target = _get_target(events, split_time, week_target)
    cust_events = _get_feature_events(events, split_time).persist()    

    # BASE MODEL
    cust_base = get_base_model(cust_events, week_target)
    # PREFERENCE MODELS
    cust_pref = get_pref_model(cust_events, dataset_name)
    # ALL TOGETHER
    customer_model = (cust_base.join(cust_pref, on=["user_id"])
        .join(cust_target, on=["user_id"]))        
    # IMPUTATION
    customer_model = _impute_customer_model(customer_model)
    customer_model = customer_model.withColumn("week_step", f.lit(week_step))
    return customer_model
    
#
##
### SPLIT, SAVE, AND UTILS
        
def split_save_customer_model(dataset_name, week_steps=11,
    week_target=4, overwrite=True):
    import pyspark.sql.functions as f
    from pyspark.sql.window import Window
    from dateutil.relativedelta import relativedelta
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"
    
    # do the steps  
    events = spark.read.format("delta").load(data_path+"events")#.sample(fraction=.1)
    max_date = events.agg(f.to_date(f.nex_day(f.max("event_time"),"Sun")\
        +relativedelta(days=-7)).alias("mdt")).collect()[0]["mdt"]
    
    for week_step in range(week_steps):
        # add some logs/prints
        temp_max_date = max_date+relativedelta(days=-(7*week_step))
        temp_split_date = temp_max_date+relativedelta(days=-(7*week_target))
        temp_customer_model = _construct_customer_model(dataset_name,
                events.where(f.col("event_time")<=temp_max_date),
                    temp_split_date, week_step, week_target)
        if "customer_model" not in locals():
            customer_model = temp_customer_model
        else:
            customer_model = customer_model.union(temp_customer_model)
    # add row_id
    customer_model = customer_model.withColumn("row_id",f.row_number()\
        .over(Window.orderBy(f.monotonically_increasing_id())))
    
    # flush it
    mode = "append"
    if overwrite:
        spark.sql("DROP TABLE IF EXISTS "\
            + f"churndb.{dataset_name}_customer_model")      
        mode = "overwrite"
    customer_model.write.format("delta").mode(mode)\
        .saveAsTable(f"churndb.{dataset_name}_customer_model")
    


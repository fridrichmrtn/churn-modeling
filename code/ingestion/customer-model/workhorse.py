# Databricks notebook source
# MAGIC %run "./base-features"

# COMMAND ----------

# MAGIC %run "./preference-features"

# COMMAND ----------

### NOTE: ADD DOCSTRINGS

#
##
### TARGET AND FEATURES

def _get_target(events, split_time):
    import pyspark.sql.functions as f
    
    target = events.select("user_id").distinct()\
        .join((events.where(f.col("event_time")>split_time)
            .groupBy("user_id").agg(
                f.when(f.sum("purchase")>0,1).otherwise(0).alias("target_event"),
                f.sum(f.col("revenue")).alias("target_revenue"))),
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
    zero_cols = [c for c in customer_model.columns for fc in ["_lag[0-9]+$", "_ma[0-9]+$", "_stddev$"]\
        if re.search(fc, c) is not None]
    customer_model = customer_model.fillna(0, subset=zero_cols)

    # max imputation
    max_cols = [c for c in customer_model.columns for fc in ["^inter_", "^time_to_", "_variation$"]\
        if re.search(fc, c) is not None]
    max_expr = [f.max(c).alias(c) for c in max_cols]
    max_vals = customer_model.agg(*max_expr).toPandas().transpose().to_dict()[0]
    customer_model = customer_model.fillna(max_vals)
    return customer_model

def _construct_customer_model(dataset_name, events, split_time):
    import pyspark.sql.functions as f
    import mlflow
    
    cust_target = _get_target(events, split_time)
    cust_events = _get_feature_events(events, split_time).persist()    

    # BASE MODEL
    cust_base = get_base_features(cust_events)

    # PREFERENCE MODELS
    cust_pref = get_pref_features(cust_events, dataset_name, refit=True)
    
    # ALL TOGETHER
    customer_model = (cust_base.join(cust_pref, on=["user_id"])
        .join(cust_target, on=["user_id"]))    

    # IMPUTATION
    customer_model = _impute_customer_model(customer_model)
    return customer_model
    
#
##
### SPLIT, SAVE, AND UTILS

def remove_customer_model(dataset_name):
    #data_path = f"dbfs:/mnt/{dataset_name}/delta/customer_model"
    spark.sql("DROP TABLE IF EXISTS "\
        + f"churndb.{dataset_name}_customer_model")
    
def load_customer_model(dataset_name):
    return spark.table(f"churndb.{dataset_name}_customer_model")        
    
def split_save_customer_model(dataset_name, week_steps=11, week_target=4, overwrite=True):
    # construct temp cust models
    import pyspark.sql.functions as f
    from dateutil.relativedelta import relativedelta
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"
    
    if overwrite:
        remove_customer_model(dataset_name)    
    # do the steps  
    events = spark.read.format("delta").load(data_path+"events")#.sample(fraction=.1)
    max_date = events.agg(f.to_date(f.max(f.col("event_time"))).alias("mdt"))\
        .collect()[0]["mdt"]    
    for week_step in range(week_steps):
        # add some logs/prints
        temp_max_date = max_date+relativedelta(days=-(7*week_step))
        temp_split_date = temp_max_date+relativedelta(days=-(7*week_target))
        customer_model = _construct_customer_model(dataset_name,
            events.where(f.col("event_time")<=temp_max_date), temp_split_date)
        customer_model = customer_model.withColumn("week_step", f.lit(week_step))
        customer_model.write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_customer_model")

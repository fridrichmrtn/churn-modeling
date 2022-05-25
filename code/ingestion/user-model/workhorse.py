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
### USER MODEL

def _construct_user_model(events, split_time, seed=42):
    import pyspark.sql.functions as f
    import mlflow
    
    transaction_run_id = ""
    view_run_id = ""
    user_target = _get_target(events, split_time)
    user_events = _get_feature_events(events, split_time).persist()    

    # BASE MODEL
    user_base = get_base_features(user_events)

    # PREFERENCE MODELS
    
    # ARE THERE OPT PARAMS? IF YES, JUST FIT THE PARAMS
    # IF NOT, DO THE OPT, PUSH MODEL
    
    user_transactions = user_events.where(f.col("event_type_name")=="purchase")
    transaction_recommendation_model = fit_optimized_recom(user_transactions, "transactions",,seed)
    #transaction_recommendation_model =  mlflow.spark.load_model("models:/refit_recom_transactions/None").stages[0]
    user_transaction_preference = get_user_factors(transaction_recommendation_model, "transactions")
    user_views = user_events.where(f.col("event_type_name")=="view")
    view_recommendation_model = fit_optimized_recom(user_views, "views", seed)
    #transaction_recommendation_model =  mlflow.spark.load_model("models:/refit_recom_views/None").stages[0]
    user_interaction_preference = get_user_factors(view_recommendation_model, "views")    
    # tran in feature period > 0
    return (user_base.join(user_transaction_preference, on=["user_id"])
        .join(user_interaction_preference, on=["user_id"]).join(user_target, on=["user_id"]))
    
#
##
### SPLIT AND SAVE

def remove_user_model(dataset_name):
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"
    dbutils.fs.rm(data_path+"user_model", True)
    
def load_user_model(dataset_name):
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"  
    
def split_save_user_model(dataset_name, week_steps=11, week_target=4, overwrite=True):
    # construct temp user models
    import pyspark.sql.functions as f
    from dateutil.relativedelta import relativedelta
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"
    
    events = spark.read.format("delta").load(data_path+"events")#.sample(fraction=.001)
    max_date = events.agg(f.to_date(f.max(f.col("event_time"))).alias("mdt"))\
        .collect()[0]["mdt"]
    # do the time steps
    if overwrite:
        remove_user_model(dataset_name)
    # split and construct       
    for week_step in range(week_steps):
        # add some logs/prints
        temp_max_date = max_date+relativedelta(days=-(7*week_step))
        temp_split_date = temp_max_date+relativedelta(days=-(7*week_target))
        temp_user_model = _construct_user_model(
            events.where(f.col("event_time")<=temp_max_date), temp_split_date)
        temp_user_model = temp_user_model.withColumn("week_step", f.lit(week_step))
        temp_user_model.write.format("delta").mode("append").save(data_path+"user_model")


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

def _construct_customer_model(events, split_time, seed=42):
    import pyspark.sql.functions as f
    import mlflow
    
    transaction_run_id = "d7711ccc392e4c01844ffeca197660e0"
    view_run_id = "db0941699fd94fb08b1c23d3350a3a2c"
    user_target = _get_target(events, split_time)
    user_events = _get_feature_events(events, split_time).persist()    

    # BASE MODEL
    user_base = get_base_features(user_events)

    # PREFERENCE MODELS
    # NOTE: possibly to extend with model loads
    user_transactions = user_events.where(f.col("event_type_name")=="purchase")
    transaction_recommendation_model = fit_optimized_recom(user_transactions, "transactions", transaction_run_id, seed)
    #transaction_recommendation_model =  mlflow.spark.load_model("models:/refit_recom_transactions/None").stages[0]
    user_transaction_preference = get_user_factors(transaction_recommendation_model, "transactions")
    user_views = user_events.where(f.col("event_type_name")=="view")
    view_recommendation_model = fit_optimized_recom(user_views, "views",view_run_id, seed)
    user_interaction_preference = get_user_factors(view_recommendation_model, "views")    
    # tran in feature period > 0
    return (user_base.join(user_transaction_preference, on=["user_id"])
        .join(user_interaction_preference, on=["user_id"]).join(user_target, on=["user_id"]))
    
#
##
### SPLIT AND SAVE

def remove_customer_model(dataset_name):
    #data_path = f"dbfs:/mnt/{dataset_name}/delta/customer_model"
    spark.sql("DROP TABLE IF EXISTS "\
        + f"churndb.{dataset_name}_customer_model")
    
# NOTE: fix this
def load_user_model(dataset_name):
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"  
    
def split_save_customer_model(dataset_name, week_steps=5, week_target=4, overwrite=True):
    # construct temp cust models
    import pyspark.sql.functions as f
    from dateutil.relativedelta import relativedelta
    data_path = f"dbfs:/mnt/{dataset_name}/delta/"
    
    if overwrite:
        remove_customer_model(dataset_name)    
    # do the steps  
    events = spark.read.format("delta").load(data_path+"events").sample(fraction=.1)
    max_date = events.agg(f.to_date(f.max(f.col("event_time"))).alias("mdt"))\
        .collect()[0]["mdt"]    
    for week_step in range(week_steps):
        # add some logs/prints
        temp_max_date = max_date+relativedelta(days=-(7*week_step))
        temp_split_date = temp_max_date+relativedelta(days=-(7*week_target))
        customer_model = _construct_customer_model(
            events.where(f.col("event_time")<=temp_max_date), temp_split_date)
        customer_model = customer_model.withColumn("week_step", f.lit(week_step))
        customer_model.write.format("delta").mode("append").saveAsTable(f"churndb.{dataset_name}_customer_model")


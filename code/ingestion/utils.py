# Databricks notebook source
### NOTE: ADD DOCSTRINGS

#
##
### TARGET

def get_target(events, split_time):
    import pyspark.sql.functions as f    
    target = events.select("user_id").join((events.where(f.col("event_time")>split_time)
            .groupBy("user_id").agg(
                f.when(f.sum("purchase")>0,1).otherwise(0).alias("target_event"),
                f.sum(f.col("revenue")).alias("target_revenue"))),
        on=["user_id"], how="left").na.fill(0)
    return target

#
##
### FEATURE DATA

def get_feature_events(events, split_time):
    import pyspark.sql.functions as f
    return events.where(f.col("event_time")<=split_time)


# Databricks notebook source
### NOTE: ADD DOCSTRINGS

#
##
### HELPERS

def _cv(colname):
    # coeficient of variation
    import pyspark.sql.functions as f
    return (f.stddev_samp(colname)/f.mean(colname))

def _add_lags(df, colnames, lags=[0,1,2,3], ma=3):
    # lagg features
    import pyspark.sql.functions as f
    from pyspark.sql.window import Window
    
    uw = Window.partitionBy("user_id").orderBy("start_monthgroup")
    uwr = Window.partitionBy("user_id").orderBy(f.col("start_monthgroup")).rowsBetween(-ma+1,0)
    # lags
    for c in colnames:
        for l in lags:
            df = df.withColumn(c+"_month_lag"+str(l), f.lag(c,l).over(uw)) # lags
        df = df.withColumn(c+"_month_ma"+str(ma), f.mean(c).over(uwr)) # moving avg  
        # possibly add diff, and a few other interactions
    return df

def _get_user_months(sessions):
    users = sessions.select("user_id").distinct()
    time = sessions.agg(f.max(f.col("start").alias("maxt")),
        f.min(f.col("start").alias("mint"))).collect()[0]
    max_time, min_time = time[0]+relativedelta(seconds=+1), time[1]
    results, temp_time = [], max_time
    while temp_time>min_time:
        temp_time += relativedelta(weeks=-4)
        results.append(temp_time)
    results = pd.DataFrame(results, columns=["start_time"])
    results["end_time"] = results["start_time"].shift().fillna(max_time)
    results = results.sort_values("start_time").reset_index(drop=True).reset_index()\
        .rename(columns={"index":"start_monthgroup"})
    return users.crossJoin(spark.createDataFrame(results))

#
##
### SESSIONS

def _get_sessions(events):
    import pandas as pd
    import pyspark.sql.functions as f
    from pyspark.sql.window import Window
    
    # base sessions
    sessions = events.groupBy("user_session_id", "user_id").agg(
        
        # time chars
            # overall session properties
            f.min("event_time").alias("start"), f.max("event_time").alias("end"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/60).alias("length"),
            # date-time components
            #f.make_date(f.year(f.min("event_time")),
            #  f.month(f.min("event_time")),f.lit(1)).alias("start_monthgroup"),
            #(f.next_day(f.min("event_time"), "Sun")-\
            #    f.expr("INTERVAL 4 WEEKS")).alias("start_monthgroup"),
            f.year(f.min("event_time")).alias("start_year"),
            f.dayofyear(f.min("event_time")).alias("start_yearday"), 
            f.month(f.min("event_time")).alias("start_month"),
            f.dayofmonth(f.min("event_time")).alias("start_monthday"),
            f.weekofyear(f.min("event_time")).alias("start_week"),
            f.dayofweek(f.min("event_time")).alias("start_weekday"),
            (f.when(f.dayofweek(f.min("event_time"))==1,1)\
                .when(f.dayofweek(f.min("event_time"))==7,1).otherwise(0)).alias("start_isweekend"),
            f.hour(f.min("event_time")).alias("start_hour"),
        # events
            # clicks
            f.max(f.when(f.col("event_type_name")=="purchase",1).otherwise(0)).alias("haspurchase"),
            f.count("user_id").alias("click_count"), f.sum("view").alias("view_count"), 
            f.sum("cart").alias("cart_count"), f.sum("purchase").alias("purchase_count"),    
            # time to action
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60*f.count("user_id"))).alias("time_to_click"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60*f.sum("view"))).alias("time_to_view"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60*f.sum("cart"))).alias("time_to_cart"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60*f.sum("purchase"))).alias("time_to_purchase"),    
        # revenue
            # sums
            f.sum((f.col("view")*f.col("price"))).alias("view_revenue"),
            f.sum((f.col("cart")*f.col("price"))).alias("cart_revenue"),
            f.sum((f.col("purchase")*f.col("price"))).alias("purchase_revenue"),
        
            # time to revenue
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60*f.sum("price"))).alias("time_to_click_revenue"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60* f.sum((f.col("view")*f.col("price"))))).alias("time_to_view_revenue"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60* f.sum((f.col("cart")*f.col("price"))))).alias("time_to_cart_revenue"),
            ((f.max("event_time")-f.min("event_time")).cast("long")/\
                (60* f.sum((f.col("purchase")*f.col("price"))))).alias("time_to_purchase_revenue"),
        # profit
           f.sum((f.col("purchase")*f.col("profit"))).alias("purchase_profit"))
    
    # windowing
    ws = Window().partitionBy("user_id").orderBy("start")
    last_session_start = sessions.agg(f.max(f.col("start")).alias("lsd"))\
        .collect()[0].__getitem__("lsd")
    sessions = sessions\
        .withColumn("session_number",f.row_number().over(ws))\
        .withColumn("inter_session_time", (f.col("start")-f.lag("start",1).over(ws)).cast("long")/(3600*24))\
        .withColumn("session_recency", ((last_session_start-f.col("start")).cast("long")/(3600*24)))
    purchases = sessions.where(f.col("haspurchase")==1)\
        .withColumn("purchase_number",f.row_number().over(ws))\
        .withColumn("inter_purchase_time", (f.col("start")-f.lag("start",1).over(ws)).cast("long")/(3600*24))\
        .withColumn("purchase_recency", ((last_session_start-f.col("start")).cast("long")/(3600*24)))\
        .select("user_session_id", "purchase_number", "inter_purchase_time", "purchase_recency")
    
    return sessions.join(purchases,["user_session_id"], "left")
  
#
##
### BASE FEATURES

def get_base_model(events, week_target):
    from functools import partial
    import pyspark.sql.functions as f
    from pyspark.sql.window import Window
    sessions = _get_sessions(events)
    user_months = _get_user_months(sessions)
    sessions = sessions.join(user_months, on=["user_id"])\
        .where((f.col("start")>=f.col("start_time"))&(f.col("start")<f.col("end_time")))\
            .drop(["start_time", "end_time"])
    
    # statistics
    excl_cols = set(["user_session_id", "user_id", "start", "end",
        "start_monthgroup", "purchase_profit"])
    stat_cols = [c for c in sessions.columns if c not in excl_cols]
    stat_funcs = [f.mean, f.sum, f.min, f.max, f.stddev_samp, _cv] # extend?
    stat_exp = [f(c).alias(c+"_"+list(filter(None,str(f.__name__).split("_")))[0])                
        for f in stat_funcs for c in stat_cols]
    # hand-crafted interactions, possibly extend
    int_exp = [(f.max("session_number")/f.max("session_recency")).alias("session_count_ratio"),
        (f.sum("click_count")/f.max("session_recency")).alias("click_count_ratio"),
        (f.sum("purchase_count")/f.max("session_recency")).alias("transaction_count_ratio")]
    agg_exp = stat_exp + int_exp
    base_features = sessions.groupBy("user_id").agg(*agg_exp)
    # lags
    uwc = Window.partitionBy("user_id").orderBy(f.col("start_monthgroup"))\
        .rowsBetween(Window.unboundedPreceding,0)
    user_month_groups = sessions.groupBy("user_id", "start_monthgroup").agg(
        f.count("user_session_id").alias("session_count"),
        f.sum("haspurchase").alias("purchase_count"),
        f.sum("purchase_revenue").alias("purchase_revenue"),
        f.sum("purchase_profit").alias("purchase_profit"))
    user_month_groups = user_month.join(user_month_groups,
        on=["user_id", "start_monthgroup"], how="left")\
            .fillna(0).withColumn("customer_value", f.mean(f.col("purchase_profit")).over(uwc))
    user_month_lags = _add_lags(user_month_groups,
        ["session_count", "purchase_count", "purchase_revenue", "customer_value"])
    last_monthgroup = user_month_lags.agg(f.max("start_monthgroup").alias("smg"))\
        .collect()[0].__getitem__("smg")
    lag_cols = [c for c in user_month_lags.columns if("_ma"in c)|("_lag" in c)|("user_id" in c)]
    user_month_lags = (user_month_lags.where(f.col("start_monthgroup")==last_monthgroup)
        .select(*lag_cols).fillna(0))
    # push it out
    return base_features.join(user_month_lags, on=["user_id"], how="inner")

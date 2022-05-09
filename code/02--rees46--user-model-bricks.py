# Databricks notebook source
# MAGIC %md
# MAGIC # User churn model
# MAGIC 
# MAGIC ### Data loading

# COMMAND ----------

import pyspark.sql.functions as f

# load raw data
DATA_IN = "dbfs:/mnt/rees46/raw/concatenated/sample/"
categories = spark.read.parquet(DATA_IN+"categories")
event_types = spark.read.parquet(DATA_IN+"event_types")
events = spark.read.parquet(DATA_IN+"events")
products = spark.read.parquet(DATA_IN+"products")

# denorm
events = events.join(event_types, on="event_type_id", how="inner")\
    .join(products, on="product_id", how="inner")\
    .join(categories, on="category_id", how="left")\
    .withColumn("view", (f.col("event_type_name")=="view").cast("int"))\
    .withColumn("cart", (f.col("event_type_name")=="cart").cast("int"))\
    .withColumn("purchase", (f.col("event_type_name")=="purchase").cast("int"))\
    .withColumn("revenue", f.col("purchase")*f.col("price"))

events = events.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate sessions

# COMMAND ----------

# sessions agg
sessions = events.groupBy("user_session_id", "user_id").agg(    
    # time chars
        # overall session properties
        f.min("event_time").alias("start"), f.max("event_time").alias("end"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/60).alias("length"),
        # date-time components
        f.make_date(f.year(f.min("event_time")),f.month(f.min("event_time")),f.lit(1)).alias("start_monthgroup"),
        f.year(f.min("event_time")).alias("start_year"), f.dayofyear(f.min("event_time")).alias("start_yearday"), 
        f.month(f.min("event_time")).alias("start_month"), f.dayofmonth(f.min("event_time")).alias("start_monthday"),
        f.weekofyear(f.min("event_time")).alias("start_week"), f.dayofweek(f.min("event_time")).alias("start_weekday"),
        (f.when(f.dayofweek(f.min("event_time"))==1,1).when(f.dayofweek(f.min("event_time"))==7,1).otherwise(0)).alias("start_isweekend"),
        f.hour(f.min("event_time")).alias("start_hour"),
    
    # events
        # clicks
        f.max(f.when(f.col("event_type_name")=="purchase",1).otherwise(0)).alias("haspurchase"),
        f.count("user_id").alias("click_count"), f.sum("view").alias("view_count"), 
        f.sum("cart").alias("cart_count"), f.sum("purchase").alias("purchase_count"),    
        # time to action
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.count("user_id"))).alias("time_to_click"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("view"))).alias("time_to_view"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("cart"))).alias("time_to_cart"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("purchase"))).alias("time_to_purchase"),    
    
    # revenue
        # sums
        f.sum((f.col("view")*f.col("price"))).alias("view_revenue"), f.sum((f.col("cart")*f.col("price"))).alias("cart_revenue"),
        f.sum((f.col("purchase")*f.col("price"))).alias("purchase_revenue"),
        # time to revenue
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60*f.sum("price"))).alias("time_to_click_revenue"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60* f.sum((f.col("view")*f.col("price"))))).alias("time_to_view_revenue"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60* f.sum((f.col("cart")*f.col("price"))))).alias("time_to_cart_revenue"),
        ((f.max("event_time")-f.min("event_time")).cast("long")/(60* f.sum((f.col("purchase")*f.col("price"))))).alias("time_to_purchase_revenue"))

# windowing
from pyspark.sql.window import Window
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

sessions = sessions.join(purchases,["user_session_id"], "left")
sessions.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Aggregate users

# COMMAND ----------

# statistics
def cv(colname):
    # coeficient of variation
    import pyspark.sql.functions as f
    return (f.mean(colname)/f.stddev_samp(colname))
excl_cols = set(["user_session_id", "user_id", "start", "end", "start_monthgroup"])
stat_cols = [c for c in sessions.columns if c not in excl_cols]
stat_funcs = [f.mean, f.sum, f.min, f.max, f.stddev_samp, cv] # extend?
stat_exp = [f(c).alias(c+"_"+str(f.__name__).split("_")[0])for f in stat_funcs for c in stat_cols]
# hand-crafted interactions
int_exp = [(f.max("session_number")/f.max("session_recency")).alias("session_count_daily_ratio"),
    (f.sum("click_count")/f.max("session_number")).alias("click_count_ratio"),
    (f.sum("purchase_count")/f.max("session_number")).alias("transaction_count_ratio")]
agg_exp = stat_exp + int_exp
users = sessions.groupBy("user_id").agg(*agg_exp)
users.filter(f.col("user_id")==3960141).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lagged features

# COMMAND ----------

user_month = sessions.select("user_id").distinct().crossJoin(sessions.select("start_monthgroup").distinct())
user_month_groups = sessions.groupBy("user_id", "start_monthgroup").agg(
    (f.count("user_session_id")).alias("session_count"),
    (f.sum("haspurchase")).alias("purchase_count"),
    (f.sum("purchase_revenue")).alias("purchase_revenue"))
user_month_groups = user_month.join(user_month_groups,
    on=["user_id", "start_monthgroup"], how="left")

def add_lags(df, colnames, lags=[0,1,2,3], ma=3):
    import pyspark.sql.functions as f
    from pyspark.sql.window import Window
    uw = Window.partitionBy("user_id").orderBy("start_monthgroup")
    uwr = Window.partitionBy("user_id").orderBy(f.col("start_monthgroup")).rowsBetween(-ma,0)
    # lags
    for c in colnames:
        for l in lags:
            df = df.withColumn(c+"_month_lag"+str(l), f.lag(c,l).over(uw)) # lags
        df = df.withColumn(c+"_month_ma"+str(ma), f.mean(f.col("session_count")).over(uwr)) # moving avg
        # possibly add diff, and a few other interactions
    return df

user_month_lags = add_lags(user_month_groups,
    ["session_count", "purchase_count", "purchase_revenue"])

last_monthgroup = user_month_lags.agg(f.max("start_monthgroup").alias("smg"))\
    .collect()[0].__getitem__("smg")
cols = [c for c in user_month_lags.columns if ("_ma" in c) or ("_lag" in c) or ("user_id" in c)]
user_month_lags = user_month_lags.where(f.col("start_monthgroup")==last_monthgroup).select(*cols)
users = users.join(user_month_lags, on=["user_id"], how="inner")    
users.filter(f.col("user_id")==3960141).toPandas()
del user_month_groups, user_month_lags

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Construct preferences

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
rs = 202205
# transactions
transactions = events.where(f.col("event_type_name")=="purchase").groupBy("user_id", "product_id")\
    .agg(f.count(f.col("user_session_id")).alias("implicit_feedback"))
(training, test) = transactions.randomSplit([0.6, 0.4], seed=rs)

als = ALS(userCol="user_id", itemCol="product_id", ratingCol="implicit_feedback", seed=rs,
          rank=5, maxIter=25, alpha=40, regParam=0.1, implicitPrefs=True, coldStartStrategy="drop")

model = als.fit(training)
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="implicit_feedback", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))


# COMMAND ----------

len(model.userFactors.select("features").toPandas().iloc[0,:].values[0])

# COMMAND ----------

from pyspark.sql.types import FloatType
dims = als.getRank()
fact_names = ["implicit_preference_f"+str(d) for d in range(dims)]
splits = [f.udf(lambda val:val[d], FloatType()) for d in range(dims)]
model.userFactors.select(f.col("id").alias("user_id"),*[s('features').alias(j) for s,j in zip(splits,fact_names)]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### WHAT TO DO TOMORROW
# MAGIC * pick preference representation - DONE
# MAGIC * construct preference vec for each user - DONE
# MAGIC * add hyperparameter tuning
# MAGIC * add vizualizations - user and item maps + centroids and interpretations - START
# MAGIC 
# MAGIC ### WHAT TO DO THE DAY AFTER TOMORROW
# MAGIC * TARGET VECTORS
# MAGIC * GENERAL FILTERS
# MAGIC * ML PIPE

# COMMAND ----------

# dont forget to do the data split

# rectify the timezone

# implement rfm on both ses and tran

# implement preferences on product basis through reco
    # als
    # w2v
    # others?

# what is others
    # hour
    # day of month
    # month

# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC Working plan
# MAGIC * load sample data
# MAGIC * implement the existing workflow in spark
# MAGIC * add target exploration and decide on the time-window
# MAGIC 
# MAGIC Next steps
# MAGIC * propose a churn model based on previous publication
# MAGIC * implement the churn model

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

# sessions
sessions = events.groupBy("user_session_id") \
    .agg(f.min("user_id").alias("user_id"), f.min("event_time").alias("session_start"),
         f.max("event_time").alias("session_end"), f.count("event_time").alias("click_count"),
         f.sum("view").alias("view_count"), f.sum("cart").alias("cart_count"),
         f.sum("purchase").alias("purchase_count"), f.sum("revenue").alias("revenue"))\
    .withColumn("is_purchase", (f.col("revenue")>0).cast("int"))\
    .withColumn("is_abadoned", ((f.col("is_purchase")==0) & (f.col("cart_count")>0)).cast("int"))\
    .withColumn("duration", ((f.col("session_end").cast("long")-f.col("session_start").cast("long"))/60).cast("double"))
#sessions = sessions.persist()
sessions.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sessions, purchases, and trends

# COMMAND ----------

## sessions/events
    ## revenue, no of sessions, avg transaction size
    ## no of sessions, abadoned sessions, transaction sessions
    ## conversion rates
    
# get session data to pandas
session_plots = sessions.groupBy(f.to_date(sessions.session_start).alias("session_start"))\
    .agg(f.count("user_session_id").alias("session_count"), f.sum("is_abadoned").alias("abadoned_count"),
         f.sum("is_purchase").alias("purchase_count"), f.sum("revenue").alias("revenue"))\
    .withColumn("left_count",f.col("session_count")-f.col("abadoned_count")-f.col("purchase_count"))\
    .withColumn("revenue_avg", f.col("revenue")/f.col("purchase_count"))\
    .withColumn("tocart_conversion", (f.col("abadoned_count")+f.col("purchase_count"))/f.col("session_count"))\
    .withColumn("purchase_conversion", f.col("purchase_count")/f.col("session_count")).toPandas()
session_plots.head()

# COMMAND ----------

# general info
display(session_plots)

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
# revenue trends
fig, ax = plt.subplots(3,1,figsize=(15,18))
from matplotlib.lines import Line2D
palt = sns.color_palette()
_ = sns.lineplot(x="session_start", y="revenue",
    data=session_plots, color=palt[0],ax=ax[0]);
tax =_.axes.twinx()
sns.lineplot(x="session_start", y="revenue_avg",
    data=session_plots, color=palt[1], ax=tax);
_.legend(handles=[Line2D([], [], marker="_", color=palt[0], label="total revenue"),
    Line2D([], [], marker="_", color=palt[1], label="avg revenue")])
ax[0].set_ylabel("revenue");
tax.set_ylabel("average revenue")
ax[0].set_xlabel("");
ax[0].set_title("revenue trends");

# session trends
melted = pd.melt(session_plots,"session_start",
    ["tocart_conversion", "purchase_conversion"])
melted["variable"] = melted["variable"].\
    apply(lambda x: x.split("_")[0])    

sns.lineplot(x="session_start", y="value",
    hue="variable", data=melted, ax=ax[1])
ax[1].set_ylabel("conversion rate");
ax[1].set_xlabel("");
ax[1].set_title("conversion trends");
ax[1].get_legend().set_title("");
del melted;

# conversion rate trends
melted = pd.melt(session_plots,"session_start",
    ["session_count", "abadoned_count", "purchase_count"])
melted["variable"] = melted["variable"].\
    apply(lambda x: x.split("_")[0])    

sns.lineplot(x="session_start", y="value",
    hue="variable", data=melted, ax=ax[2])
ax[2].set_ylabel("no of sessions");
ax[2].set_xlabel("");
ax[2].set_title("session trends");
ax[2].get_legend().set_title("");
del melted, session_plots;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Users

# COMMAND ----------

## users
        ## no of active users, no of transact users, propensity to buy
        ## intersession time, no of sessions, browsed revenue
        ## interpurchase time, no of transactions, purchased revenue
        ## account maturity
grpr = f.to_date(f.col("session_start")).alias("session_start")
user_plots = sessions.groupBy(grpr).agg(f.countDistinct("user_id").alias("user_count"))\
    .join(sessions.where(f.col("purchase_count")>0).groupBy(grpr)\
        .agg(f.countDistinct("user_id").alias("user_purchase_count")),
    on="session_start", how="left")\
    .withColumn("propensity", f.col("user_purchase_count")/f.col("user_count")).toPandas()
user_plots.head()

# COMMAND ----------

# general info
display(user_plots)

# COMMAND ----------

## changes over time
fig, ax = plt.subplots(1,1,figsize=(15,6))
_ = sns.lineplot(x="session_start", y="value", hue="variable",
    data=pd.melt(user_plots, id_vars="session_start",
    value_vars=["user_count", "user_purchase_count"]), ax=ax);
ax.set_xlabel("");
ax.set_ylabel("no of users");
tax =_.axes.twinx()
palt = sns.color_palette()
sns.lineplot(x="session_start", y="value",
    data=pd.melt(user_plots, id_vars="session_start",
    value_vars=["propensity"]),color=palt[2], ax=tax);
tax.set_ylabel("propensity to buy");
_.legend(handles=[Line2D([], [], marker="_", color=palt[0], label="total users"),
    Line2D([], [], marker="_", color=palt[1], label="purchased users"),
    Line2D([], [], marker="_", color=palt[2], label="propensity to buy")]);
del user_plots;

# COMMAND ----------

## inter-session/purchase time
from pyspark.sql.window import Window
w = Window.partitionBy("user_id").orderBy("session_start")
# inter session
sessions = sessions.withColumn("inter_session_days",
    (f.col("session_start").cast("long")-f.lag("session_start",1).over(w).cast("long"))/(24*3600))
# inter purchase
sessions = sessions.join(sessions.where(f.col("is_purchase")==1).withColumn("inter_purchase_days",
        (f.col("session_start").cast("long")-f.lag("session_start",1).over(w).cast("long"))/(24*3600))\
            .select("user_session_id", "inter_purchase_days"),
    on="user_session_id", how="left")

# COMMAND ----------

# just push it into pandas
user_plots = sessions.groupby("user_id").agg(
    f.mean(f.col("inter_session_days")).alias("inter_session_days"),
    f.mean(f.col("inter_purchase_days")).alias("inter_purchase_days"),
    f.countDistinct(f.col("user_session_id")).alias("session_count"),
    f.sum(f.col("is_purchase")).alias("purchase_count"),
    f.sum("revenue").alias("purchase_revenue")).toPandas()

# inter session/purchase time
fig, ax = plt.subplots(1,2,figsize=(18,6))
(user_plots[["inter_session_days"]]).\
    plot(kind="hist", ax=ax[0]);
ax[0].set_xlabel("days");
ax[0].set_title("inter session time");

(user_plots[["inter_purchase_days"]]).\
    plot(kind="hist", ax=ax[1]);
ax[1].set_xlabel("days");
ax[1].set_title("inter purchase time");

# frequency
fig, ax = plt.subplots(1,2,figsize=(18,6))
(user_plots[["session_count"]]).\
    plot(kind="hist", logy=True, ax=ax[0]);
ax[0].set_xlabel("count");
ax[0].set_title("session frequency");

(user_plots[["purchase_count"]]).\
    plot(kind="hist", logy=True, ax=ax[1]);
ax[1].set_xlabel("count");
ax[1].set_title("purchase frequency");

# monetary
fig, ax = plt.subplots(1,1,figsize=(9,6))
(user_plots[["purchase_revenue"]]).\
    plot(kind="hist", logy=True, ax=ax);
ax.set_xlabel("revenue");
ax.set_title("purchase monetary");

# consider additional viz wrt RFM
del user_plots;

# COMMAND ----------

events.show(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categories and products

# COMMAND ----------

events.groupby("product")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Target

# COMMAND ----------

# date components
sessions = sessions.withColumn("month", f.month(f.col("session_start")))\
    .withColumn("year", f.year(f.col("session_start")))
# construct target
w = Window.partitionBy("user_id").orderBy("year", "month")
target = sessions[["year","month"]].distinct().crossJoin(sessions[["user_id"]].distinct())\
    .join(sessions.groupBy("year", "month", "user_id").agg(f.sum(f.col("revenue")).alias("revenue")),
        on=["year", "month", "user_id"], how="left").fillna(0, subset=["revenue"])\
    .withColumn("revenue_lag", f.lag("revenue",1).over(w)).fillna(0, subset=["revenue_lag"])\
    .withColumn("date", f.expr("make_date(year, month, 1)"))\
    .withColumn("target", f.col("revenue")-f.col("revenue_lag"))
target_plots = target.join(target.where(f.col("revenue")>0).select("user_id").distinct(), #usrs tran>0
    on="user_id", how="inner").toPandas()
target_plots.head()

# COMMAND ----------

display(target_plots.groupby("date", as_index=False).agg(
    revenue_avg=("revenue", "mean"), target_avg=("target","mean")))

# COMMAND ----------

# first order revenue difference
fig, ax = plt.subplots(1,1,figsize=(9,6))
(target_plots[["target"]]).\
    plot(kind="hist", logy=True, ax=ax);
ax.set_xlabel("revenue diff");
ax.set_title("target");

# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
from sklearn.metrics import (accuracy_score,
    precision_score,recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error)
from scipy.stats import t, sem, ttest_rel

# COMMAND ----------

#
##
### NATURAL METRICS

def _get_reg_metrics(df):
    metrics = {m.__name__:m for m in\
            [r2_score, mean_absolute_error, mean_squared_error]}
    results = {n:f(df["target_actual_profit"], df["predictions"]) for n,f in metrics.items()}
    return pd.Series(results)

def _get_class_metrics(df):
    metrics = {m.__name__:m for m in\
            [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    results = {n:f(df["target_event"], df.predictions) if n in "roc_" else \
               f(df["target_event"], (df["predictions"]>0.5).astype("int")) for n,f in metrics.items()}
    return pd.Series(results)

#
##
### CAMPAIGN METRICS

def _get_campaign_metrics(df):
    df = df.copy().sort_values("expected_profit", ascending=False)
    df["cumulative_expected_profit"] = df.expected_profit.cumsum()
    df["cumulative_actual_profit"] = df.target_actual_profit.cumsum()
    df["percentile"] = df.expected_profit.rank(ascending=False, pct=True)
    opt_ind =  df.cumulative_expected_profit.idxmax()
    return pd.Series({"maximum_expected_profit":df.cumulative_expected_profit[opt_ind],
            "maximum_actual_profit":df.cumulative_actual_profit[opt_ind],
            "percentile":df.percentile[opt_ind]})
    
def _get_reg_profit(df):
    return _get_campaign_metrics(
        df.rename(columns={"predictions":"expected_profit"}))

def _get_class_profit(df, params):
    df = df.merge(params, on="user_id", how="inner")
    df["expected_profit"] = df.predictions*df.gamma*(df.target_customer_value_lag1-df.delta)\
        -(1-df.predictions)*df.psi*df.delta
    df = df.groupby("user_id", as_index=False)[["expected_profit", "target_actual_profit"]].mean()    
    return _get_campaign_metrics(df)

def evaluate_predictions(dataset_name):
    customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
    predictions = spark.table(f"churndb.{dataset_name}_predictions")
    params = spark.table(f"churndb.{dataset_name}_campaign_params").toPandas()
    cols = ["row_id", "target_event", "target_actual_profit", "target_customer_value_lag1"]
    groups = ["pipe", "task", "set_type", "week_step"]
    # regression metrics
    reg_predictions = predictions.where((f.col("task")=="regression")).join(
        customer_model.select(cols), on=["row_id"]).toPandas()
    reg_metrics = reg_predictions\
        .groupby(groups, as_index=False).apply(_get_reg_metrics)\
            .melt(id_vars=groups, var_name="metric")
    reg_profit = reg_predictions\
        .groupby(groups, as_index=False).apply(_get_reg_profit)\
            .melt(id_vars=groups, var_name="metric")
    # classification
    class_predictions = predictions.where((f.col("task")=="classification")).join(
        customer_model.select(cols), on=["row_id"]).toPandas()
    class_metrics = class_predictions\
        .groupby(groups, as_index=False).apply(_get_class_metrics)\
            .melt(id_vars=groups, var_name="metric")
    class_profit = class_predictions\
        .groupby(groups, as_index=False).apply(_get_class_profit, params)\
            .melt(id_vars=groups, var_name="metric")
    return pd.concat([reg_metrics, reg_profit,
        class_metrics, class_profit]).reset_index(drop=True)
    
def save_evaluation(dataset_name, evaluation):  
    spark.createDataFrame(evaluation)\
        .write.format("delta").mode("overwrite")\
            .saveAsTable(f"churndb.{dataset_name}_evaluation")
    return None        
    

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
def plot_simulated_profit(sp):    
    f, a = plt.subplots(1,1, figsize=(10,7))
    sns.lineplot(#data=sp,
        x=sp.perc, y=sp.cecp, legend=False,
        color=sns.color_palette("rocket")[0], ax=a);
    sns.lineplot(#data=sp,
        x=sp.perc, y=sp.cacp, legend=False,
        color=sns.color_palette("rocket")[3], ax=a);
    a.set_ylabel("profit");
    a.set_xlabel("percentile");
    a.legend(loc="lower left",
        labels=["expected profit", "actual profit"]);
    a.axhline(0, linestyle="dotted", c="k");
    return None
#plot_simulated_profit(profit)    

# COMMAND ----------

#
##
### ADDITIONAL DIAGNOSTICS

# plot cumulative curves
def plot_simulated_profit(sp):    
    f, a = plt.subplots(1,1, figsize=(15,10))
    sns.lineplot(#data=sp,
        x=sp.perc, y=sp.cecp, legend=False,
        color=sns.color_palette("rocket")[0], ax=a);
    sns.lineplot(#data=sp,
        x=sp.perc, y=sp.cacp, legend=False,
        color=sns.color_palette("rocket")[3], ax=a);
    a.set_ylabel("profit");
    a.set_xlabel("percentile");
    a.legend(loc="lower left",
        labels=["expected profit", "actual profit"]);
    a.axhline(0, linestyle="dotted", c="k");
    return None    

# expected values & ci bounds
def _ci(vec, alpha=0.95):
    mju = np.mean(vec)
    low, hi  = t.interval(alpha=alpha,
        df=len(vec)-1, loc=mju, scale=sem(vec))
    return (low, mju , hi)

def get_ci(df):
    df = df.groupby(["pipe", "set_type", "task", "metric"], as_index=False)\
        .agg(bounds=("value", _ci))
    df = pd.concat([df, pd.DataFrame(df["bounds"].tolist(),
        columns=["lb","mju","hb"])], axis=1)\
            .drop("bounds",axis=1)
    return df

def _tt(df, a="value_x", b="value_y"):    
    tstat, pval = ttest_rel(df[a], df[b])
    diff = np.mean(df[a]-df[b])
    return (diff, tstat, pval)

def get_tt(df):
    df = df.merge(df, on=["set_type", "week_step", "metric"])\
        .groupby(["pipe_x", "pipe_y","set_type", "metric"])\
            .apply(_tt).to_frame('ttest').reset_index()
    df = pd.concat([df, pd.DataFrame(df["ttest"].tolist(),
        columns=["diff", "stat", "pval"])], axis=1)\
            .drop("ttest",axis=1)
    df = df[df["pipe_x"]>df["pipe_y"]]
    return df

# b-v trade-off
def plot_bias_variance(df, metrics, figsize=(16,5)):
    tdf = pd.pivot_table(df,
        index=["pipe","week_step", "metric"],
            columns=["set_type"]).reset_index()
    tdf.columns = ["pipe","week_step", "metric", "test", "train"]
    f, axs = plt.subplots(1,3, figsize=figsize);
    for i,m in enumerate(metrics.items()):
        a = axs.flatten()[i]
        scatter = sns.scatterplot(data=tdf[tdf.metric==m[0]], x="train", y="test", hue="pipe", ax=a);        
        sns.lineplot(x=[0,1],y=[0,1], color="gray", ax=a, linestyle="dotted",  transform=scatter.transAxes);
        a.set_xlim(m[1]["xlim"]);
        #a.set_ylim(a.set_xlim());
        a.set_xlabel(m[1]["label"]+" on training split");
        a.set_ylabel(m[1]["label"]+" on testing split");
        a.legend_.remove();
    axs.flatten()[-1].legend(loc="lower right", frameon=False);

# COMMAND ----------

# # # TEST
# #dataset_name = "retailrocket"
# evaluation = spark.table(f"churndb.{dataset_name}_evaluation").toPandas()
# display(get_ci(evaluation).fillna(0))
# display(get_tt(evaluation))

# # metrics = {"accuracy_score":{"label":"acc", "xlim":(0,1)},
# #      "f1_score":{"label":"f1", "xlim":(0,1)},
# #      "roc_auc_score":{"label":"auc", "xlim":(0,1)}}

# metrics = {"r2_score":{"label":"r2", "xlim":(None,None)},
#      "mean_squared_error":{"label":"mse", "xlim":(None,None)},
#      "mean_absolute_error":{"label":"mae", "xlim":(None,None)}}   

# plot_bias_variance(evaluation, metrics=metrics)  

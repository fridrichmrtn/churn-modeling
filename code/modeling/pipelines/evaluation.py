# Databricks notebook source
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
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
    results = {n:f(df["target_event"],  df.predictions) if "roc_" in n else \
               f(df["target_event"], (df["predictions"]>0.5).astype("int")) for n,f in metrics.items()}
    return pd.Series(results)

#
##
### CAMPAIGN METRICS

def _get_campaing_metrics_data(df):
    df = df.copy().sort_values("expected_profit", ascending=False)
    df["cumulative_expected_profit"] = df.expected_profit.cumsum()
    df["cumulative_actual_profit"] = df.target_actual_profit.cumsum()
    df["percentile"] = df.expected_profit.rank(ascending=False, pct=True)
    return df
    
def _get_campaign_metrics(df):
    df = _get_campaing_metrics_data(df)
    opt_ind =  df.cumulative_expected_profit.idxmax()
    return pd.Series({"maximum_expected_profit":df.cumulative_expected_profit[opt_ind],
            "maximum_actual_profit":df.cumulative_actual_profit[opt_ind],
            "percentile":df.percentile[opt_ind]})
    
def _get_reg_profit_data(df):
    return df.rename(columns={"predictions":"expected_profit"})
    
def _get_reg_profit(df):
    return _get_campaign_metrics(
        _get_reg_profit_data(df))
    
def _get_class_profit_data(df, params):
    df = df.merge(params, on="user_id", how="inner")
    df["expected_profit"] = df.predictions*df.gamma*(df.target_customer_value_lag1-df.delta)\
        -(1-df.predictions)*df.psi*df.delta
    df = df.groupby("user_id", as_index=False)[["expected_profit", "target_actual_profit"]].mean()
    return df
    
def _get_class_profit(df, params):  
    return _get_campaign_metrics(_get_class_profit_data(df, params))

#
##
### RUNTIMES

def _get_run_duration(run_id):
    run_artifact = mlflow.get_run(run_id=run_id)
    return (datetime.fromtimestamp(run_artifact.info.end_time/1000)-\
         datetime.fromtimestamp(run_artifact.info.start_time/1000)).total_seconds()    
    
def _get_experiment_duration(exp_name):
    experiment_id = [experiment for experiment in mlflow.list_experiments()
        if exp_name == experiment.name][0].experiment_id
    last_run_id = mlflow.list_run_infos(experiment_id=experiment_id, order_by=["start_time DESC"])[0].run_uuid
    last_run = mlflow.get_run(run_id=last_run_id)
    if "mlflow.parentRunId" in last_run.data.tags.keys():
        last_run_parent_id = mlflow.get_run(run_id=last_run_id).data.tags["mlflow.parentRunId"]
        duration = _get_run_duration(last_run_parent_id)
    else:
        duration = _get_run_duration(last_run_id)
    return duration

def _get_runtimes(row, dataset_name):
    row_dict = row.to_dict()
    pipe_name = "_".join([row_dict["pipe"],str(row_dict["time_step"])])
    hyperopt_name = f"/{dataset_name}/modeling/hyperopt/{pipe_name}"
    hyperopt_duration = _get_experiment_duration(hyperopt_name)
    refit_name = f"/{dataset_name}/modeling/refit/{pipe_name}"
    refit_duration = _get_experiment_duration(refit_name)
    duration_dict = {"runtime_hyperopt":hyperopt_duration, "runtime_refit":refit_duration}
    return pd.Series({**row_dict,**duration_dict})

#
##
### EVALUATION

def evaluate_predictions(dataset_name):
    customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
    predictions = spark.table(f"churndb.{dataset_name}_predictions")
    params = spark.table(f"churndb.{dataset_name}_campaign_params").toPandas()
    cols = ["row_id", "target_event", "target_actual_profit", "target_customer_value_lag1"]
    groups = ["pipe", "task", "set_type", "time_step"]
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
    # runtimes
    runtimes = predictions.where(f.col("set_type")=="train")\
        .select(groups).distinct().toPandas()\
            .apply(_get_runtimes, axis=1, dataset_name=dataset_name)\
                .melt(id_vars=groups, var_name="metric")
        
    return pd.concat([reg_metrics, reg_profit,
        class_metrics, class_profit, runtimes]).reset_index(drop=True)
    
def save_evaluation(dataset_name, evaluation):
    spark.sql(f"DROP TABLE IF EXISTS churndb.{dataset_name}_evaluation;")
    spark.createDataFrame(evaluation)\
        .write.format("delta").mode("append")\
            .saveAsTable(f"churndb.{dataset_name}_evaluation")
    return None        


# COMMAND ----------

#
##
### ADDITIONAL DIAGNOSTICS

# plot cumulative curves
def get_cumulative_data(dataset_name, pipe_name,
    set_type="test", time_step=0):
    customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
    predictions = spark.table(f"churndb.{dataset_name}_predictions")
    cols = ["row_id", "target_event", "target_actual_profit", "target_customer_value_lag1"]
    pred_filter = (f.col("pipe")==pipe_name) & (f.col("set_type")==set_type) & (f.col("time_step")==time_step)
    predictions = predictions.where(pred_filter).join(customer_model.select(cols), on=["row_id"]).toPandas()
    if "class" in pipe_name:
        params = spark.table(f"churndb.{dataset_name}_campaign_params").toPandas()
        df = _get_class_profit_data(predictions, params)
    else:
        df = _get_reg_profit_data(predictions)
    return _get_campaing_metrics_data(df)

def plot_cumulative_curves(sp, remove_legend=True):    
    f, a = plt.subplots(1,1, figsize=(7,7))
    sns.lineplot(#data=sp,
        x=sp.percentile, y=sp.cumulative_expected_profit, legend=False,
        color=sns.color_palette("rocket")[0], ax=a);
    sns.lineplot(#data=sp,
        x=sp.percentile, y=sp.cumulative_actual_profit, legend=False,
        color=sns.color_palette("rocket")[3], ax=a);
    #a.set_ylim(-110000,30000)
    a.set_ylim(-200000,210000)
    a.set_ylabel("profit");
    a.set_xlabel("percentile");
    a.legend(loc="lower right",
        labels=["expected profit", "actual profit"]);
    a.axhline(0, linestyle="dotted", c="k");
    if remove_legend:
        a.get_legend().remove();    
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
    df = df.merge(df, on=["set_type", "time_step", "metric"])\
        .groupby(["pipe_x", "pipe_y","set_type", "metric"])\
            .apply(_tt).to_frame('ttest').reset_index()
    df = pd.concat([df, pd.DataFrame(df["ttest"].tolist(),
        columns=["diff", "stat", "pval"])], axis=1)\
            .drop("ttest",axis=1)
    df = df[df["pipe_x"]>df["pipe_y"]]
    return df

# b-v trade-off
# NOTE: clean-up model names
def plot_bias_variance(df, metrics, figsize=(16,5)):
    tdf = pd.pivot_table(df,
        index=["pipe","time_step", "metric"],
            columns=["set_type"]).reset_index()
    tdf.columns = ["pipe","time_step", "metric", "test", "train"]
    tdf["pipe"] = tdf["pipe"].apply(lambda x: "-".join(x.split("_")[:-1]))
    pipe_order = ["lr", "svm-lin", "svm-rbf", "mlp", "dt", "rf", "gbm"]
    
    f, axs = plt.subplots(1,3, figsize=figsize);
    for i,m in enumerate(metrics.items()):
        a = axs.flatten()[i]
        scatter = sns.scatterplot(data=tdf[tdf.metric==m[0]], x="train", y="test",
            hue="pipe", hue_order=pipe_order, palette="rocket", ax=a);        
        sns.lineplot(x=[0,1],y=[0,1], color="gray", ax=a, linestyle="dotted",  transform=scatter.transAxes);
        a.set_xlim(m[1]["xlim"]);
        a.set_ylim(a.set_xlim());
        a.set_xlabel(m[1]["label"]+" on training split");
        a.set_ylabel(m[1]["label"]+" on testing split");
        a.legend_.remove();
    axs.flatten()[-1].legend(loc="lower left", bbox_to_anchor=(1.04,0), frameon=False);
    f.tight_layout();

# COMMAND ----------

dataset_name = "retailrocket"
evaluation = spark.table(f"churndb.{dataset_name}_evaluation").toPandas()
evaluation = evaluation.loc[(evaluation.metric=="r2_score") & (evaluation.set_type=="train") & (evaluation["pipe"]=="gbm_reg"),:]
evaluation.sort_values("value")

# COMMAND ----------

# TEST
dataset_name = "retailrocket"
evaluation = spark.table(f"churndb.{dataset_name}_evaluation").toPandas()
evaluation = evaluation.loc[evaluation.time_step<4,:]
#display(get_ci(evaluation).fillna(0))
#display(get_tt(evaluation))

metrics = {"accuracy_score":{"label":"acc", "xlim":(0.8,1.01)},
     "f1_score":{"label":"f1", "xlim":(0.8,1.01)},
     "roc_auc_score":{"label":"auc", "xlim":(0.8,1.01)}}

# metrics = {"r2_score":{"label":"r2", "xlim":(-0.01,1.01)},
#      "mean_absolute_error":{"label":"mae", "xlim":(None,None)},
#      "mean_squared_error":{"label":"mse", "xlim":(None,None)}}   

plot_bias_variance(evaluation, metrics=metrics)  

# COMMAND ----------

# metrics = {"accuracy_score":{"label":"acc", "xlim":(0.85,1.01)},
#      "f1_score":{"label":"f1", "xlim":(0.85,1.01)},
#      "roc_auc_score":{"label":"auc", "xlim":(0.85,1.01)}}

metrics = {"r2_score":{"label":"r2", "xlim":(-0.01,1.01)},
     "mean_absolute_error":{"label":"mae", "xlim":(350,1750)},
     "mean_squared_error":{"label":"mse", "xlim":(10**5,5*10**6)}}   

plot_bias_variance(evaluation, metrics=metrics)  

# COMMAND ----------

def plot_cumulative_curves(sp, remove_legend=True):    
    f, a = plt.subplots(1,1, figsize=(7,7))
    percentiles = (sp.percentile*100).astype("int").values
    sns.lineplot(#data=sp,
        x=percentiles, y=sp.cumulative_expected_profit, legend=False,
        color=sns.color_palette("rocket")[0], ax=a);
    sns.lineplot(#data=sp,
        x=percentiles, y=sp.cumulative_actual_profit, legend=False,
        color=sns.color_palette("rocket")[3], ax=a);
    a.set_ylim(-110000,30000)
    #a.set_ylim(-210000,210000)
    a.set_ylabel("profit");
    a.set_xlabel("percentile");
    a.legend(loc="lower right",
        labels=["expected profit", "actual profit"]);
    a.axhline(0, linestyle="dotted", c="k");
    if remove_legend:
        a.get_legend().remove();    
    return None

# COMMAND ----------

df = get_cumulative_data("rees46", "gbm_class")
plot_cumulative_curves(df, True)

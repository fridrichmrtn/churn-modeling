# Databricks notebook source
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
from sklearn.metrics import (accuracy_score,
    precision_score,recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error)
from scipy.stats import t, sem, ttest_rel

simulation_config = {"retailrocket":{
    #"gamma":{"alpha":22.3, "beta":200},
    "gamma":{"alpha":20.5, "beta":113},
    "delta":800,
    "psi":{"alpha":9, "beta":1},
    "n_iter":100,
    "seed":1}}

# COMMAND ----------

#
##
### SIMULATED PROFIT

def get_simulated_profit(df, config):
    gamma = config["gamma"]
    delta = config["delta"]
    psi = config["psi"]
    n_iter=config["n_iter"]
    seed = config["seed"]
    np.random.seed(seed)
    n_users = df.user_id.nunique()
    sp = []
    for i in range(n_iter):
        gamma_psi = pd.DataFrame.from_dict({
            "user_id":df.user_id.unique(),
            "gamma":np.random.beta(gamma["alpha"], gamma["beta"], size=n_users),
            "psi":np.random.beta(psi["alpha"], psi["beta"], size=n_users)})
        temp = df.merge(gamma_psi, on=["user_id"])
        temp["ecp"] = (temp["y_pred_event_proba"]*temp["gamma"]*(temp["prev_target_cap"]-delta)
            + (1-temp["y_pred_event_proba"])*(-temp["psi"]*delta))
        temp["acp"] = (temp["target_event"]*temp["gamma"]*(temp["target_cap"]-delta)
            + (1-temp["target_event"])*(-temp["psi"]*delta))
        sp.append(temp.loc[:,["ecp", "acp"]])
    sp = pd.concat(sp)
    sp = sp.sort_values("ecp", ascending=False).reset_index(drop=True)
    sp["cecp"] = sp.ecp.cumsum()/n_iter
    sp["perc"] = sp.ecp.rank(ascending=False, pct=True) 
    sp["cacp"] = sp.acp.cumsum()/n_iter
    return sp

def get_campaign_profit(sp):
    imax = sp.cecp.idxmax()
    return {"mep_score":sp.cecp[imax],
            "map_score":sp.cacp[imax],
            "perc":sp.perc[imax]}

#
##
### EVALUATION METRICS

def _evaluate_predictions(df, config):    
    class_metrics = {m.__name__:m for m in\
        [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    reg_metrics = {m.__name__:m for m in\
        [r2_score, mean_squared_error]}

    class_dict = {n:f(df["target_event"], df["y_pred_event_proba"]) if "roc_auc" in n\
        else f(df["target_event"], df["y_pred_event"]) for n,f in class_metrics.items()}
    reg_dict = {n:f(df["target_cap"],df["y_pred_cap"]) if df["y_pred_cap"].isnull().sum()==0\
        else np.nan for n, f in reg_metrics.items()}
                 
    sp = get_simulated_profit(df, config)
    profit_dict = get_campaign_profit(sp)
    return pd.Series({**class_dict, **reg_dict, **profit_dict})
  
def evaluate_pipeline(dataset_name, config=simulation_config):
    customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
    predictions = spark.table(f"churndb.{dataset_name}_predictions")#.toPandas()
    prev_target_cap = customer_model.withColumn("week_step", f.col("week_step")-1)\
        .withColumnRenamed("target_cap", "prev_target_cap")\
            .select("user_id", "week_step", "prev_target_cap")
    observations = customer_model.join(prev_target_cap,
        on=["user_id", "week_step"], how="left").fillna(0)\
            .select("row_id", "target_event", "target_cap", "prev_target_cap")
    predictions = predictions.join(observations, on=["row_id"], how="left").toPandas()
    return predictions.groupby(["pipe", "type", "week_step"], as_index=False)\
        .apply(_evaluate_predictions, config[dataset_name])
    
def save_evaluation(dataset_name, evaluation):  
    spark.createDataFrame(evaluation)\
        .write.format("delta").mode("overwrite")\
            .saveAsTable(f"churndb.{dataset_name}_evaluation")
    return None    

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
    df = df.groupby(["pipe", "type", "metric"], as_index=False)\
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
    df = df.merge(df, on=["type", "week_step", "metric"])\
        .groupby(["pipe_x","pipe_y","type","metric"])\
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
            columns=["type"]).reset_index()
    tdf.columns = ["pipe","week_step", "metric", "test", "train"]
    f, axs = plt.subplots(1,3, figsize=figsize);
    for i,m in enumerate(metrics.items()):
        a = axs.flatten()[i]
        sns.scatterplot(data=tdf[tdf.metric==m[0]], x="train", y="test", hue="pipe", ax=a);
        sns.lineplot(x=[0,1],y=[0,1], color="gray", ax=a, linestyle="dotted");
        a.set_xlim(m[1]["xlim"]);
        #a.set_ylim(a.set_xlim());
        a.set_xlabel(m[1]["label"]+" on training split");
        a.set_ylabel(m[1]["label"]+" on testing split");
        a.legend_.remove();
    axs.flatten()[-1].legend(loc="lower right", frameon=False);

# COMMAND ----------

# # TEST
# dataset_name = "retailrocket"
# evaluation = spark.table(f"churndb.{dataset_name}_evaluation")
# melted_evaluation = evaluation.melt(id_vars=["type","week_step", "pipe"],
#     var_name="metric", value_name="value")
# display(get_ci(melted_evaluation).fillna(0))
# display(get_tt(melted_evaluation))
# metrics = {"accuracy_score":{"label":"acc", "xlim":(0,1)},
#     "f1_score":{"label":"f1", "xlim":(0,1)},
#     "roc_auc_score":{"label":"auc", "xlim":(0,1)}}    
# plot_bias_variance(melted_evaluation, metrics=metrics)  

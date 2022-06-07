# Databricks notebook source
# error metrics and confidence bounds
# create dummy dataframe and move on from there
import pandas as pd
import numpy as np
    
experimental_data = pd.DataFrame([
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"train", "week_step":0,"pipe":"lr"},
    
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"test", "week_step":0,"pipe":"lr"},
    
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"train", "week_step":0,"pipe":"dt"},
    
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"test", "week_step":0,"pipe":"dt"},

    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"train", "week_step":1,"pipe":"lr"},
    
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"test", "week_step":1,"pipe":"lr"},
    
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"train", "week_step":1,"pipe":"dt"},
    
    {"accuracy_score":np.random.random(),"precision_score":np.random.random(),
     "precision_recal":np.random.random(),"f1_score":np.random.random(),
     "roc_auc_score":np.random.random(),"type":"test", "week_step":1,"pipe":"dt"}    
])
experimental_data = experimental_data.melt(id_vars=["type","week_step", "pipe"],
    var_name="metric", value_name="value")
experimental_data.head()

# COMMAND ----------

 # expected values & ci bounds
def _ci(vec, alpha=0.95):
    import scipy.stats as st
    mju = np.mean(vec)
    # tcfs - small population, sigma unknown
    low, hi  = st.t.interval(alpha=alpha,
        df=len(vec)-1, loc=mju, scale=st.sem(vec))
    return (low, mju , hi)

def get_ci(df):
    df = df.groupby(["pipe", "type", "metric"], as_index=False)\
        .agg(bounds=("value", _ci))
    df = pd.concat([df, pd.DataFrame(df["bounds"].tolist(),
        columns=["lb","mju","hb"])], axis=1)\
            .drop("bounds",axis=1)
    return df

get_ci(experimental_data)

# COMMAND ----------

# testing diffs, in production dont forget to add bonf
# NOTE: consider also power

def _tt(df, a="value_x", b="value_y"):
    from scipy.stats import ttest_rel
    tstat, pval = ttest_rel(df[a], df[b])
    return (tstat, pval)

def get_tt(df):
    df = df.merge(df, on=["type", "week_step", "metric"])\
        .groupby(["pipe_x","pipe_y","type","metric"])\
            .apply(_tt).to_frame('ttest').reset_index()
    df = pd.concat([df, pd.DataFrame(df["ttest"].tolist(),
        columns=["stat", "pval"])], axis=1)\
            .drop("ttest",axis=1)
    df = df[df["pipe_x"]>df["pipe_y"]]
    return df

get_tt(experimental_data)

# COMMAND ----------

# JUST FINISH THIS SHIT AND GO HOME

import matplotlib.pyplot as plt
import seaborn as sns

tdf = pd.pivot_table(experimental_data,
    index=["pipe","week_step", "metric"],
        columns=["type"]).reset_index()
tdf.columns = ["pipe","week_step", "metric", "train", "test"]

f, axs = plt.subplots(1,3, figsize=(16,5));
for i,m in enumerate(["accuracy_score", "f1_score", "roc_auc_score"]):
    a = axs.flatten()[i]
    sns.scatterplot(data=tdf[tdf.metric==m], x="train", y="test", hue="pipe", ax=a);
    sns.lineplot(x=[0,1],y=[0,1], color="gray", ax=a, linestyle="dotted");
    #a.set_xlim(mets[m]);
    #a.set_ylim(ax.set_xlim());
    a.set_xlabel(m.split("_")[0]+" on training split");
    a.set_ylabel(m.split("_")[0]+" on validation split");
    a.legend_.remove();
axs.flatten()[-1].legend(loc="lower right", frameon=False);

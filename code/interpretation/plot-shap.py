# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances_argmin
from pyspark.sql import functions as f

# COMMAND ----------

# from mlflow.sklearn import load_model, save_model
# full_model = load_model("models:/retailrocket_svm_rbf_class_0/None")
# save_model(full_model,"/dbfs/mnt/retailrocket/pipelines/svm_rbf_class")
# full_model = load_model("models:/rees46_gbm_class_0/None")
# save_model(full_model,"/dbfs/mnt/rees46/pipelines/gbm_class")

# COMMAND ----------

class SHAP():
    def __init__(self, dataset_name, pipe):
        self.dataset_name_ = dataset_name
        self.pipe_ = pipe
        
    def _get_features(self, df):
        out_cols = ["user_id", "row_id", "time_step", "pipe", "base_values"]+\
            [c for c in df.columns if "target_" in c]
        return [c for c in df.columns if c not in out_cols] 
    
    def get_data(self):
        values = spark.table(f"churndb.{self.dataset_name_}_shap_values")\
            .where(f.col("pipe")==self.pipe_).orderBy(f.col("row_id"))
        self.feature_names = self._get_features(values)
        self.values = values.toPandas()\
            .loc[:,self.feature_names].values
        self.base_values = values.toPandas()\
            .loc[:,"base_values"].values
        self.data = spark.table(f"churndb.{self.dataset_name_}_customer_model")\
            .join(values.select("row_id").distinct(), on="row_id")\
                .orderBy(f.col("row_id")).toPandas()\
                    .loc[:,self.feature_names].values
        return self
    
    def save_data(self, path=None):
        if path is None:
            path = f"/dbfs/mnt/{self.dataset_name_}/pipelines/shap_{self.pipe_}"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        
def plot_feature_importance(shap_values, max_features=10, figsize=(5,4)):
    # data
    values = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    importance = values.abs().mean().sort_values(ascending=False)\
        .head(max_features).reset_index()
    importance.columns = ["variable", "shap"]
    # plot
    f, a = plt.subplots(1,1, figsize=figsize)
    sns.barplot(x=importance.shap, y=importance.variable,
        color=sns.color_palette("rocket")[0], ax=a);
    a.set_ylabel("");
    a.set_xlabel("$E(|SHAP|)$");
    
def plot_feature(shap_values, feature_name=None, xlim=None, bins="auto", figsize=(9,4)):
    # data
    cf = [c==feature_name for c in shap_values.feature_names]
    y = shap_values.values[:,cf].reshape(-1)
    x = shap_values.data[:,cf].reshape(-1)
    # plots
    f, axs = plt.subplots(1,2, figsize=figsize)
    sns.scatterplot(x=x,y=y, ax=axs[0],
        color=sns.color_palette("rocket")[0], alpha=.75)
    sns.histplot(x=x, bins=bins, ax=axs[1],
        color=sns.color_palette("rocket")[0])
    axs[0].set_xlim(xlim)
    axs[0].set_ylabel("SHAP")
    axs[0].set_xlabel(feature_name+" value")
    axs[1].set_xlim(xlim)
    axs[1].set_ylabel("frequency")
    axs[1].set_xlabel(feature_name+" value")
    return None

def plot_feature(shap_values, feature_name=None, xlim=None, bins="auto", figsize=(9,4)):
    # data
    cf = [c==feature_name for c in shap_values.feature_names]
    y = shap_values.values[:,cf].reshape(-1)
    x = shap_values.data[:,cf].reshape(-1)
    # plots
    f, axs = plt.subplots(1,2, figsize=figsize)
    sns.scatterplot(x=x,y=y, ax=axs[0],
        color=sns.color_palette("rocket")[0], alpha=.75)
    sns.histplot(x=x, bins=bins, ax=axs[1],
        color=sns.color_palette("rocket")[0])
    axs[0].set_xlim(xlim)
    axs[0].set_ylabel("SHAP")
    axs[0].set_xlabel(feature_name+" value")
    axs[1].set_xlim(xlim)
    axs[1].set_ylabel("frequency")
    axs[1].set_xlabel(feature_name+" value")
    return None

def plot_feature_interactions(shap_values, feature_name1=None, feature_name2=None,
    tiles=15, limits=None, figsize=(10,8)):

    c1 = [c==feature_name1 for c in shap_values.feature_names]
    c2 = [c==feature_name2 for c in shap_values.feature_names]
    shv = np.sum(shap_values.values[:,np.logical_or(c1,c2)], axis=1)
    fv1 = shap_values.data[:,c1].reshape(-1)
    fv2 = shap_values.data[:,c2].reshape(-1)

    plotx, ploty = np.meshgrid(
        np.linspace(np.min(fv1),np.max(fv1),tiles),
        np.linspace(np.min(fv2),np.max(fv2),tiles))
    plotz = griddata((fv1,fv2), shv, (plotx, ploty))

    if limits is not None:
        if "x" in limits.keys():
            plotx = set_lims(plotx, limits["x"])
        if "y" in limits.keys():
            ploty = set_lims(ploty, limits["y"])
        if "z" in limits.keys():
            plotz = set_lims(plotz, limits["z"])

    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(111, projection="3d")
    ax.plot_surface(plotx, ploty, plotz, cmap="rocket");
    ax.set_xlabel(feature_name1, labelpad=6);
    ax.set_ylabel(feature_name2, labelpad=6);
    ax.set_zlabel("SHAP", labelpad=6);
    f.tight_layout();
    return None

def get_inertia(df):
    dist = pairwise_distances(df.values,
        df.mean().values.reshape(1,-1))
    return np.sum(np.array(dist))

def evaluate_clustering(shap_values, range=range(3,10)):
    df = pd.DataFrame(shap_values.values,
        columns=shap_values.feature_names)
    score_dict = {}
    for n_clusters in range:
        clustering_fit = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering_fit.fit_predict(df)
        silh_score = silhouette_score(df, cluster_labels)
        intert_score = df.groupby(cluster_labels).apply(get_inertia).sum()
        score_dict[n_clusters] = {"silhouette_score":silh_score,
            "inertia_score":intert_score}
    score_df = pd.DataFrame.from_dict(score_dict).T.reset_index()
    score_df.columns = ["n_clusters"]+ list(score_df.columns[1:])
    return score_df

def plot_clusters(shap_values, cluster_labels, max_features=5, figsize=(7.5,4.5)):

    n_clusters = len(set(cluster_labels))
    shap_total = np.sum(shap_values.values, axis=1)
    df = pd.DataFrame(shap_values.values,
        columns=shap_values.feature_names)

    plt.rcParams.update({"font.size": 8, "axes.titlesize":8})
    f = plt.figure(figsize=figsize)
    subfigs = f.subfigures(1, 2, width_ratios=[1.1, 1])

    # left plot
    la = subfigs[0].subplots(n_clusters, 1, sharey=True, sharex=True)
    for cluster in range(n_clusters):
        sns.histplot(x=shap_total[cluster_labels==cluster],
            ax=la[cluster], binwidth=(np.max(shap_total)-np.min(shap_total))/50,
            stat="probability", color=sns.color_palette("rocket")[0], alpha=.75)
        la[cluster].set_ylabel("")

    # add axis labels
    subfigs[0].add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", which="both",
        top=False, bottom=False, left=False, right=False)
    plt.xlabel("SHAP");
    plt.ylabel("proportion");
    # add labels after axis adjustment
    for cluster in range(n_clusters):
        xlims = la[cluster].get_xlim()
        ylims = la[cluster].get_ylim()
        la[cluster].text(x=xlims[0]+(xlims[1]-xlims[0])*0.8,
            y=ylims[0]+(ylims[1]-ylims[0])*0.8,
            s=f"n = {len(shap_total[cluster_labels==cluster])}")
    # right plot
    ra = subfigs[1].subplots(n_clusters, 1,)
    cbar_ax = subfigs[1].add_axes([1.05, .30, .03, .40])
    for cluster in range(n_clusters):
        cluster_data = df.loc[cluster_labels==cluster,:].mean()
        feature_order = cluster_data.abs().sort_values(ascending=False).head(max_features)
        plot_data = pd.DataFrame(cluster_data.loc[feature_order.index], columns=["shap"]).T
        sns.heatmap(plot_data, ax=ra[cluster],
            cbar=True, vmin=-1250, vmax=1250, cbar_ax=cbar_ax,
            yticklabels="", xticklabels="",
            annot=np.array(["\n".join(c) for c in plot_data.columns.str.split("_")])\
                .reshape(plot_data.shape),
            annot_kws={"rotation":0}, fmt="");
    f.tight_layout();
    plt.rcParams.update(plt.rcParamsDefault);
                                                                                     
def get_cluster_centers(shap_values):
    def _get_center(df):
        min_row = pairwise_distances_argmin(df.values,
            df.mean().values.reshape(1,-1), axis=0)
        return df.iloc[min_row,:].index.values[0]
    centers = pd.DataFrame(shap_values.values,
        columns=shap_values.feature_names)\
            .groupby(cluster_labels, as_index=False)\
                .apply(_get_center).reset_index()
    centers.columns = ["cluster_label", "idx"]
    return centers

def plot_observation(shap_values, row, max_features=5, shap_xlim=(-.1,.1)):

    # data
    shap_row = pd.DataFrame(shap_values.values[row,:].reshape(1,-1),
        columns=shap_values.feature_names, index=["shap"]).T
    feature_names = shap_row.abs().sort_values("shap", ascending=False).head(max_features).index
    feature_comp_names = [i for i in shap_row.index if i not in feature_names]
    complement_shap = shap_row.loc[feature_comp_names,:].sum()
    complement_shap.name = f"contribution of remaining {len(feature_comp_names)} features"
    total_shap = shap_row.sum()
    total_shap.name = "shap expected"
    feature_shaps = shap_row.loc[feature_names,:]\
        .append(complement_shap, ignore_index=False)
    feature_shaps = pd.DataFrame(feature_shaps).reset_index()\
        .rename(columns={"index":"name"})
    feature_values = pd.DataFrame(shap_values.data,
        columns=shap_values.feature_names)
    feature_values.loc[row,feature_names]
    feature_values = feature_values.loc[row,feature_names]
    feature_labels = [feature_values.index[i]+" = {:.2f}".format(v)
        if np.abs(v)<10000 else feature_values.index[i]+" = {:.2e}".format(v)
            for i,v in enumerate(feature_values)]+[complement_shap.name]

    plt.rcParams.update({"font.size": 8, "axes.titlesize":8})
    f = plt.figure(constrained_layout=True, figsize=(8, 6))
    subfigs = f.subfigures(1, 2, width_ratios=[1.75, 1])

    # left plot
    la = subfigs[0].subplots(1,1) 
    cust_pal = {feature_shaps.name[v]:sns.color_palette("rocket")[0]\
        if feature_shaps.shap[v]>0 else sns.color_palette("rocket")[3]\
            for v in feature_shaps.index}
    a = sns.barplot(data=feature_shaps, x="shap", y="name",
        palette=cust_pal, ax=la)
    a.set_title("SHAP = {:.3f}".format(total_shap.values[0]),
        loc="left")
    a.set_ylabel("");
    a.set_yticklabels(feature_labels);
    a.set_xlabel("SHAP");
    a.set_xlim(shap_xlim);
    for container in a.containers:
        a.bar_label(container, fmt="%.3f", padding=3)

    # right plot
    feature_values = pd.DataFrame(shap_values.data,
        columns=shap_values.feature_names)        
    ra = subfigs[1].subplots(max_features+1, 1, gridspec_kw={"hspace":.10})    
    for i in range(max_features):
        a = sns.stripplot(data=feature_values, x=feature_names[i],
            color=sns.color_palette("rocket")[0], alpha=.1, ax=ra.flat[i])
        a.vlines(feature_values.loc[row, feature_names[i]], -1, 1,
            color=sns.color_palette("rocket")[3], linestyles="solid", linewidths=3)
        a.set_xlabel("");
        a.set_title(feature_names[i], loc="left");
    # hack with additional subplot
    ra[-1].set_frame_on(False)
    ra[-1].get_xaxis().set_ticks([])
    ra[-1].get_yaxis().set_ticks([])    
    plt.rcParams.update(plt.rcParamsDefault)

# COMMAND ----------

shap_list = [("retailrocket", "svm_rbf_class"), ("retailrocket", "gbm_reg"),
    ("rees46", "gbm_class"), ("rees46", "gbm_reg")]
for dataset_name, pipe in shap_list:
    shap_values = SHAP(dataset_name, pipe)
    shap_values = shap_values.get_data()
    shap_values.save_data()

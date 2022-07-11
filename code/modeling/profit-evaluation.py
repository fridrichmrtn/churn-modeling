# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # NOTE: MOVE THIS TO PROFIT DEV IPYNB
# MAGIC 
# MAGIC ### Profit estimation
# MAGIC 
# MAGIC 
# MAGIC #### Product margin
# MAGIC 
# MAGIC Unfortunately, margin on user-item transaction are omitted in the available datasets. To include the profit aspect of the customer relationship management, we propose simulation approach describing changes in product margin over time, which are consequently used for estimating customer profit. That is, for each of the product, we draw from initial random normal distribution. This draw serves as a starting point for the random walk, which we simulate using draws from random normal distribution for step difference and put them together using cumulative sum. In other words, we use one dimensional random walk with random normal steps. For product *p* in time *t*, we estimate the margin *m* like this:
# MAGIC    
# MAGIC $$m^{p}_t = Normal(\mu_0, \sigma_0)+\sum^{t}_{n=1}{Normal(\mu_{diff}, \sigma_{diff})}$$
# MAGIC 
# MAGIC where the first element represents the starting draw, and the second element represents the cumulative sum of difference draws. For simplicity, we assume that initial variability across products and variability of the product, within the observed period, are the same. Thus, we set \\(\mu_{diff} = 0\\), and \\(\sigma_{diff}=\frac{\sigma_0}{\sqrt{t}}\\).  As a result, we are able to estimate product profit with respect to just parameters of the initial random draw \\(\mu_{0}\\) and \\(\sigma_{0}\\).
# MAGIC 
# MAGIC #### Customer profit
# MAGIC 
# MAGIC The customer profit is computed using product revenue and simulated margin, result is scaled to reflect the target window size. This approach allows us to differentiate customer profit on individual level, indicate changes in customer behavior (partial churn), and can be used as secondary target for modeling. On the other hand, it is fairly limited by the window size and does not reflect on lifecycle length such as CLV. Let as have a customer \\(i\\) at time \\(t\\), the average customer profit can be computed as
# MAGIC 
# MAGIC $$ ACP^{i}_t = \frac{n_t}{t}\sum^{t}_{n=1}{m^{p}_n  r^{p}_n} $$
# MAGIC 
# MAGIC where \\(n_t\\) represent the length of the target window, \\(m^{p}_n\\) stands for simulated margin for product \\(p\\) in the time \\(n\\), and \\(r^{p}_n\\) is the revenue from transaction realized on product \\(p\\) in the time \\(n\\).
# MAGIC 
# MAGIC NOTE: INCLUDE TIME AXIS, REFERENCE OTHER APPROACHES
# MAGIC 
# MAGIC #### Plotting
# MAGIC 
# MAGIC 
# MAGIC * AVGCP VS TIME
# MAGIC * AVGCP DISTRIBUTION
# MAGIC * vs BOTH DATASETS
# MAGIC 
# MAGIC #### Sensitivity analysis
# MAGIC 
# MAGIC  * AVGCP VS mu0, sigma
# MAGIC  * vs BOTH DATASETS

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Expected campaign profit
# MAGIC 
# MAGIC The classical approach to retention management is to identify the customers at high risk of churning and persuade them to stay active with promotional offers. Predictive solutions are, however, primarily evaluated in terms of the classification task, not in the retention management context. To reflect on those additional steps, we propose an evaluation procedure inspired by the retention campaign (Neslin et al., 2006) and generated profit frameworks (Tamaddoni et al., 2015).
# MAGIC 
# MAGIC However, we focus on differentiated customer value and 
# MAGIC 
# MAGIC 
# MAGIC $$ \pi_i = p_i[\gamma_i V_i (1-\delta_i)] + (1-p_i)[-\psi_i \delta_i]$$
# MAGIC 
# MAGIC ### Maximum expected campaign profit

# COMMAND ----------

#
##
### SIMULATED PROFIT

def get_simulated_profit(predictions, config):
    import numpy as np
    import pandas as pd
    gamma = config["gamma"]
    delta = config["delta"]
    psi = config["psi"]
    n_iter=config["n_iter"]
    seed = config["seed"]
    
    np.random.seed(seed)
    n_users = predictions.user_id.nunique()
    sp = []
    for i in range(n_iter):
        gamma_psi = pd.DataFrame.from_dict({
            "user_id":predictions.user_id.unique(),
            "gamma":np.random.beta(gamma["alpha"], gamma["beta"], size=n_users),
            "psi":np.random.beta(psi["alpha"], psi["beta"], size=n_users)})
        temp = predictions.merge(gamma_psi, on=["user_id"])
        temp["ecp"] = (temp["y_pred_event_proba"] * temp["gamma"]*(temp["expected_cap"]-delta)
            + (1-temp["y_pred_event_proba"])*(-temp["psi"]*delta))
        temp["acp"] = (temp["target_event"]*temp["gamma"]*(temp["target_cap"]-delta)
            + (1-temp["target_event"])*(-temp["psi"]*delta))
        # NOTE: OPTIMIZE DTYPES
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
    
def plot_simulated_profit(sp):
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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

#
##
### EVALUATION

def _evaluate_predictions(df, config):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (accuracy_score,
        precision_score,recall_score, f1_score, roc_auc_score,
        r2_score, mean_squared_error)
    
    
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

# COMMAND ----------

#
##
### EVALUATE METRICS

# expected values & ci bounds
def _ci(vec, alpha=0.95):
    import scipy.stats as st
    import numpy as np
    
    mju = np.mean(vec)
    low, hi  = st.t.interval(alpha=alpha,
        df=len(vec)-1, loc=mju, scale=st.sem(vec))
    return (low, mju , hi)

def get_ci(df):
    import pandas as pd
    
    df = df.groupby(["pipe", "type", "metric"], as_index=False)\
        .agg(bounds=("value", _ci))
    df = pd.concat([df, pd.DataFrame(df["bounds"].tolist(),
        columns=["lb","mju","hb"])], axis=1)\
            .drop("bounds",axis=1)
    return df

def _tt(df, a="value_x", b="value_y"):
    from scipy.stats import ttest_rel
    import numpy as np
    
    tstat, pval = ttest_rel(df[a], df[b])
    diff = np.mean(df[a]-df[b])
    return (diff, tstat, pval)

def get_tt(df):
    import pandas as pd
    
    df = df.merge(df, on=["type", "week_step", "metric"])\
        .groupby(["pipe_x","pipe_y","type","metric"])\
            .apply(_tt).to_frame('ttest').reset_index()
    df = pd.concat([df, pd.DataFrame(df["ttest"].tolist(),
        columns=["diff", "stat", "pval"])], axis=1)\
            .drop("ttest",axis=1)
    df = df[df["pipe_x"]>df["pipe_y"]]
    return df

def plot_bias_variance(df, metrics, figsize=(16,5)):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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

import pyspark.sql.functions as f

profit_simulation_config = {"retailrocket":{
    "gamma":{"alpha":22.3, "beta":200},
    #"gamma":{"alpha":20.5, "beta":113},
    "delta":800,
    "psi":{"alpha":9, "beta":1},
    "n_iter":100,
    "seed":1}}

# load
dataset_name = "retailrocket"
customer_model = spark.table(f"churndb.{dataset_name}_customer_model")
predictions = spark.table(f"churndb.{dataset_name}_predictions")#.toPandas()

# put it together
prev_target_cap = customer_model.withColumn("week_step", f.col("week_step")-1)\
    .withColumnRenamed("target_cap", "prev_target_cap")\
        .select("user_id", "week_step", "prev_target_cap")
observations = customer_model.join(prev_target_cap,
    on=["user_id", "week_step"], how="left").fillna(0)\
        .select("row_id", "target_event", "target_cap", "prev_target_cap")

predictions = predictions.join(observations, on=["row_id"], how="left").toPandas()
predictions["expected_cap"] = predictions[["y_pred_cap", "prev_target_cap"]]\
    .bfill(axis=1).iloc[:,0]
    
evaluation = predictions.groupby(["pipe", "type", "week_step"], as_index=False)\
    .apply(_evaluate_predictions, profit_simulation_config["retailrocket"])

melted_evaluation = evaluation.melt(id_vars=["type","week_step", "pipe"],
    var_name="metric", value_name="value")

display(get_ci(melted_evaluation).fillna(0))
display(get_tt(melted_evaluation))

metrics = {"accuracy_score":{"label":"acc", "xlim":(0,1)},
    "f1_score":{"label":"f1", "xlim":(0,1)},
    "roc_auc_score":{"label":"auc", "xlim":(0,1)}}    
plot_bias_variance(melted_evaluation, metrics=metrics)  

# COMMAND ----------

melted_evaluation[melted_evaluation["pipe"]=="combinet"]

# COMMAND ----------

melted_evaluation[(melted_evaluation["pipe"]=="combinet") & (melted_evaluation.week_step==1)\
    & (melted_evaluation["type"]=="test")]

# COMMAND ----------

melted_evaluation[(melted_evaluation["pipe"]=="hgb") & (melted_evaluation.week_step==1)\
    & (melted_evaluation["type"]=="test")]

# COMMAND ----------



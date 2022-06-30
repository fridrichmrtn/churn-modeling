# Databricks notebook source
# MAGIC %md
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

def get_purchases(dataset_name):
    import pyspark.sql.functions as f
    import pyspark.pandas as ps
    
    data_path = f"dbfs:/mnt/{dataset_name}/delta/events"
    events = spark.read.format("delta").load(data_path)
    purchases = events.where(f.col("event_type_name")=="purchase")\
        .withColumn("date", f.to_date("event_time"))\
        .groupBy(["product_id", "date"])\
            .agg(f.sum("revenue").alias("revenue"))
    products = purchases.select("product_id").distinct()
    dates = purchases.select("date").distinct()    
    return {"purchases":ps.DataFrame(purchases),
            "products":ps.DataFrame(products), "dates":ps.DataFrame(dates)}

def randomize_m(products, dates, mu, scale, seed):
    import numpy as np
    import pyspark.pandas as ps
    
    n_products = len(products) 
    n_dates = len(dates)
    scale_diff = scale/np.sqrt(n_dates)
    # product-date margin
    np.random.seed(int(seed))
    product_baseline = np.random.normal(loc=mu, scale=scale, size=(n_products,1))
    product_diffs = np.random.normal(loc=0, scale=scale_diff, size=(n_products, n_dates))
    concat_diffs = np.concatenate([product_baseline, product_diffs], axis=1)
    margins = ps.DataFrame(np.cumsum(concat_diffs, axis=1)[:,1:],
        columns=dates, index=products)
    margins = ps.melt(margins.reset_index().rename(columns={"index":"product_id"}),
        id_vars=["product_id"], value_name="margin", var_name="date")
    return margins.to_spark()

def get_avgm(purchases, margins):
    import pyspark.sql.functions as f
    
    avg_margin = purchases.join(margins, on=["product_id", "date"], how="inner")\
        .agg((f.sum(f.col("revenue")*f.col("margin"))/f.sum("revenue")))\
            .toPandas().values[0]
    return float(avg_margin)
    
def construct_space(seed):
    import numpy as np
    from itertools import product
    import pyspark.pandas as ps

    np.random.seed(int(seed))
    loc = np.linspace(0.05, 0.35, num=5)
    scale = np.linspace(0.01, 0.15, num=5)
    seed = np.random.randint(low=0, high=2**16, size=5)
    space = ps.DataFrame(list(product(loc, scale, seed)),
        columns=["loc", "scale", "seed"])
    return space

def apply_avgm(row, purchases):
    margins = randomize_m(
        purchases["products"], purchases["dates"],
        mu=row["loc"], scale=row["scale"], seed=row["seed"])
    return get_avgm(purchases["purchases"], margins)

def plot_space(space):
    import numpy as np
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt
    
    X, Y, Z = space["loc"], space["scale"], space["margin"]
    plotx, ploty = np.meshgrid(
        np.linspace(np.min(X),np.max(X),25),
        np.linspace(np.min(Y),np.max(Y),25))
    plotz = griddata((X,Y), Z, (plotx, ploty), method="cubic")

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection="3d")
    #ax.view_init(elev=50, azim=30)
    ax.plot_surface(plotx, ploty, plotz, cmap="viridis")
    ax.set_xlabel("$\mu_0$")
    ax.set_ylabel("$\sigma_0$")
    ax.set_zlabel("expected margin")
    return ax

# COMMAND ----------

import pyspark.pandas as ps
ps.set_option("compute.ops_on_diff_frames", True)
purchases = get_purchases("rees46")
space = construct_space(0)
space["margin"] = space.apply(apply_avgm, axis=1, purchases=purchases)

# COMMAND ----------

plot_space(space.to_pandas());`

# COMMAND ----------

purchases = get_purchases("rees46")
products = purchases["products"]
dates = purchases["dates"]

# COMMAND ----------

import numpy as np

loc=0.15
scale=0.15
seed=1

n_products = products.shape[0]
n_dates = dates.shape[0]
scale_diff = scale/np.sqrt(n_dates)

# COMMAND ----------

import numpy as np
import pyspark.pandas as ps

n_products = len(products) 
n_dates = len(dates)
scale_diff = scale/np.sqrt(n_dates)
# product-date margin
np.random.seed(int(seed))
product_baseline = np.random.normal(loc=loc, scale=scale, size=(n_products,1))
product_diffs = np.random.normal(loc=0, scale=scale_diff, size=(n_products, n_dates))
concat_diffs = np.concatenate([product_baseline, product_diffs], axis=1)
margins = ps.DataFrame(np.cumsum(concat_diffs, axis=1)[:,1:],
    columns=dates, index=products)
margins = ps.melt(margins.reset_index().rename(columns={"index":"product_id"}),
    id_vars=["product_id"], value_name="margin", var_name="date")
margins.to_spark()

# COMMAND ----------

# ok, lets generate the shit using spark
import pyspark.sql.functions as f
import pyspark.pandas as ps
from pyspark.mllib.random import RandomRDDs
import datetime

sc = spark.getActiveSession()
min_date = min(dates["date"].to_numpy())-datetime.timedelta(1)
product_base = RandomRDDs.normalVectorRDD(sc,  numRows=n_products, numCols=1, seed=seed)\
    .map(lambda x: 0.15 + 0.15*x).map(lambda x: x.tolist()).toDF([str(min_date)])
product_diffs = RandomRDDs.normalVectorRDD(sc, numRows=n_products, numCols=n_dates, seed=seed)\
    .map(lambda x: 0 + 0.15*x).map(lambda x: x.tolist()).toDF([str(d) for d in dates["date"].to_numpy()])\

product_base = ps.concat([ps.DataFrame(product_base),ps.DataFrame(product_diffs)], axis=1)
product_base["product_id"] = products.product_id
margins = ps.melt(product_base, id_vars=["product_id"], value_name="margin", var_name="date")
margins[margins.product_id==38900088].sort_values("date").margin.cumsum()

# COMMAND ----------

import pyspark.pandas as ps
ps.set_option("compute.ops_on_diff_frames", True)
product_diffs["product_baseline"] = product_base["product_baseline"]

# COMMAND ----------

# RETRY ON THE REES46

# USE SPARK FOR THE SIMULATION`

# CONSIDER PUSHING RESULTS TO DELTA? LARGE SIMULATION TAB?
# CONSIDER WIDE DATASET, WITH GENERATED MARGIN COLUMNS

# DOCUMENT SIMULATION - DESCRIBE CALC MORE PRECISELY  
# DOCUMENT SIMULATION - PLOT CUSTOMER MARGIN DISTRIBUTION?
# DOCUMENT SIMULATION - PLOT ALL RANDOM WALKS / ONE-PRODUCT WALK? 

# COMMAND ----------

import pyspark.sql.functions as f
import pyspark.pandas as ps

dataset_name = "retailrocket"
data_path = f"dbfs:/mnt/{dataset_name}/delta/events"
events = spark.read.format("delta").load(data_path)



purchases = events.where(f.col("event_type_name")=="purchase")\
    .withColumn("date", f.to_date("event_time"))\
    .groupBy(["user_id", "date"])\
        .agg(f.sum("revenue").alias("revenue"))

purchases.count()

# COMMAND ----------



# COMMAND ----------

dates = purchases.select("date").distinct()
users = purchases.select("user_id").distinct()
revenue = users.crossJoin(dates).join(purchases, on=["user_id", "date"], how="left")\
    .fillna(0, subset=["revenue"]).toPandas()

# COMMAND ----------



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

profit_simulation_config = {"retailrocket":{
    "gamma":{"alpha":20, "beta":113},
    "delta":2000,
    "psi":{"alpha":9, "beta":1},
    "n_iter":1000,
    "seed":1}}

# COMMAND ----------

def _evaluate_predictions(df, config):
    # standard metrics
    from sklearn.metrics import accuracy_score, precision_score
    from sklearn.metrics import recall_score, f1_score, roc_auc_score
    import pandas as pd
    metrics = {m.__name__:m for m in\
        [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]}
    y = df["y"]
    predicted = df["y_pred"]
    predicted_proba = df["y_pred_proba"]
    results ={}
    for n,f in metrics.items():
        if "roc_auc" in n:
            results[n] = f(y, predicted_proba)
        else:
            results[n] = f(y, predicted)
    simulated_profit = get_simulated_profit(df, config)
    results["mep_score"] = simulated_profit.cecp.max()
    results["map_score"] = simulated_profit.cacp[simulated_profit.cecp.idxmax()]
    return pd.Series(results)

# COMMAND ----------

# NOTE: THIS MIGHT REFACTORED - SIMULATE THE PARAMS SEPARATELY
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
    simulated_profit = []
    
    for i in range(n_iter):
        gamma_psi = pd.DataFrame.from_dict({
            "user_id":predictions.user_id.unique(),
            "gamma":np.random.beta(gamma["alpha"], gamma["beta"], size=n_users),
            "psi":np.random.beta(psi["alpha"], psi["beta"], size=n_users)})
        temp = predictions.merge(gamma_psi, on=["user_id"])
        temp["ecp"] = (temp["y_pred_proba"] * temp["gamma"]*(temp["cap"]-delta)
            + (1-temp["y_pred_proba"])*(-temp["psi"]*delta))
        temp["acp"] = (temp["y"]*temp["gamma"]*(temp["cap"]-delta)
            + (1-temp["y"])*(-temp["psi"]*delta))
        # NOTE: OPTIMIZE DTYPES
        simulated_profit.append(temp.loc[:,["ecp", "acp"]])
    simulated_profit = pd.concat(simulated_profit)
    simulated_profit = simulated_profit\
        .sort_values("ecp", ascending=False).reset_index(drop=True)
    simulated_profit["cecp"] = simulated_profit.ecp.cumsum()/n_iter
    simulated_profit["perce"] = simulated_profit.ecp.rank(ascending=False, pct=True)
    simulated_profit = simulated_profit\
        .sort_values("acp", ascending=False).reset_index(drop=True)    
    simulated_profit["cacp"] = simulated_profit.acp.cumsum()/n_iter
    simulated_profit["perca"] = simulated_profit.acp.rank(ascending=False, pct=True)
    return simulated_profit

def plot_simulated_profit(simulated_profit):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    f, a = plt.subplots(1,1, figsize=(15,10))
    sns.lineplot(#data=simulated_profit,
        x=sp.perce, y=sp.cecp, legend=False,
        color=sns.color_palette("rocket")[0], ax=a);
    sns.lineplot(#data=simulated_profit,
        x=sp.perca, y=sp.cacp, legend=False,
        color=sns.color_palette("rocket")[3], ax=a);
    a.set_ylabel("profit");
    a.set_xlabel("percentile");
    a.legend(loc="lower left",
        labels=["expected profit", "actual profit"]);
    a.axhline(0, linestyle="dotted", c="k");
    return None

# COMMAND ----------

dataset_name = "retailrocket"
predictions = spark.table(f"churndb.{dataset_name}_predictions").toPandas()
predictions.groupby(["pipe","type", "week_step"], as_index=False)\
    .apply(_evaluate_predictions, profit_simulation_config[dataset_name])

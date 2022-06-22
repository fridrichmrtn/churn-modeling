# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Margin estimation
# MAGIC 
# MAGIC Unfortunately, margin on user-item transaction are omitted in the available datasets. To include the profit aspect of the customer relationship management, we propose simulation approach describing changes in product margin over time, which are consequently used for average customer profit estimation. That is, for each of the product, we draw from initial random normal distribution. This draw serves as a starting point for the random walk, which we simulate using draws from random normal distribution for step difference and put them together using cumulative sum. In other words, we use one dimensional random walk with random normal steps. For product *p* in time *t*, we estimate the margin *m* like this:
# MAGIC    
# MAGIC $$m^{p}_t = Normal(\mu_0, \sigma_0)+\sum^{t}_{n=1}{Normal(\mu_{diff}, \sigma_{diff})}$$
# MAGIC 
# MAGIC where the first element represents the starting draw, and the second element represents the cumulative sum of difference draws. We also expect \\(\mu_{diff} = 0\\), and \\(\sigma_{diff}=\frac{\sigma_0}{\sqrt{t}}\\) across all scenarios. As a result, we are able to estimate central tendency for customer's profit with respect to just \\(\mu_{0}\\) and \\(\sigma_{0}\\).

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

# COMMAND ----------

margins[margins.product_id==38900088].sort_values("date").margin.cumsum()

# COMMAND ----------

margins.where()

# COMMAND ----------

import pyspark.pandas as ps
ps.set_option("compute.ops_on_diff_frames", True)
product_diffs["product_baseline"] = product_base["product_baseline"]

# COMMAND ----------

# RETRY ON THE REES46

# USE SPARK FOR THE SIMULATION

# CONSIDER PUSHING RESULTS TO DELTA? LARGE SIMULATION TAB?
# CONSIDER WIDE DATASET, WITH GENERATED MARGIN COLUMNS

# DOCUMENT SIMULATION - DESCRIBE CALC MORE PRECISELY  
# DOCUMENT SIMULATION - PLOT CUSTOMER MARGIN DISTRIBUTION?
# DOCUMENT SIMULATION - PLOT ALL RANDOM WALKS / ONE-PRODUCT WALK? 

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Profit estimation
# MAGIC 
# MAGIC 
# MAGIC * initial equations
# MAGIC * classification matrix?

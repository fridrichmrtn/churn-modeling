# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Margin estimation
# MAGIC 
# MAGIC Unfortunately, margin on user-item transaction are omitted in the available datasets. To include the profit aspect of the customer relationship management, we propose simulation-based approach based on one-dimensional random walk with Gaussian steps. For each of the product, we draw from initial random normal distribution. This draw serves as a starting point for the random walk, which we simulate using draws from random normal distribution for step difference and put them together using cumulative sum. For product *p* in time *t*, we estimate the margin *m* like this:
# MAGIC    
# MAGIC $$m^{p}_t = Normal(\mu_0, \sigma_0)+\sum^{t}_{n=1}{Normal(\mu_{diff}, \sigma_{diff})}$$
# MAGIC 
# MAGIC where the first element represents the starting draw, and the second element represents the cumulative sum of difference draws. We also expect \\(\mu_{diff} = 0\\), and \\(\sigma_{diff}=\frac{\sigma_0}{t}\\) across all scenarios. As a result, we are able to estimate central tendency for customer's profit with respect to just \\(\mu_{0}\\) and \\(\sigma_{0}\\).

# COMMAND ----------

def get_purchases(dataset_name):
    import pyspark.sql.functions as f
    
    data_path = f"dbfs:/mnt/{dataset_name}/delta/events"
    events = spark.read.format("delta").load(data_path)
    purchases = events.where(f.col("event_type_name")=="purchase").persist()
    products = purchases.select("product_id").distinct()\
        .toPandas().values.reshape(-1)
    dates = purchases.select(f.to_date(f.col("event_time")).alias("date"))\
        .distinct().toPandas().sort_values("date").values.reshape(-1)    
    return {"purchases":purchases, "products":products, "dates":dates}

def randomize_m(products, dates, mu, scale, seed):
    import numpy as np
    import pyspark.pandas as ps
    
    n_products = len(products) 
    n_dates = len(dates)
    scale_diff = scale/n_dates
    # product-date margin
    np.random.seed(int(seed))
    product_baseline = np.random.normal(loc=mu, scale=scale, size=(n_products,1))
    product_diffs = np.random.normal(loc=0, scale=scale, size=(n_products, n_dates))
    concat_diffs = np.concatenate([product_baseline, product_diffs], axis=1)
    margins = ps.DataFrame(np.cumsum(concat_diffs, axis=1)[:,1:],
        columns=dates, index=products)
    margins = ps.melt(margins.reset_index().rename(columns={"index":"product_id"}),
        id_vars=["product_id"], value_name="margin", var_name="date")
    return margins.to_spark()

def get_avgm(purchases, margins):
    import pyspark.sql.functions as f
    
    purchases = purchases.withColumn("date", f.to_date("event_time"))\
        .join(margins, on=["product_id", "date"], how="inner")\
        .withColumn("profit", f.col("purchase")*f.col("price")*f.col("margin"))
    
    return float(purchases.withColumn("date", f.to_date("event_time"))\
        .join(margins, on=["product_id", "date"], how="inner")\
        .agg((f.sum("profit")/f.sum("revenue")).alias("margin"))\
            .toPandas().values[0])
    
def construct_space(seed):
    import numpy as np
    from itertools import product
    import pyspark.pandas as ps

    np.random.seed(int(seed))
    loc = np.linspace(0.05, 0.5, num=5)
    scale = np.linspace(0.05, 0.5, num=5)
    seed = np.random.randint(low=0, high=2**16, size=25)
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

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(plotx, ploty, plotz, cmap="viridis")
    ax.set_xlabel("$\mu_0$")
    ax.set_ylabel("$\sigma_0$")
    ax.set_zlabel("observed margin")
    return ax

# COMMAND ----------

import pyspark.pandas as ps
ps.set_option("compute.ops_on_diff_frames", True)
purchases = get_purchases("retailrocket")
space = construct_space(0)
space["margin"] = space.apply(apply_avgm, axis=1, purchases=purchases)

# COMMAND ----------

plot_space(space.to_pandas());

# COMMAND ----------



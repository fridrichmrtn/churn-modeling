# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ### Margin estimation
# MAGIC 
# MAGIC Unfortunately, margin on user-item transaction are omitted in the available datasets. To still include the profit aspect of the customer relationship management, we propose straightforward simulation-based approach based on one-dimensional random walk. For each of the product, we draw from initial random normal distribution. This draw serves as a starting point for the random walk, which we simulate using draws from random normal distribution for step changes and put them together using cumulative sum.
# MAGIC 
# MAGIC WRITE THIS IN LATEX    
# MAGIC productMarginBaseline = Normal(mju baseline, sigma baseline)  
# MAGIC 
# MAGIC REFORMULATE THIS EQUATIONS  
# MAGIC productMargin = Eta

# COMMAND ----------

events = spark.read.format("delta").load("dbfs:/mnt/retailrocket/delta/events")

# COMMAND ----------

events.show(5)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

days = 130
products = 22691
product_baseline = np.random.normal(loc=0.15, scale=0.15, size=(products,1))
product_diffs = np.random.normal(loc=0, scale=.15, size=(products, days))
concat_diffs = np.concatenate([product_baseline, product_diffs], axis=1)
results = np.cumsum(concat_diffs, axis=1)[:,1:]

# COMMAND ----------

plt.plot(results[50,:]);

# COMMAND ----------

np.mean(np.std(results))

# COMMAND ----------

# CLV SCENARIOS

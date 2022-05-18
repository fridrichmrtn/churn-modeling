# Databricks notebook source
# MAGIC %md
# MAGIC # User churn model - Preference
# MAGIC 
# MAGIC ### Data loading

# COMMAND ----------

# basic py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pyspark
import pyspark.sql.functions as f
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
# hopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

# mlops
import mlflow

# COMMAND ----------

# load raw data
DATA_IN = "dbfs:/mnt/rees46/raw/concatenated/sample/"
categories = spark.read.parquet(DATA_IN+"categories")
event_types = spark.read.parquet(DATA_IN+"event_types")
events = spark.read.parquet(DATA_IN+"events")
products = spark.read.parquet(DATA_IN+"products")

# denorm
events = events.join(event_types, on="event_type_id", how="inner")\
    .join(products, on="product_id", how="inner")\
    .join(categories, on="category_id", how="left")\
    .withColumn("view", (f.col("event_type_name")=="view").cast("int"))\
    .withColumn("cart", (f.col("event_type_name")=="cart").cast("int"))\
    .withColumn("purchase", (f.col("event_type_name")=="purchase").cast("int"))\
    .withColumn("revenue", f.col("purchase")*f.col("price"))

events = events.persist()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data transformation

# COMMAND ----------

# put the data together
transactions = events.where(f.col("event_type_name")=="purchase").groupBy("user_id", "product_id")\
    .agg(f.count(f.col("user_session_id")).alias("implicit_feedback"))

#user_filter = transactions.groupBy("user_id").agg(
#    f.sum(f.col("implicit_feedback")).alias("implicit_feedback")).where(f.col("implicit_feedback")>4)
#product_filter = transactions.groupBy("product_id").agg(
#    f.sum(f.col("implicit_feedback")).alias("implicit_feedback")).where(f.col("implicit_feedback")>9)
#transactions = transactions.join(user_filter.select("user_id"), on=["user_id"])\
#    .join(product_filter.select("product_id"), on=["product_id"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Recommendation engine
# MAGIC 
# MAGIC #### Hyperparameter optimization

# COMMAND ----------

# define reco training func
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator    

RECO_SEED = 202205

# param space
reco_space = {
    "rank": hp.randint("rank", 5, 30),
    "maxIter": hp.randint("maxIter", 5, 40),
    "regParam": hp.loguniform("regParam", -5, 3),
    "alpha": hp.uniform("alpha", 25, 350)
}

def train_reco_model(data, params, seed):
    # unpack params
    rank = int(params["rank"]) # int
    maxiter = int(params["maxIter"]) # int
    regparam = params["regParam"] # float
    alpha = params["alpha"] # float    
    # init
    als = ALS(userCol="user_id", itemCol="product_id", ratingCol="implicit_feedback",
        implicitPrefs=True, coldStartStrategy="drop", nonnegative=True,
        seed=seed, rank=rank, maxIter=maxiter, alpha=alpha, regParam=regparam)
    # fit
    return als.fit(data)


# COMMAND ----------

# IMPLEMENT CROSSVALIDATION
# define hyperopt space and funcs, run hyperopt
(training_data, validation_data) = transactions.randomSplit([0.6, 0.4], seed=RECO_SEED)
def train_reco_hyperopt(params):
    
    # NOTE : consider distributing the data for large training scenario
        # train
    with mlflow.start_run(run_name="hyperopt_reco"):
        mlflow.log_params(params)
        model = train_reco_model(training_data, params, RECO_SEED)
        # evaluate
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="implicit_feedback",
                predictionCol="prediction")
        training_rmse = evaluator.evaluate(model.transform(training_data))
        validation_rmse = evaluator.evaluate(model.transform(validation_data))
        mlflow.log_metric("training_rmse", training_rmse)
        mlflow.log_metric("validation_rmse", validation_rmse) 
        return {"loss":validation_rmse, "status":STATUS_OK}
    
# run the trials, peek at the results
reco_params = fmin(
    fn=train_reco_hyperopt,
    space=reco_space,
    algo=tpe.suggest,
    max_evals=25)
# explore the tuning    
space_eval(reco_space, reco_params)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Re-fit on the complete dataset

# COMMAND ----------

# retrain on full data with best params
with mlflow.start_run(run_name="refit_reco"):
    run = mlflow.active_run()
    mlflow.log_params(reco_params)
    transaction_reco = train_reco_model(transactions, reco_params, RECO_SEED)
    mlflow.spark.log_model(transaction_reco, "tran_reco_refit")
    path = f"runs:/{run.info.run_id}/tran_reco_refit"
    mlflow.register_model(path, "tran_reco_refit")

# COMMAND ----------

# reload
transaction_reco = mlflow.spark.load_model("models:/tran_reco_refit/None")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore learned characteristics
# MAGIC 
# MAGIC ##### Understanding the recommendations

# COMMAND ----------

# NOTE - CATEGORY CODES SEEMS ODD, CHECK THEM IN THE ORIGINAL DATASET
# pick one customer and show purchase history
print("Purchased items>\n")
transactions.where(f.col("user_id")==2581464)\
    .join(products, on=["product_id"]).join(categories, on=["category_id"], how="left").show(10)

# show suggestions
print("Recommended items>\n")
user_subset = transactions.select("user_id").where(f.col("user_id")==2581464).distinct()
transaction_reco.stages[0].recommendForUserSubset(user_subset,5)\
    .withColumn("temp", f.explode("recommendations")).select("user_id", "temp.product_id", "temp.rating")\
    .join(products, on=["product_id"]).join(categories, on=["category_id"], how="left").show()

# COMMAND ----------

# product latent space
dims = range(transaction_reco.stages[0].rank)
product_factors = transaction_reco.stages[0].itemFactors\
    .select(f.col("id").alias("product_id"),
        *[f.col("features").getItem(d).alias("latent_factor"+str(d)) for d in dims])\
    .persist()
product_factors.show()
# define hyperopt space and funcs, run hyperopt
feature_cols = [c for c in product_factors.columns if c not in "product_id"]
product_features = VectorAssembler(inputCols=feature_cols, outputCol="features")\
    .transform(product_factors).select("product_id","features").persist()
product_features.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Product clustering

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator

CLU_SEED = -42

# param space
clu_space = {
    "k": hp.randint("k",2,25),
    "maxIter": hp.randint("maxIter", 25, 350),
    "aggregationDepth": hp.randint("aggregationDepth", 5, 8),
}

# clustering over product latent space
def train_cluster_model(data, params, seed):
    # unpack params
    k = int(params["k"])
    maxiter = int(params["maxIter"])
    aggdepth = int(params["aggregationDepth"])
    # contruct pipe
    gmm = GaussianMixture(featuresCol="features",
        seed=seed, k=k, maxIter=maxiter, aggregationDepth=aggdepth)
    # fit
    return gmm.fit(data)

# COMMAND ----------

(training_data, validation_data) = product_features.randomSplit([0.6, 0.4], seed=CLU_SEED)
# extract prob
from pyspark.sql.types import DoubleType
@f.udf(returnType=DoubleType()) 
def get_max_arg(arr, a):
    import numpy as np
    arr = np.array(arr)
    try:
        return float(arr[a])
    except ValueError:
        return None
# hyperopt func   
def train_clu_hyperopt(params):
    # NOTE : consider distributing the data for large training scenario
        # train
    with mlflow.start_run(run_name="hyperopt_clu"):
        mlflow.log_params(params)
        model = train_cluster_model(training_data, params, CLU_SEED)
        evaluator = ClusteringEvaluator(featuresCol="features",
            metricName="silhouette", distanceMeasure="squaredEuclidean")
        validation_prediction = model.transform(validation_data)
        validation_silhouette = evaluator.evaluate(validation_prediction)
        validation_ll = validation_prediction\
            .withColumn("ll", f.log10(get_max_arg(f.col("probability"), f.col("prediction"))))\
                .agg(f.mean(f.col("ll")).alias("ll")).collect()[0]["ll"]
        mlflow.log_metric("validation_silhouette_score", validation_silhouette)
        mlflow.log_metric("validation_loglikelihood", validation_ll)
        # combined loss
        return {"loss":-(0.2*10**(validation_ll)+0.8*validation_silhouette), "status":STATUS_OK}

# hyperopt run
clu_params = fmin(
    fn=train_clu_hyperopt,
    space=clu_space,
    algo=tpe.suggest,
    max_evals=20)
# explore the tuning  
space_eval(clu_space, clu_params)    

# COMMAND ----------

# after manual eval, we see that there is better compromise between the sill and ll tradeoff
# clu_params = {"aggregationDepth":6.74, "k":7.69, "maxIter":196.81}

# COMMAND ----------

# retrain on full data with best params
with mlflow.start_run(run_name="refit_clu"):
    run = mlflow.active_run()
    mlflow.log_params(clu_params)
    clu = train_cluster_model(product_features, clu_params, CLU_SEED)
    mlflow.spark.log_model(clu, "clu_refit")
    path = f"runs:/{run.info.run_id}/clu_refit"
    mlflow.register_model(path, "clu_refit")

# COMMAND ----------

# load & predict
clu = mlflow.spark.load_model("models:/clu_refit/None").stages[0]
product_features = product_features\
    .join(clu.transform(product_features).select("product_id", "prediction"),on=["product_id"])
product_factors = product_factors\
    .join(product_features.select("product_id", "prediction"),on=["product_id"])
product_features.show()
product_factors.show()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.ml.feature import BucketedRandomProjectionLSH
buck_projection = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=100)
buck_model = buck_projection.fit(product_features)
cluster_centers = buck_model.approxSimilarityJoin(product_features, product_features, 10**3, distCol="EuclideanDistance")\
    .select(f.col("datasetA.product_id").alias("pid0"), f.col("datasetA.prediction").alias("pg0"),
        f.col("datasetB.product_id").alias("pid1"), f.col("datasetB.prediction").alias("pg1"), f.col("EuclideanDistance"))\
    .where((f.col("pg0")==f.col("pg1")) & (f.col("pid0")!=f.col("pid1")))\
    .groupBy(f.col("pid0"),f.col("pg0")).agg(f.mean(f.col("EuclideanDistance")).alias("dist"))\
    .withColumn("product_id", f.first("pid0").over(Window.partitionBy("pg0").orderBy("dist")))\
    .select("product_id", f.col("pg0").alias("prediction")).distinct()
cluster_centers.show()

# COMMAND ----------

from sklearn.manifold import TSNE
from pyspark.ml.feature import PCA

# data prep
dims = 4
pca = PCA(k=dims, inputCol="features", outputCol="factors")
pca_model = pca.fit(product_features)
pca_products = pca_model.transform(product_features)\
    .select("product_id", "prediction",
            *[vector_to_array(f.col("factors")).getItem(d).alias("pca"+str(d)) for d in range(dims)])\
    .persist()
print("PCA retained {0:.2f} % of variance.".format(np.sum(pca_model.explainedVariance)*100))

id = "product_id"
cluster = "prediction"
ploting_sample = product_factors.sample(fraction=1.0).toPandas()
features = [c for c in ploting_sample.columns if c not in set([id, cluster])]
tsne = TSNE(n_components=2, perplexity=160, learning_rate=1000.0, n_iter=1000)
tsne_transformed = pd.DataFrame(tsne.fit_transform(ploting_sample.loc[:,features]))
tsne_transformed.columns = ["component 0", "component 1"]
tsne_transformed["cluster"] = ploting_sample[cluster]

# plot
plt.figure(figsize=(20,10))
sns.scatterplot(data=tsne_transformed,
    x="component 0", y="component 1", hue="cluster", alpha=.9, palette="deep");

# COMMAND ----------

# TODO
# TOMORROW
# rectify timezone - DONE, rectify category ids - ORIGINAL IDS KEPT,
# clean code - data aquisition - DONE, customer model
# put everything together, design the workflows - OOPS
# construct targets

# FOLLOWING DAYS
# univariate - stats, sparsity, correlation with the target
# multivariate - intra correlation, feature clustering (consider this as part of the pipeline)

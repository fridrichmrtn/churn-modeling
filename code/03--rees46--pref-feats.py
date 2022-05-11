# Databricks notebook source
# MAGIC %md
# MAGIC # User churn model - Preference
# MAGIC 
# MAGIC ### Data loading

# COMMAND ----------

import pyspark.sql.functions as f
import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

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

user_filter = transactions.groupBy("user_id").agg(
    f.sum(f.col("implicit_feedback")).alias("implicit_feedback")).where(f.col("implicit_feedback")>4)
product_filter = transactions.groupBy("product_id").agg(
    f.sum(f.col("implicit_feedback")).alias("implicit_feedback")).where(f.col("implicit_feedback")>9)
transactions = transactions.join(user_filter.select("user_id"), on=["user_id"])\
    .join(product_filter.select("product_id"), on=["product_id"])

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
param_space = {
    "rank": hp.uniform("rank", 5, 25),
    "maxIter": hp.uniform("maxIter", 5, 35),
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
optimized_params = fmin(
    fn=train_reco_hyperopt,
    space=param_space,
    algo=tpe.suggest,
    max_evals=25)
# explore the tuning    
space_eval(param_space, optimized_params)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Re-fit on the complete dataset

# COMMAND ----------

# retrain on full data with best params
with mlflow.start_run(run_name="refit_reco"):
    run = mlflow.active_run()
    mlflow.log_params(optimized_params)
    transaction_reco = train_reco_model(transactions, optimized_params, RECO_SEED)
    mlflow.spark.log_model(transaction_reco, "tran_reco_refit")
    path = f"runs:/{run.info.run_id}/tran_reco_refit"
    mlflow.register_model(path, "tran_reco_refit")

# COMMAND ----------

# reload
transaction_reco = mlflow.spark.load_model("models:/tran_reco_refit/None")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explore learned characteristics

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
        *[f.col("features").getItem(d).alias("latent_factor"+str(d)) for d in dims])
product_factors.show()

# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture

CLU_SEED = -42

# param space
param_space = {
    "k": hp.uniform("rank", 2, 25),
    "maxIter": hp.uniform("maxIter", 25, 250),
    "aggregationDepth": hp.uniform("aggregationDepth", 2, 10),
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



# COMMAND ----------

from pyspark.ml.clustering import GaussianMixture
(training_data, validation_data) = product_factors.randomSplit([0.6, 0.4], seed=CLU_SEED)
va = VectorAssembler(inputCols=[c for c in training_data.columns if c not in "product_id"],
    outputCol="features")
training_data = va.transform(training_data).select("product_id", "features")
validation_data = va.transform(validation_data).select("product_id", "features")

CLU_SEED = -42
params = {"k":5, "maxIter":25, "aggregationDepth":2}
test_model = train_cluster_model(training_data, params, CLU_SEED)

# COMMAND ----------

clu_df = test_model.transform(validation_data)
clu_df.show()

# COMMAND ----------

clu_df.select(f.col("probability.0"))

# COMMAND ----------

from pyspark.sql.types import DoubleType
@f.udf(returnType=DoubleType()) 
def get_max_arg(arr, a):
    import numpy as np
    arr = np.array(arr)
    try:
        return float(arr[a])
    except ValueError:
        return None
# loglikelyhood calc on unseen data
clu_df.withColumn("ll", f.log10(get_max_arg(f.col("probability"), f.col("prediction")))).agg(f.mean(f.col("ll")).alias("ll")).collect()[0]["ll"]

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import udf,col
from pyspark.sql.types import StringType, FloatType
udf_star_desc = udf(lambda x:np.max(np.array(x)), FloatType())

test_model.transform(training_data).withColumn("rating_description",udf_star_desc(col("probability"))).show()

# COMMAND ----------

# try the like calc here


# COMMAND ----------

# define hyperopt space and funcs, run hyperopt
(training_data, validation_data) = product_factors.randomSplit([0.6, 0.4], seed=CLU_SEED)
def train_clu_hyperopt(params):
    # NOTE : consider distributing the data for large training scenario
        # train
    with mlflow.start_run(run_name="hyperopt_clu"):
        mlflow.log_params(params)
        model = train_cluster_model(training_data, params, CLU_SEED)
        silhouette_score = evaluator.evaluate(model.transform(validation_data))
        loglike = model.predictProbability(validation_data)

# COMMAND ----------

# define hyperopt space and funcs, run hyperopt
(training_data, validation_data) = transactions.randomSplit([0.6, 0.4], seed=reco_seed)
def train_reco_hyperopt(params):
    # NOTE : consider distributing the data for large training scenario
        # train
    with mlflow.start_run(run_name="hyperopt_reco"):
        mlflow.log_params(params)
        model = train_reco_model(training_data, params, reco_seed)
            # evaluate
        training_predictions = model.transform(training_data)
        validation_predictions = model.transform(validation_data)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="implicit_feedback",
                predictionCol="prediction")
        training_rmse = evaluator.evaluate(training_predictions)
        validation_rmse = evaluator.evaluate(validation_predictions)
        mlflow.log_metric("training_rmse", training_rmse)
        mlflow.log_metric("validation_rmse", validation_rmse) 
        return {"loss":validation_rmse, "status":STATUS_OK}
    
# run the trials, peek at the results
optimized_params = fmin(
    fn=train_reco_hyperopt,
    space=param_space,
    algo=tpe.suggest,
    max_evals=25)
# explore the tuning    
space_eval(param_space, optimized_params)

# COMMAND ----------

# now do the clustering

# product clustering

# rewrite this as hyperopt task

from pyspark.ml.evaluation import ClusteringEvaluator

evaluator = ClusteringEvaluator(featuresCol="features",
    metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette_scores = []
for k in range(2,25):
    gm = KMeans(k=k)
    model = gm.fit(transaction_reco.itemFactors)
    cluster_predictions = model.transform(transaction_reco.itemFactors)
    silhouette_score = evaluator.evaluate(cluster_predictions)
    silhouette_scores.append(silhouette_score)
    print("Silhouette with squared euclidean distance = " + str(silhouette_score))

# COMMAND ----------

# plan - do clustering
# plot tsne + clusters + centroid labels
# check fakin delta lake

# COMMAND ----------

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

id = "product_id"
features = [c for c in product_space.columns if c not in set([id])]

tsne = product_space_tsne(n_components=2, perplexity=500, n_iter=1000)
tsne_transformed = tsne.fit_transform(product_space.select(*features).toPandas())
tsne_transformed.columns = ["component 0", "component 1"]
plt.figure(figsize=(20,10))
sns.scatterplot(data=product_space_tsne,
    x="component 0", y="component 1",
    alpha=.6)

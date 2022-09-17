# Databricks notebook source
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
from mlflow.sklearn import load_model, save_model
import pyspark.sql.functions as f
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType

# COMMAND ----------

def _optimize_numeric_dtypes(df):
    float_cols = df.select_dtypes("float").columns
    int_cols = df.select_dtypes("integer").columns
    df[float_cols] = df[float_cols].\
        apply(pd.to_numeric, downcast="float")
    df[int_cols] = df[int_cols].\
        apply(pd.to_numeric, downcast="integer")
    return df

# NOTE: port this to spark
def _get_samples(df, in_cols, n_samples, random_state=1):
    df = _optimize_numeric_dtypes(df.toPandas())
    # clustering
    n_clusters = int(n_samples/5)
    pipeline = Pipeline([("scale", QuantileTransformer()),
        ("clu", AgglomerativeClustering(n_clusters=n_clusters))])
    labels = pipeline.fit_predict(df.loc[:,in_cols])
    labels = pd.DataFrame(labels, columns=["label"])
    fractions = labels.reset_index()\
        .groupby("label", as_index=False).count()\
            .rename(columns={"index":"count"})
    fractions["fractions"] = fractions["count"]/n_samples
    labels = labels.merge(fractions, on=["label"])   
    # sampling
    df = df.sample(n=n_samples, weights=labels.fractions,
        random_state=random_state)
    return spark.createDataFrame(df)

def _get_features(df):
    out_cols = ["user_id", "row_id", "time_step"]+\
        [c for c in df.columns if "target_" in c]
    return [c for c in df.columns if c not in out_cols]

def _get_data(dataset_name, time_step):
    df = spark.table(f"churndb.{dataset_name}_customer_model")
    train = df.where(f.col("time_step")>time_step)
    test = df.where(f.col("time_step")==time_step)
    test = test.orderBy("row_id").sample(fraction=1.0, seed=1)\
        .limit(1000).repartition(20)
    return {"train":train, "test":test}

def _get_explainer(model, df):
    df = _optimize_numeric_dtypes(df.toPandas())
    return shap.Explainer(model.predict,
        df.loc[:, _get_features(df)], max_evals=550)

def _compute_shap(explainer, df):
    def _get_shap(iterator, explainer=explainer):
        for X in iterator:
            # possibly add expected values here
            shap_instance = explainer(X.loc[:,_get_features(X)])
            shap_values = np.column_stack((X.loc[:,"row_id"].values,
                shap_instance.values, shap_instance.base_values))
            yield pd.DataFrame(shap_values)
    schema = StructType([f for f in df.schema.fields\
        if f.name in ["row_id"]+_get_features(df)]\
            +[StructField("base_values", FloatType(), False)])
    return df.mapInPandas(_get_shap, schema=schema)
    
def glue_shap(dataset_name, time_step, pipe):
    model = load_model(f"models:/{dataset_name}_{pipe}_{time_step}/None")
    shap_data = _get_data(dataset_name, time_step)
    explainer = _get_explainer(model, shap_data["train"]) 
    shap_values = _compute_shap(explainer, shap_data["test"])
    shap_values.withColumn("pipe", f.lit(pipe))\
      .write.saveAsTable(f"churndb.{dataset_name}_shap_values",
             mode="append")
    return None

# COMMAND ----------

# spark.sql("DROP TABLE IF EXISTS churndb.retailrocket_shap_values;")
# glue_shap("retailrocket", 0, "gbm_class")
# glue_shap("retailrocket", 0, "gbm_reg")

# COMMAND ----------

# spark.table("churndb.retailrocket_shap_values").where(f.col("pipe")=="svm_rbf_class")\
#     .toPandas().iloc[:,1:-1].sum(axis=1).sort_values()

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS churndb.rees46_shap_values;")
glue_shap("rees46", 0, "gbm_class")
glue_shap("rees46", 0, "gbm_reg")

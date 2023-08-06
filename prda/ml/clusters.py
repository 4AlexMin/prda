import time
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

__all__ = ['kmeans_spark', 'match_clusters']

def kmeans_spark(data: pd.DataFrame, k_: int, inertia: bool = False, input_cols: list = None) -> pd.DataFrame:
    """Local version of Spark K-Means clustering.

    Parameters
    ----------
    data : pd.Dataframe
        Data to perform "K-Means", columns as features and rows as samples.
    k_ : int
        number of clusters
    inertia : bool, optional
        Indicator of whether to return \
        `sum of squared distances to the nearest centroid for all points in the training dataset (equivalent to sklearn’s inertia.)`
        , by default False
    input_cols : list, optional
        column names for k-means input, by default None

    Returns
    -------
    pd.DataFrame
        _description_
    """
    if not input_cols:
        input_cols = data.columns.to_list()

    spark = SparkSession.builder\
        .master('local[*]')\
        .config("spark.driver.memory", '4g')\
        .appName(str(time.time()))\
        .getOrCreate()
        # .config('spark.default.parallelism', 300)\
        # .config('spark.sql.shuffle.partitions', 50)\
        # .config('spark.debug.maxToStringFields', 200)\
    vector_assembler = VectorAssembler()\
    .setInputCols(input_cols)\
    .setOutputCol('features')
    scales = spark.createDataFrame(data)
    scales = vector_assembler.transform(scales)

    km = KMeans().setK(k_).setFeaturesCol('features')
    km_model = km.fit(scales)

    results = km_model.transform(scales).toPandas()
    results.set_index(data.index, inplace=True)
    cost = km_model.summary.trainingCost
    spark.stop()
    
    if inertia:
        return results, cost
    else:
        return results


def match_clusters(alpha_clusters: dict, beta_clusters: dict, by: str='jaccard', with_similarities: bool=True)-> dict:
    """Pair the most similar clusters of two cluster results

    Parameters
    ----------
    alpha_clusters : dict
        {`cluster_id`: `elements_of_this_cluster`, } cluster results of one clustering method.
    beta_clusters : dict
        cluster results of the other clustering method.
    by : str, by default 'jaccard'
        similarities judgement indicator, `jaccard` indicates the `Jaccard Coefficient`

    Returns
    -------
    dict
        {`alpha_cluster_id`: ('most_similar_beta_cluster_id', 'similarity'), ..., `beta_cluster_id`: ('most_similar_alpha_cluster_id', 'similarity')}
    """

    similarities = pd.DataFrame(index=alpha_clusters.keys(), columns=beta_clusters.keys(), dtype=np.float32)
    for alpha_id in similarities.index:
        for beta_id in similarities.columns:
            if 'jaccard' in by.lower():
                coefficient = len(set(alpha_clusters[alpha_id]) & set(beta_clusters[beta_id])) / len(set(alpha_clusters[alpha_id]) | set(beta_clusters[beta_id]))
            else:
                raise KeyError(by)
            similarities.loc[alpha_id, beta_id] = coefficient
    matches = dict()
    for ax in [0, 1]:
        if with_similarities:
            indices = similarities.idxmax(axis=ax)
            max_sims = similarities.max(axis=ax)
            for idx_m in indices.index:
                matches[idx_m] = (indices[idx_m], max_sims[idx_m])
        else:
            matches.update(similarities.idxmax(axis=ax))
    return matches

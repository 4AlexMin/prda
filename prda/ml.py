""" The :mod:`prda.ml` contains machine learning algos.
"""

import time
import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

__all__ = ['kmeans_spark', 'match_clusters', 'evaluate_param_combinations']

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


def evaluate_param_combinations(X, y, algorithm, param_grid, scoring_metric='accuracy', cv=5, visualize_results=False):
    """
    Evaluate multiple combinations of hyperparameters for a given algorithm using cross-validation. 
    
    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values.
    algorithm : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
    param_grid : dict or list of dictionaries.
        Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, 
        or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
    scoring_metric : str, callable, list/tuple or dict, default='accuracy'
        A string (see scikit-learn documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
    cv : int or cross-validation generator, default=5
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 5-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
    visualize_results : bool, default=False
        Whether to plot a visualization of the cross-validation results.
    
    Returns:
    --------
    results : dict.
        A dictionary with the following keys:
            - 'params': the list of parameter combinations tested.
            - 'mean_test_score': the mean score over the cv folds for each parameter combination on the test set.
            - 'std_test_score': the standard deviation over the cv folds for each parameter combination on the test set.
            - 'mean_train_score': the mean score over the cv folds for each parameter combination on the train set.
            - 'std_train_score': the standard deviation over the cv folds for each parameter combination on the train set.
    """
    grid_search = GridSearchCV(algorithm, param_grid=param_grid, scoring=scoring_metric, cv=cv, return_train_score=True)
    grid_search.fit(X, y)

    cv_results = {
        'params': grid_search.cv_results_['params'],
        'mean_test_score': grid_search.cv_results_['mean_test_score'],
        'std_test_score': grid_search.cv_results_['std_test_score'],
        'mean_train_score': grid_search.cv_results_['mean_train_score'],
        'std_train_score': grid_search.cv_results_['std_train_score']
    }

    if visualize_results:
        length = int(len(cv_results['mean_test_score'])/10) + 6
        plt.figure(figsize=(10, length))
        plt.errorbar(cv_results['mean_train_score'], range(len(cv_results['params'])), xerr=cv_results['std_train_score'], fmt='o-', capsize=5, label='train')
        plt.errorbar(cv_results['mean_test_score'], range(len(cv_results['params'])), xerr=cv_results['std_test_score'], fmt='o-', capsize=5, label='test')
        plt.yticks(range(len(cv_results['params'])), [str(p) for p in cv_results['params']])
        plt.xlabel(scoring_metric)
        plt.ylabel('Hyperparameters')
        plt.legend()
        plt.show()
        
    best_idx = np.argmax(cv_results['mean_test_score'])
    print("Best combination: ", cv_results['params'][best_idx])
    print("Best test-score: {:.4f}".format(cv_results['mean_test_score'][best_idx]), "; Best train-score: {:.4f}".format(cv_results['mean_train_score'][best_idx]))
    
    return cv_results


   


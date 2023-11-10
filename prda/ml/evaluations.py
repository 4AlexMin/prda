import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score

__all__ = ['test_algorithm_performance', 'evaluate_param_combinations']



def test_algorithm_performance(algorithm, X, y, scoring_metric='accuracy', cv=10, fit_model=True):
    """
    Test the performance of a given algorithm on a given dataset using 10-fold cross-validation.

    Parameters:
    -----------
    algorithm : object
        The algorithm to be tested. It should have a fit() method for training and a predict() method for making predictions.
    X : array-like
        The input features of the dataset.
    y : array-like
        The target labels of the dataset.
    scoring_metric : str, optional
        The scoring metric to use. Possible values are 'accuracy', 'f1', 'mcc' (Matthew's correlation coefficient), 'kappa' (Cohen's kappa).
    cv : int or cross-validation generator, default=10
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - None, to use the default 10-fold cross-validation,
            - integer, to specify the number of folds.
            - An object to be used as a cross-validation generator.
            - An iterable yielding train/test splits.
    fit_model : bool, optional
        Whether to fit the model on the entire dataset before cross-validation. Default is True.

    Returns:
    --------
    float
        The average score across the 10 folds based on the specified scoring metric.
    """
    
    if fit_model:
        # Fit the algorithm on the entire dataset
        algorithm.fit(X, y)
    
    # Define scoring methods based on scoring_metric
    scoring_methods = {
        'accuracy': 'accuracy',
        'f1': 'f1_macro',
        'mcc': make_scorer(matthews_corrcoef),
        'kappa': make_scorer(cohen_kappa_score),
    }
    
    if scoring_metric not in scoring_methods:
        raise ValueError("Invalid scoring_metric. Supported values are 'accuracy', 'f1', 'mcc', 'kappa'.")
    
    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(algorithm, X, y, cv=cv, scoring=scoring_methods[scoring_metric])

    # Calculate the average score based on the specified scoring metric
    avg_score = cv_scores.mean()

    return avg_score




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
    scoring_methods = {
        'accuracy': 'accuracy',
        'f1': 'f1_macro',
        'mcc': make_scorer(matthews_corrcoef),
        'kappa': make_scorer(cohen_kappa_score),
    }
    if scoring_metric not in scoring_methods:
        raise ValueError("Invalid scoring_metric. Supported values are 'accuracy', 'f1', 'mcc', 'kappa'.")


    grid_search = GridSearchCV(algorithm, param_grid=param_grid, scoring=scoring_methods[scoring_metric], cv=cv, return_train_score=True)
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
    print("Best test-score: {:.4f}".format(cv_results['mean_test_score'][best_idx]))
    
    return cv_results


   


""" The :mod:`prda.prep` contains preprocessing methodologies.
"""

from sklearn import preprocessing
import pandas as pd
import numpy as np

__all__ = ['normalization', 'convert_df', 'drop_first_n', 'pca', 'handle_missing_data', 'select_continuous_variables', 'apply_linear_func']
        
def normalization(data: np.ndarray, min_: float = 0, max_: float = 1.0, bias: float = 0)-> np.ndarray:
    """
    Scaling data to lie between a given minimum and maximum value (MinMaxScaler).

    [Note] that this is a `feature-wise` scaler, rather than sample-wise which is the case in Sci-kit Learn

    Considering all datas `as one dataset`.

    The transformation is given by::

        X_std = (X - X.min) / (X.max - X.min)
        X_scaled = X_std * (max_ - min_) + min_

    Parameters
    ----------
    data : numpy.ndarray
        ndarray as original shape
    """
    data = np.asarray(data)
    datalen = 1
    for i in data.shape:
        if i != 0:
            datalen = datalen * i
    scaler = preprocessing.MinMaxScaler(feature_range=(min_, max_))
    data_norm = scaler.fit_transform(data.reshape(datalen, 1)).reshape(data.shape) + bias
    return data_norm

def convert_df(df: pd.DataFrame, method: str = 'wide2long')-> pd.DataFrame:
    """Convert wide-form `pd.DataFrame` to long-form, vice versa.

    Parameters
    ----------
    df : pd.DataFrame
        `pd.DataFrame` to be transformed
    method : str, optional
        specifier of the transformation purpose, by default 'wide2long'

    Returns
    -------
    pd.DataFrame
    """
    result = pd.DataFrame(columns=['val', 'attr_row', 'attr_col'])
    curr_idx = 0
    for row in df.index:
        for col in df.columns:
            if pd.notna(df.loc[row, col]):
                result.loc[curr_idx] = [df.loc[row, col], row, col]
                curr_idx += 1
    return result

def drop_first_n(data: pd.DataFrame, n_first: int = 3, inplace: bool = False):
    """Drop `n` maximum values in `data`, column-wisely.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    n_first : int, optional
        `n_first` maximum values to drop, by default 3
    """
    if inplace:
        results = data
    else:
        results = data.copy()
    for _ in range(n_first):
        max_vals = results.max()
        results.replace(max_vals, np.nan, inplace=True)
    
    if not inplace:
        return results


def pca(data: np.ndarray, npcs: int, return_variance: bool = False):
    """This implementation takes as input a numpy array data with shape (n_samples, n_features) and an integer npcs specifying how many principal components to return. It first centers the data by subtracting the mean of each feature, then computes the covariance matrix. Next, it computes the eigenvectors and eigenvalues of the covariance matrix and sorts the eigenvectors in descending order of eigenvalues. It keeps the top npcs eigenvectors and computes the variance explained by each component. Finally, it projects the data onto the top principal components and returns the principal components and variance explained as a tuple.

    Parameters
    ----------
    data : np.ndarray
        Input data as a numpy array with shape (n_samples, n_features).
    npcs : int
        Number of top principal components to return.
    return_variance : bool
        Whether outputs variance for each principal components, by default False.
    Returns
    -------
    tuple (if return_variance is True)
        Tuple of (principal components, variance explained by each component).
    np.ndarray
        Principal components.
    """
    # Center data
    data = data - np.mean(data, axis=0)

    # Compute covariance matrix
    cov = np.cov(data, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eig(cov)

    # Sort eigenvectors in descending order of eigenvalues
    sort_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[:, sort_indices]

    # Keep top npcs components
    eigvecs = eigvecs[:, :npcs]

    # Compute variance explained by each component
    var_exp = eigvals / np.sum(eigvals)

    # Project data onto principal components
    proj_data = np.dot(data, eigvecs)
    if return_variance:
        return proj_data, var_exp
    else:
        return proj_data


def handle_missing_data(df, row_thresh=None, col_thresh=None, method='mean', output_stats=False):
    """
    Function to handle continuous missing data in a Pandas DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with missing values
    row_thresh (float): Threshold for dropping rows with too many NaNs (default=None, i.e., do not drop rows)
    col_thresh (float): Threshold for dropping columns with too many NaNs (default=None, i.e., do not drop columns)
    method (str): Method to handle missing data - 'mean', 'median', 'mode', or 'fill' (default='mean')
    output_stats (bool): Whether to return statistics about missing values (default=False)
    
    Returns:
    pd.DataFrame: Output DataFrame after processing missing values
    """
    
    # Copy input DataFrame to avoid modifying the original DataFrame
    df_out = df.copy()
    
    # Drop rows with too many NaNs
    if row_thresh is not None:
        drop_rows = (df_out.isna().sum(axis=1) / df_out.shape[1]) > row_thresh
        df_out = df_out.drop(index=df_out[drop_rows].index)
    
    # Drop columns with too many NaNs
    if col_thresh is not None:
        drop_cols = (df_out.isna().sum(axis=0) / df_out.shape[0]) > col_thresh
        df_out = df_out.drop(columns=df_out.columns[drop_cols])
    
    # Handle missing data
    if method == 'mean':
        df_out = df_out.fillna(df_out.mean(numeric_only=True))
    elif method == 'median':
        df_out = df_out.fillna(df_out.median(numeric_only=True))
    elif method == 'mode':
        df_out = df_out.fillna(df_out.mode(numeric_only=True).iloc[0])
    elif method == 'fill':
        df_out = df_out.fillna(method='ffill').fillna(method='bfill')
    
    # Return output DataFrame and statistics (if specified)
    if output_stats:
        stats = {
            'total_missing_values': df.isna().sum().sum(),
            'percentage_missing_values': round((df.isna().sum().sum() / df.size) * 100, 2),
            'total_rows_dropped': len(df.index) - len(df_out.index) if row_thresh is not None else 0,
            'total_cols_dropped': len(df.columns) - len(df_out.columns) if col_thresh is not None else 0
        }
        print(stats)
    return df_out


def select_continuous_variables(data, unique_threshold=10):
    """exclude columns (discrete variables in numerical format) by checking if the number of unique values is less than a certain threshold


    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The input data.
    unique_threshold : int, optional
        number of unique values larger than `unique_threshold` will be considered as continuous, by default 10

    Returns
    -------
    pd.DataFrame or np.ndarray
        subset of data containing only continuous variables
    """
    if isinstance(data, pd.DataFrame):
        # Get column data types
        dtypes = data.dtypes

        # Find columns with continuous variables
        continuous_cols = []
        for col in data.columns:
            if np.issubdtype(dtypes[col], np.number) and data[col].nunique() >= unique_threshold:
                continuous_cols.append(col)

        # Create new DataFrame with continuous columns only
        data_continuous = data[continuous_cols].copy()

        return data_continuous

    elif isinstance(data, np.ndarray):
        # Get number of unique values for each column
        unique_counts = np.apply_along_axis(lambda x: np.unique(x).size, axis=0, arr=data)

        # Find columns with continuous variables
        continuous_cols = np.where((data.dtype == np.float64) & (unique_counts >= unique_threshold))[0]

        # Create new ndarray with continuous columns only
        data_continuous = data[:, continuous_cols].copy()

        return data_continuous
    
    else:
        raise ValueError("Input data must be a Pandas DataFrame or NumPy ndarray")


def apply_linear_func(X, weights=None, bias=0):
    """
    Applies a linear function to an np.ndarray row-wisely.

    Parameters
    ----------
    X : np.ndarray
        The input array to apply the linear function to.
    weights : np.ndarray, optional
        The weights to use for the linear function. Should have the same shape as X along the specified axis.
    bias : int, optional
        The bias to use for the linear function, by default 0.

    Returns
    -------
    np.ndarray
    """
    if weights is None:
        weights = np.ones(X.shape[1])


    weights = np.asarray(weights)
    bias = np.asarray(bias)

    return np.sum(X * weights, axis=1) + bias


""" The :mod:`prda.prep` contains preprocessing methodologies.
"""

from sklearn import preprocessing
import numpy as np

__all__ = ['normalization']
        
def normalization(data: np.ndarray, min_: float = 0, max_: float = 1.0)-> np.ndarray:
    """
    Scaling data to lie between a given minimum and maximum value.

    [Note] that this is a `feature-wise` scaler, rather than sample-wise which is the case in Sci-kit Learn

    Considering all datas `as one dataset`.

    The transformation is given by::

        X_std = (X - X.min) / (X.max - X.min)
        X_scaled = X_std * (max_ - min_) + min_

    Parameters
    ----------
    data : numpy.ndarray
    """
    datalen = 1
    for i in data.shape:
        if i != 0:
            datalen = datalen * i
    scaler = preprocessing.MinMaxScaler(feature_range=(min_, max_))
    data_norm = scaler.fit_transform(data.reshape(datalen, 1)).reshape(data.shape)
    return data_norm
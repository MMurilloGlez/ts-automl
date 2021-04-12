"""A module containing all preprocessing steps for time series analysis"""

"""

Module containing all necessary steps to convert an univariate input time series
with a constant frequency into a multivariate dataframe with lags and features,
so that conventional regression techniques can be applied.

Parameters
----------
data: pd.Dataframe
    Input time series data with constant frequency. interpolated if necessary by
    the data import functions.
val_split: float {0.1, other}
    Validation split size. Can be either a number indicating number of samples
    for validation or a decimal, indicating proportion of values for validation.
lag_num: int
    Number of lags to compute from the time series
features: list of str
    Features to compute for each lag or set of lags
rolling_window: int
    Size of rolling window for features such as moving average

Returns
-------
pd.Dataframe x2
    Two dataframes containing the generated features and lags. One containing 
    the validation data and one containing the train data.

Raises
------

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def ts_split(data, val_size=0.1):
    """split training data into train and val"""

    """
    Splits input dataframe into train and test. Works for both univariate and
    multivariable dataframes, in case the split is done after generating the 
    features and lags.

    Parameters
    ----------
    data: pd.Dataframe
        Input dataframe containing the time series. Index must be a datetime
        object.
    val_size: float {0.1, other}
        Number of values dedicated to the validation portion of the dataset. Can
        either be a proportion of dataset size or a set number of points.
    
    Returns
    -------
    pd.Dataframe x2
        Two dataframes, each containing the proportion of the series dedicated 
        to the train and validation of the model respectively.
    
    """

    ts_train, ts_val = train_test_split(data, test_size=val_size, shuffle=False)
    return(ts_train, ts_test)

def
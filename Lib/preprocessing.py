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
import scipy
import pywt

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

# Extract Features -------------------------------------------------------------

def lag_feature(data, num):
    """Compute lag corresponding to num argument."""

    """
    Computes the lag component corresponding to a number of periods in the
    past.
    
    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.
        
    Returns
    -------
    float
        Corresponds to lag component to be calculated.
    """
    return data[-num]


def minute_feature(date, num):
    """One Hot encoding for minute of hour feature"""

    """
    One Hot encoding for the minute feature of a given sample. takes a 1 as the
    value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """
    minute = date.minute
    if minute == num:
        return 1
    else:
        return 0


def hour_feature(date, num):
    """One Hot encoding for hour of day feature"""

    """
    One Hot encoding for the hour feature of a given sample. takes a 1 as the
    value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """
    if hour == num:
        return 1
    else:
        return 0


def dayofweek_feature(date, num):
    """One Hot encoding for day of week feature"""

    """
    One Hot encoding for the day of week feature of a given sample. takes a 1 as
    the value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """    
    dayofweek = date.dayofweek
    if dayofweek == num:
        return 1
    else:
        return 0


def day_feature(date, num):
    """One Hot encoding for day of month feature"""

    """
    One Hot encoding for the day of month feature of a given sample. takes a 1
    as the value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """ 
    day = date.day
    if day == num:
        return 1
    else:
        return 0
    

def month_feature(date, num):
    """One Hot encoding for month of year feature"""

    """
    One Hot encoding for the month of year feature of a given sample. takes a 1
    as the value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """ 
    month = date.month
    if month == num:
        return 1
    else:
        return 0

    
def quarter_feature(date, num):
    """One Hot encoding for fiscal quarter feature"""

    """
    One Hot encoding for the fiscal quarter feature of a given sample. takes a 1
    as the value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """ 
    quarter = date.quarter
    if quarter == num:
        return 1
    else:
        return 0
    

def weekofyear_feature(date, num):
    """One Hot encoding for week of year feature"""

    """
    One Hot encoding for the week of year feature of a given sample. takes a 1
    as the value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """ 
    weekofyear = date.weekofyear
    if weekofyear == num:
        return 1
    else:
        return 0

    
def weekend_feature(date, num):
    """One Hot encoding for weekend feature"""

    """
    One Hot encoding for the weekend feature of a given sample. takes a 1
    as the value if the feature is computed, 0 otherwise.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.  

    Returns
    -------
    int
        1 or 0 
    """ 
    dayofweek = date.dayofweek
    if dayofweek == 5 or dayofweek == 6:
        return 1
    else:
        return 0


def mean_feature(data, rolling_window, num):
    """Computes the mean for a given lag and its rolling window"""

    """
    Given a set number of lags in the past, a set rolling window and the input 
    dataset, computes the rolling average mean for the given window.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.
    rolling_window: list of int
        rolling window over which to calculate the feature.
    
    Returns
    -------
    float
        calculated feature
    """
    if num == 1:
        return np.mean(data[-rolling_window:])
    else:
        return np.mean(data[-(rolling_window-1+num):-(num-1)])

    
def std_feature(data, rolling_window, num):
    """Computes the standard deviation for a given lag and its rolling window"""

    """
    Given a set number of lags in the past, a set rolling window and the input 
    dataset, computes the rolling standard deviation for the given window.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.
    rolling_window: list of int
        rolling window over which to calculate the feature.
    
    Returns
    -------
    float
        calculated feature
    """
    if num == 1:
        return np.std(data[-rolling_window:], ddof=1)
    else:
        return np.std(data[-(rolling_window-1+num):-(num-1)], ddof=1)


def max_feature(data, rolling_window, num):
    """Computes the maximum value for a given lag and its rolling window"""

    """
    Given a set number of lags in the past, a set rolling window and the input 
    dataset, computes the rolling maximum value for the given window.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.
    rolling_window: list of int
        rolling window over which to calculate the feature.
    
    Returns
    -------
    float
        calculated feature
    """
    if num == 1:
        return np.max(data[-rolling_window:])
    else:
        return np.max(data[-(rolling_window-1+num):-(num-1)])


def min_feature(data, rolling_window, num):
    """Computes the minimum value for a given lag and its rolling window"""

    """
    Given a set number of lags in the past, a set rolling window and the input 
    dataset, computes the rolling minimum value for the given window.

    Parameters
    ----------
    data: pd.Dataframe
        input dataframe containing the time series.
    num: int
        lag number to calculate.
    rolling_window: list of int
        rolling window over which to calculate the feature.
    
    Returns
    -------
    float
        calculated feature
    """
    if num == 1:
        return np.min(data[-rolling_window:])
    else:
        return np.min(data[-(rolling_window-1+num):-(num-1)])



"""
Module containing metrics to evaluate the performance of the different models

This module contains metric with which to evaluate the performance of the time
series prediction pipelines in the library. It contains both absolute metrics, 
such as mse or mape, as well as a measure of error with respect to a naïve 
model.
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sktime.forecasting.naive import NaiveForecaster

#--Traditional------------------------------------------------------------------

def mse(y_true, y_pred):
    """mean square error calculation.

    Function calculating the mean squared error between the predicted values and
    the real ones (test)

    Parameters
    ----------
    y_pred: np.array
        An array containing all predicted values output by the model
    y_true: np.array
        Actual values to test accuracy of prediction

    Returns
    -------
    float
        value of the error metric
    """
    error = metrics.mean_squared_error(y_true, y_pred, squared= True)

    return error

def rmse(y_true, y_pred):
    """root mean square error calculation"""
    error = metrics.mean_squared_error(y_true, y_pred, squared= False)

    return error

def rsquare(y_true, y_pred):
    """R-Square error calculation"""
    error = metrics.r2_score(y_true, y_pred)

    return error

def mape(y_true, y_pred):
    """Mean absolute percentage error calculation"""
    error = metrics.mean_absolute_percentage_error(y_true, y_pred)

    return error

def exp_var(y_true, y_pred):
    """Mean absolute percentage error calculation"""
    error = metrics.explained_variance_score(y_true, y_pred)
    
    return error

#--With respect to naive-----------------------------------------------------------------------------

def relative_error(y_pred, y_test):
    """Relative error improvement with respect to a naïve prediction model

    Returns
    -------
    float
        Percentage of improvement with respect to naive model
    """
    naive = NaiveForecaster(strategy='mean', window_length=window)
    return error
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


def switch_abs_error(value, y_true, y_pred):
    """Switch for selecting error metrics for prediction tasks
    
    
    A switch, that takes as input the list of error metrics to be calculated for
    a forecasting task and calls the corresponding calculations.

    Parameters
    ----------
    value: list of str
        list of the error metric to calculate
    y_true: np.array
        array of true values from the test dataset
    y_pred: np.array
        array of predicted values from the ML model
    
    Returns: function call
        function call to the corresponding metric or metric to be calculated for
        the forecasting task
    """

    return {
        'mse': lambda : mse(y_true, y_pred),
        'rmse': lambda : rmse(y_true, y_pred),
        'mape': lambda : mape(y_true, y_pred),
        'r2': lambda : rsquare(y_true, y_pred),
        'exp_var': lambda : exp_var(y_true, y_pred)
    }.get(value)()

#--With respect to naive-----------------------------------------------------------------------------

def relative_error(y_true, y_pred, y_naive):
    """Relative error improvement with respect to a naïve prediction model

    Takes the original time series as an input as well as y_pred and y_true. 
    Performs a naive forecast on the time series and returns the relative mse 
    error of the ML model with respect to the naive one.

    Parameters
    ----------
    y_true: np.array

    y_pred: np.array

    X: pd.Series

    horizon: list of int

    Returns
    -------
    float
        Percentage of improvement with respect to naive model
    """
    error_naive = mse(y_true, Y_naive)
    error_pred = mse(y_true, y_pred)
    error = 100*(error_pred/error_naive)

    return error
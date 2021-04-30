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

    return error

def rmse(y_true, y_pred):
    """root mean square error calculation"""

    return error

def rsquare(y_true, y_pred):
    """R-Square error calculation"""

    return error

def mape(y_true, y_pred):
    """Mean absolute percentage error calculation"""

    return error

def smape(y_pred, y_test):
    """Symmetrical mean absolute percentage error calculation"""

    return error

def relative_error(y_pred, y_test):
    """Relative error improvement with respect to a naïve prediction model

    Returns
    -------
    float
        Percentage of improvement with respect to naive model
    """

    return error
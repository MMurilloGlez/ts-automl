"""
Module containing different metrics to evaluate the performance of the models

Contains both traditional metrics, such as MSE  or MAPE, as well as percentage 
errors with respect to a given naïve baseline model
"""
import numpy as np
import pandas as pd


def mse(y_pred, y_test):
    """
    Mean square error calculation


    Given a y_pred array and a y_test array with same dimesnion and shape, returns
    the mean square error of the prediction.

    Parameters
    ----------
    y_pred: np.array
        Array containing output from the model predictions
    y_test: np.array
        Array containing actual measured values for predictions.

    Returns
    -------
    float
        Absolute value of the mean square error 
    """

    return error


def rmse(y_pred, y_test):
    """
    Root mean square error calculation

    Given a y_pred array and a y_test array with same dimesnion and shape, 
    returns the root mean square error of the prediction.

    Returns
    -------
    float
        Absolute value of the root mean square error 
    
    """

    return error


def mape(y_pred, y_test):
    """
    mean absolute percentage error calculation

    Given a y_pred array and a y_test array with same dimesnion and shape, 
    returns the mean absolute percentage error of the prediction.

    Returns
    -------
    float
        Absolute value of the mean absolute percentage error 
    
    """

    return error


def smape(ypred, y_test):
    """
    Symmetrical mean absolute percentage error calculation

    Given a y_pred array and a y_test array with same dimension and shape, 
    returns the symmetrical mean absolute percentage error of the prediction.

    Returns
    -------
    float
        Absolute value of the symmetrical mean absolute percentage error 
    
    """

    return error


def R2(y_pred, y_test):
    """
    R-squarec error calculation

    Given a y_pred array and a y_test array with same dimesnion and shape, 
    returns the R-Squared error of the prediction.

    Returns
    -------
    float
        Absolute value of the R2 error
    
    """

    return error

def naive_perc(y_pred, y_test):
    """
    Error percentage with respect to a naïve prediction model

    Given a y_pred array, compares the array with that given by a naïve 
    prediction model

    Returns
    -------
    float
        Absolute percentage of error with respect to the naïve model
    
    
    """
    return error
"""Module containing the sklearn pipelines for time series forecasting"""

"""
This module contains the sklearn pipelines for the different levels of forecast
'difficulty'. Three will be proposed, a fast, a balanced and a slow prediction,
each sacrificing processing time for forecasting accuracy.

Parameters
----------
forecast_type: str {'slow','balanced','fast'}
    type of forecasting desired, goal is to automate this in terms of desired 
    processing time, frequency and time window

Returns
-------
sklearn.pipeline
    sklearn pipeline containing all steps including preprocessing, feature
    selection and prediction for the desired forecast type.

"""
from sklearn.multioutput import
from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import ReversibleImputer


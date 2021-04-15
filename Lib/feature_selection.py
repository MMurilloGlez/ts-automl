"""Feature selection methods for time series forecasting."""

"""
Module containing different feature selection models to simplify and decrease
processing time for the prediction models in the library. All wrapped in 
sklearn wrappers so they ¡can be implemented into pipelines.

Parameters
----------

preprocessor: str
    Type of feature selection to use in the prediction. i.e. PCA, Tree
data: pd.Dataframe
    Dataframe containing the full feature set and lags as well as the base time
    series for the training data
n_feat: int
    Number of features to preserve from the dataframe

Returns
-------
pd.Dataframe
    Dataframe containg the n_feat most important features for the prediction, on
    which to train and apply the forecasting models

"""
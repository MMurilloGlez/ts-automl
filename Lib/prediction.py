"""Module containing all estimators and predictors for forecasting"""


"""
Module containing the different models for time series forecasting. Including 
simple generalized linear models, Gradient boosting random forest approaches
and convolutional and recurrent neural networks.

Parameters
----------
data: pd.Dataframe x2
    A pandas dataframe, containing the lagged time series with all generated 
    features. One corresponding to the test portion and another to validation.

Returns
-------
pred: pd.Dataframe
    univariate time series containing the prediction for the desired time
    horizon and frequency

"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from tf.keras.wrappers.scikit_learn import KerasRegressor
from lightgbm.sklearn import LGBMRegressor

class glm_forecaster(BaseEstimator, RegressorMixin):
    """ A sklearn-style wrapper for statsmodels glm regressor """

    """
    GLM model from statsmodels wrapped into sklearn-friendly wrapper. so fit and
    predict can be called simply.

    Parameters
    ----------
    sm.GLM: statsmodels Model
        GLM model from the statsmodels library
    Returns
    -------
    sklearn model:
        Model adapted so it can be included in sklearn pipelines and used with 
        sklearn's own functions.
    """

    def __init__(self, model_class=sm.GLM, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


class LSTM_forecaster(KerasRegressor):
    pass


class TCN_forecaster(KerasRegressor):
    pass


class Fast_NN_forecaster(KerasRegressor):
    pass


class LGBM_forecaster(LGBMRegressor):
    pass


class knn_forecaster(RegressorMixin, BaseEstimator):
    pass

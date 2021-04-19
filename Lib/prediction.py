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

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tcn import TCN
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.linear_model import TweedieRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm.sklearn import LGBMRegressor

class glm_forecaster(BaseEstimator, RegressorMixin):
    """ A sklearn-style wrapper for statsmodels glm regressor 

    
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


def scaler():
    pass


def LSTM_Model_gen():
    """
    LSTM recurrent neural network construction for forecasting with keras

    Long Short Term Memory (LSTM) neural net construction. Consisting of seven 
    LSTM layers with 256 neurons each, all of them having relu activation 
    function. Last layer consists of one single neuron also without activation
    function.

    """
    model = Sequential()
    model.add(LSTM(256, activation='relu', input_shape=(1,40),
                   return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam', 
                  metrics = tf.keras.metrics.MeanAbsolutePercentageError())

def Mixed_Model_gen():
    """
    Mixed recurrent neural net model construcion for forecasting with keras

    Consists of a Time Convolutional Network (TCN) layer with droput rate of 0.3
    and batch normalization, followed by an LSTM layer with 256 neurons. Both 
    have relu activation function. Last layer consists of one single neuron also
    without activation function.

    """
    model2 = Sequential()
    model2.add(TCN(input_shape=(1,40),  return_sequences=True, 
                   use_batch_norm=True, activation = 'relu', padding = 'causal', 
                   dropout_rate = 0.3))
    model2.add(LSTM(256, activation='relu', return_sequences=True))
    model2.add(Dense(1))

    model2.compile(loss='mape', optimizer='adam',
                   metrics = tf.keras.metrics.MeanAbsolutePercentageError())


def TCN_Model_gen():
    """
    Time convolutional neural net model construction for forecasting with keras

    Consists of two TCN layers, with causal padding, relu activation and each
    having 0.3 dropout rate. Last layer consists of one single neuron also
    without activation function.

    """
    model3 = Sequential()
    model3.add(TCN(input_shape=(1,40),  return_sequences=True, 
                   use_batch_norm=True, activation = 'relu', padding = 'causal',
                   dropout_rate = 0.3))
    model3.add(TCN(return_sequences=True, use_batch_norm=True, 
                   activation = 'relu', padding = 'causal', dropout_rate = 0.3))
    model3.add(Dense(1, activation = 'relu'))

    model3.compile(loss='mape', optimizer='adam', 
                   metrics = tf.keras.metrics.MeanAbsolutePercentageError())


LSTM_Model = KerasRegressor(build_fn=LSTM_Model_gen, verbose=1, epochs=50)
TCN_Model = KerasRegressor(build_fn=TCN_Model_gen, verbose=1, epochs=50)
Mixed_Model = KerasRegressor(build_fn=Mixed_Model_gen, verbose=1, epochs=10)
KNN_Model = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
LGB_Model = LGBMRegressor()
GLM_Model = TweedieRegressor(power=3,, fit_intercept=True, verbose=1)

#-------------------------------------------------------------------------------

def recursive_forecast(y, model, window_length, feature_names, rolling_window,
                       n_steps):
    """
    Forecasting function which applies a given model recursively in a series

    Function, taking as an input the lagged time series with created features
    (Feature selection preprocessing optional) and applies recursive forecasting
    using a pre-trained model. Next time step is predicted with repect to the
    last prediction made.


    Parameters
    ----------
    y: pd.Series holding the input time-series to forecast
    model: pre-trained machine learning model
    lags: list of lags used for training the model
    n_steps: number of time periods in the forecasting horizon
    step: forecasting time period
   
    Returns
    -------
    fcast_values: pd.Series with forecasted values
    """
   
    # get the dates to forecast
    last_date = y.index[-1] + datetime.timedelta(minutes=15) 
    target_range = pd.date_range(last_date, periods=n_steps, freq=freq) 
    target_value = np.arange(n_steps, dtype = float)       
    max_rol = max(rolling_window, default=1)     
    lags = list(y.iloc[-(window_length+(max_rol-1)):,0].values)
    ####
    
    
    for i in range(n_steps):                                  
        train = create_features(feature_names, lags, target_range[i]) 
        new_value = model.predict(pd.DataFrame(train).transpose()) 
        target_value[i] = new_value[0]                             
        lags.pop(0)                                                
        lags.append(new_value[0])                                   
                                                        

    return target_value
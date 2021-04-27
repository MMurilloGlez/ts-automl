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
import datetime

import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.metrics import MeanAbsolutePercentageError
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

    def __init__(self, model_class=sm.GLM,
                 fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X, family=sm.families.Gaussian())
        self.results_ = self.model_.fit()
    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


def LSTM_Model_gen(n_feat):
    """
    LSTM recurrent neural network construction for forecasting with keras

    Long Short Term Memory (LSTM) neural net construction. Consisting of seven 
    LSTM layers with 256 neurons each, all of them having relu activation 
    function. Last layer consists of one single neuron also without activation
    function.

    """
    model1 = Sequential()
    model1.add(LSTM(256, activation='relu', input_shape=(1,n_feat),
                   return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(Dense(1))
    
    model1.compile(loss='mape', optimizer='adam',
                  metrics = MeanAbsolutePercentageError())
    return(model1)

def Mixed_Model_gen(n_feat):
    """
    Mixed recurrent neural net model construcion for forecasting with keras

    Consists of a Time Convolutional Network (TCN) layer with droput rate of 0.3
    and batch normalization, followed by an LSTM layer with 256 neurons. Both 
    have relu activation function. Last layer consists of one single neuron also
    without activation function.

    """
    model2 = Sequential()
    model2.add(TCN(input_shape=(1,n_feat),  return_sequences=True,  
                   use_batch_norm=True, activation = 'relu', padding = 'causal', 
                   dropout_rate = 0.3))
    model2.add(LSTM(256, activation='relu', return_sequences=True))
    model2.add(Dense(1))

    model2.compile(loss='mape', optimizer='adam',
                   metrics = MeanAbsolutePercentageError())
    return(model2)


def TCN_Model_gen(n_feat):
    """
    Time convolutional neural net model construction for forecasting with keras

    Consists of two TCN layers, with causal padding, relu activation and each
    having 0.3 dropout rate. Last layer consists of one single neuron also
    without activation function.

    """
    model3 = Sequential()
    model3.add(TCN(input_shape=(1,n_feat),  return_sequences=True, 
                   use_batch_norm=True, activation = 'relu', padding = 'causal',
                   dropout_rate = 0.5))
    model3.add(TCN(return_sequences=True, use_batch_norm=True, 
                   activation = 'relu', padding = 'causal', dropout_rate = 0.5))
    model3.add(Dense(1, activation = 'relu'))

    model3.compile(loss='mape', optimizer='adam', 
                   metrics = MeanAbsolutePercentageError())
    return(model3)

def LSTM_Model(n_feat):
    return KerasRegressor(build_fn=(lambda: LSTM_Model_gen(n_feat)), 
                          verbose=1,  batch_size=16,
                          epochs=50)
def TCN_Model(n_feat):
    return KerasRegressor(build_fn=(lambda: TCN_Model_gen(n_feat)), 
                          verbose=1,  batch_size=16, 
                          epochs=100)
def Mixed_Model(n_feat):
    return KerasRegressor(build_fn=(lambda: Mixed_Model_gen(n_feat)), 
                          verbose=1,  batch_size=16,
                          epochs=10)
KNN_Model = KNeighborsRegressor(n_neighbors=20, n_jobs=-1)
LGB_Model = LGBMRegressor()
GLM_Model = glm_forecaster()
scaler = MinMaxScaler(feature_range=(0,1))

#-------------------------------------------------------------------------------
def switch_features(value, lags_data, date, rolling_window, num):
    """Switch for selecting features to add to the initial time series"""

    """
    A switch, that takes as input the 
    
    Parameters
    ----------
    value: list of str

    X: pd.Dataframe

    lags_data: list of int

    date: datetime
        
    rolling_window: list of int

    num: int


    Returns
    -------
    function call
        Function call to corresponding feature creation 
    """

    return {
        'lag': lambda : lag_feature(lags_data, num),
        #--------------------------------------------------------------
        'mean': lambda : mean_feature(lags_data, rolling_window, num),
        'std': lambda : std_feature(lags_data, rolling_window, num),
        'max': lambda : max_feature(lags_data, rolling_window, num),
        'min': lambda : min_feature(lags_data, rolling_window, num),
        'quantile': lambda : quantile_feature(lags_data, rolling_window, num),
        'iqr': lambda : iqr_feature(lags_data, rolling_window, num),
        'entropy': lambda : entropy_feature(lags_data, rolling_window, num),
        'trimmean': lambda : trimmean_feature(lags_data, rolling_window, num),
        'variation': lambda : variation_feature(lags_data, rolling_window, num),
        'hmean': lambda : hmean_feature(lags_data, rolling_window, num),
        'gmean': lambda : gmean_feature(lags_data, rolling_window, num),
        'mad': lambda : mad_feature(lags_data, rolling_window, num),
#         'gstd': lambda : gstd_feature(lags_data, rolling_window, num),
        'tvar': lambda : tvar_feature(lags_data, rolling_window, num),
        'kurtosis': lambda : kurtosis_feature(lags_data, rolling_window, num),
        'sem': lambda : sem_feature(lags_data, rolling_window, num),
        'wav': lambda : wav_feature(lags_data, rolling_window, num),
        #--------------------------------------------------------------
        'minute': lambda : minute_feature(date, num),
        'hour': lambda : hour_feature(date, num),
        'dayofweek': lambda : dayofweek_feature(date, num),
        'day': lambda : day_feature(date, num),
        'month': lambda : month_feature(date, num),
        'quarter': lambda : quarter_feature(date, num),
        'weekofyear': lambda : weekofyear_feature(date, num),
        'weekend': lambda : weekend_feature(date, num)
    }.get(value)()

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


def quantile_feature(data, rolling_window, num):
    """Computes the median value for a given lag and its rolling window"""

    """
    Given a set number of lags in the past, a set rolling window and the input 
    dataset, computes the rolling median value for the given window.

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
        return np.quantile(data[-rolling_window:], 0.5)
    else:
        return np.quantile(data[-(rolling_window-1+num):-(num-1)], 0.5)


def iqr_feature(data, rolling_window, num):
    """Computes the interquartile range for a given lag and its rolling 
    window"""

    """
    Given a set number of lags in the past, a set rolling window and the input 
    dataset, computes the rolling interquartile range value for the given 
    window.

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
        return scipy.stats.iqr(data[-rolling_window:], rng = [25, 75])
    else:
        return scipy.stats.iqr(data[-(rolling_window-1+num):-(num-1)], 
                               rng = [25, 75])


def create_features(feature_names, lags_data, date):
    features = []
    for f in feature_names:
        ftype, info = f.split('_',1)
        if '_' in info:
            rolling_window, num = info.split('_')
            num = int(num)
            rolling_window = int(rolling_window)
        else:
            num = int(info)
            rolling_window = None
        features.append(switch_features(ftype, lags_data, date,
                                        rolling_window, num))
                                
    return features

#-------------------------------------------------------------------------------

def recursive_forecast(y, model, window_length, feature_names, rolling_window,
                       n_steps, freq):
    """
    Forecasting function which applies a given model recursively in a series

    Function, taking as an input the lagged time series with created features
    (Feature selection preprocessing optional) and applies recursive forecasting
    using a pre-trained model. Next time step is predicted with repect to the
    last prediction made.


    Parameters
    ----------
    y: pd.Series 
        holding the input time-series to forecast
    model: pre-trained machine learning model
    lags: list of str
        list of lags used for training the model
    n_steps: int 
        number of time periods in the forecasting horizon
    step: int 
        forecasting time period
   
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

def recursive_forecast_np(y, model, window_length, feature_names, 
                          rolling_window, n_steps, freq):
    """
    Recursive forecast function using numpy arrays

    Since keras LSTM and TCN models require numpy arrays that have specific 
    shapes to function, implement recursive forecasting on numpy arrays, 
    updating each data point with the previous one to predict the next.

    Parameters
    ----------
    y: pd.Series
        Input time series with constant frequency
    model: pre-trained model
        Keras RNN model
    scaler: Scaler
        Scaler to apply to Keras model input
    
    Returns
    -------
    pd.Series
        Forecasted values
    """
   
    # get the dates to forecast
    last_date = y.index[-1] + datetime.timedelta(minutes=15)
    target_range = pd.date_range(last_date, periods=n_steps, freq=freq) 
    target_value = np.arange(n_steps, dtype = 'float32')       
    max_rol = max(rolling_window, default=1)     
    lags = list(y.iloc[-(window_length+(max_rol-1)):,0].values)
    ####
    
    
    for i in range(n_steps):                                  
        train = create_features(feature_names, lags, target_range[i]) 
        train_np = np.array(train, dtype='float32')
        train_np_s = (train_np.reshape(-1,1))
        new_value = model.predict(train_np_s.reshape(-1,1,len(feature_names)))
        new_valu_0 = new_value
        target_value[i] = new_valu_0
        lags.pop(0)                                                
        lags.append(new_valu_0)                                   
                                                        

    return target_value

# Scaler -----------------------------------------------------------------------

def scale(y, scaler=scaler):
    y_scale = scaler.fit_transform(y)
    return(y_scale)


def unscale(y, scaler=scaler):
    y_orig = scaler.inverse_transform(y)
    return(y_orig)
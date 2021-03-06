"""Module containing all estimators and predictors for forecasting


Module containing the different models for time series forecasting. Including
simple generalized linear models, Gradient boosting random forest approaches
and convolutional and recurrent neural networks.
"""

import numpy as np
import pandas as pd
import datetime



from sktime.forecasting.naive import NaiveForecaster

import statsmodels.api as sm

from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from lightgbm.sklearn import LGBMRegressor

from hpsklearn import HyperoptEstimator as HE, knn_regression as knn_reg
from hpsklearn import lightgbm_regression as lgb_reg
from hyperopt import tpe, hp


def LSTM_Model_gen(n_feat):
    """
    LSTM recurrent neural network construction for forecasting with keras

    Long Short Term Memory (LSTM) neural net construction. Consisting of seven
    LSTM layers with 256 neurons each, all of them having relu activation
    function. Last layer consists of one single neuron also without activation
    function.

    """
    model1 = Sequential()
    model1.add(LSTM(256, activation='relu', input_shape=(1, n_feat),
                    return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(LSTM(256, activation='relu', return_sequences=True))
    model1.add(Dense(1))

    model1.compile(loss='mape', optimizer='adam',
                   metrics=MeanAbsolutePercentageError())
    return(model1)


def LSTM_Model(n_feat):
    return KerasRegressor(build_fn=(lambda: LSTM_Model_gen(n_feat)),
                          verbose=0,  batch_size=8,
                          epochs=50)


def GLM_Model():
    return(lambda: sm.GLM(family=sm.families.Gamma()))


lgb_reg_params = {
    'learning_rate': hp.choice('learning_rate', np.arange(0.02, 0.5, 0.05)),
    'max_depth': hp.choice('max_depth', np.arange(2, 20, 1, dtype=int)),
    'min_child_weight': hp.choice('min_child_weight', np.arange(0, 5, 1)),
    'colsample_bytree': hp.choice('colsample_bytree',
                                  np.arange(0.1, 0.6, 0.1)),
    'subsample': hp.choice('subsample', np.arange(0.1, 1, 0.05)),
    'n_estimators': hp.choice('n_estimators',
                              np.arange(10, 400, 25, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(10, 100, 15, dtype=int)),
    'reg_alpha': hp.choice('reg_alpha', np.arange(0.1, 1, 0.1)),
    'reg_lambda': hp.choice('reg_lambda', np.arange(0.5, 1.5, 0.1))
    }

knn_reg_params = {
    'n_neighbors': hp.choice('n_neighbors', np.arange(5, 50, 5)),
    'weights': hp.choice('weights', ['uniform', 'distance']),
    'leaf_size': hp.choice('leaf_size', np.arange(30, 60, 10))
    }

Naive_Model = NaiveForecaster()
KNN_Model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_jobs=-1))
LGB_Model = make_pipeline(StandardScaler(), LGBMRegressor(n_jobs=-1))


def KNN_Model_Opt(opt_runs=50, knn_reg_params=knn_reg_params):
    return(HE(regressor=knn_reg('knnopt', **knn_reg_params),
              algo=tpe.suggest,
              max_evals=opt_runs,
              n_jobs=-1))


def LGB_Model_Opt(opt_runs=50, lgb_reg_params=lgb_reg_params):

    return(HE(regressor=lgb_reg('lgbmopt', **lgb_reg_params),
              algo=tpe.suggest,
              max_evals=opt_runs,
              n_jobs=-1))



def switch_features(value, lags_data, date, rolling_window, num):
    """Switch for selecting features to add to the initial time series


    A switch, that takes as input the list of features an lags to create at
    each iteration of the recursive forecasting process.

    Parameters
    ----------
    value: list of str
        list of feature and lag names to create at each recursive forecast
        point
    X: pd.Dataframe
        input dataframe containing the time series data
    lags_data: list of int
        list of the time lags of the time series
    date: datetime
        date for which to build the next point in time's data
    rolling_window: list of int
        rolling window on which to calculate the generated features.
    num: int
        number of steps to calculate lags and features with

    Returns
    -------
    function call
        Function call to corresponding feature creation
    """

    return {
        'lag': lambda: lag_feature(lags_data, num),
        'mean': lambda: mean_feature(lags_data, rolling_window, num),
        'std': lambda: std_feature(lags_data, rolling_window, num),
        'max': lambda: max_feature(lags_data, rolling_window, num),
        'min': lambda: min_feature(lags_data, rolling_window, num),
        'quantile': lambda: quantile_feature(lags_data, rolling_window, num),
        'minute': lambda: minute_feature(date, num),
        'hour': lambda: hour_feature(date, num),
        'dayofweek': lambda: dayofweek_feature(date, num),
        'day': lambda: day_feature(date, num),
        'month': lambda: month_feature(date, num),
        'weekend': lambda: weekend_feature(date, num)
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
        lag number to calculate

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
    """One Hot encoding for hour of day feature


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
    hour = date.hour
    if hour == num:
        return 1
    else:
        return 0


def dayofweek_feature(date, num):
    """One Hot encoding for day of week feature


    One Hot encoding for the day of week feature of a given sample. takes a 1
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
    if dayofweek == num:
        return 1
    else:
        return 0


def day_feature(date, num):
    """One Hot encoding for day of month feature


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
    """One Hot encoding for month of year feature


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


def weekend_feature(date, num):
    """One Hot encoding for weekend feature


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
    """Computes the mean for a given lag and its rolling window


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
    """Computes the standard deviation for a given lag and its rolling window


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
    """Computes the maximum value for a given lag and its rolling window


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
    """Computes the minimum value for a given lag and its rolling window


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
    """Computes the median value for a given lag and its rolling window


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


def create_features(feature_names, lags_data, date):
    features = []
    for f in feature_names:
        ftype, info = f.split('_', 1)
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


def rec_forecast(y, model, window_length, feature_names, rolling_window,
                 n_steps, freq):
    """
    Forecasting function which applies a given model recursively in a series
    Function, taking as an input the lagged time series with created features
    (Feature selection preprocessing optional) and applies recursive
    forecasting using a pre-trained model. Next time step is predicted with
    respect to the last prediction made.
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
    last_date = y.index[-1] + datetime.timedelta(minutes=15)
    target_range = pd.date_range(last_date, periods=n_steps, freq=freq)
    target_value = np.arange(n_steps, dtype=float)
    max_rol = max(rolling_window, default=1)
    lags = list(y.iloc[-(window_length+(max_rol-1)):, 0].values)
    for i in range(n_steps):
        train = create_features(feature_names, lags, target_range[i])
        new_value = model.predict(pd.DataFrame(train).transpose())
        target_value[i] = new_value[0]
        lags.pop(0)
        lags.append(new_value[0])
    return target_value


def rec_forecast_np(y, model, window_length, feature_names,
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
        
    Returns
    -------
    pd.Series
        Forecasted values
    """
    last_date = y.index[-1] + datetime.timedelta(minutes=15)
    target_range = pd.date_range(last_date, periods=n_steps, freq=freq)
    target_value = np.arange(n_steps, dtype='float32')
    max_rol = max(rolling_window, default=1)
    lags = list(y.iloc[-(window_length+(max_rol-1)):, 0].values)
    for i in range(n_steps):
        train = create_features(feature_names, lags, target_range[i])
        train_np = np.array(train, dtype='float32')
        train_np_s = train_np.reshape(-1, 1)
        new_value = model.predict(train_np_s.reshape(-1, 1,
                                  len(feature_names)))
        new_valu_0 = new_value
        target_value[i] = new_valu_0
        lags.pop(0)
        lags.append(new_valu_0)
    return target_value
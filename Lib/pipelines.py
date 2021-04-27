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
import os
import sys
import numpy as np
import pandas as pd

import data_input

os.chdir(os.path.dirname(sys.argv[0]))

def fast_prediction(filename, freq, targetcol, datecol, 
                    sep, decimal, date_format,
                    points=50, window_length=50, rolling_window=[5,10,20], 
                    horizon=1, step=1, 
                    features=['mean', 'std', 'max', 'min', 'minute'], 
                    selected_feat=20, plot=True):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = preprocessing.ts_split(df, test_size=50)
    X_train = preprocessing.create_sample_features(y_train, 
                                                   window_length=window_length, 
                                                   features=features, 
                                                   rolling_window=rolling_window)
    
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_train = X_train.iloc[-2000:,:]

    y_horizon = preprocessing.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:,:]

    best_features = preprocessing.feature_selection(X_train, 
                                                y_horizon.values.ravel(), 
                                                selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.GLM_Model
    regressor.fit(X=X_train_selec, 
                  y=y_horizon.values.ravel())

    pred = prediction.recursive_forecast(y=y_train, 
                                         model=regressor, 
                                         window_length=window_length,
                                         feature_names=best_features,
                                         rolling_window=rolling_window,
                                         n_steps=points,
                                         freq=freq)

    if plot:
        plotting.plot_test_pred(y_test, pred)
    
    return(pred)

    
def balanced_prediction(filename, freq, targetcol, datecol, 
                        sep, decimal, date_format,
                        points=50, window_length=100, rolling_window=[5,10,20], 
                        horizon=1, step=1, 
                        features=['mean', 'std', 'max', 'min', 'minute'], 
                        selected_feat=40, plot=True):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = preprocessing.ts_split(df, test_size=50)
    X_train = preprocessing.create_sample_features(y_train, 
                                                   window_length=window_length, 
                                                   features=features, 
                                                   rolling_window=rolling_window)
    
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_train = X_train.iloc[-2000:,:]

    y_horizon = preprocessing.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:,:]

    best_features = preprocessing.feature_selection(X_train, 
                                                y_horizon.values.ravel(), 
                                                selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.GLM_Model
    regressor.fit(X=X_train_selec, 
                  y=y_horizon.values.ravel())

    pred = prediction.recursive_forecast(y=y_train, 
                                         model=regressor, 
                                         window_length=window_length,
                                         feature_names=best_features,
                                         rolling_window=rolling_window,
                                         n_steps=points,
                                         freq=freq)

    if plot:
        plotting.plot_test_pred(y_test, pred)
    
    return(pred)

    
def slow_prediction(filename, freq, targetcol, datecol, 
                    sep, decimal, date_format,
                    points=50, window_length=100, rolling_window=[5,10,20], 
                    horizon=1, step=1, 
                    features=['mean', 'std', 'max', 'min', 'minute'], 
                    selected_feat=50, plot=True):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = preprocessing.ts_split(df, test_size=50)
    X_train = preprocessing.create_sample_features(y_train, 
                                                   window_length=window_length, 
                                                   features=features, 
                                                   rolling_window=rolling_window)
    
    X_train = X_train.loc[:,~X_train.columns.duplicated()]
    X_train = X_train.iloc[-2000:,:]

    y_horizon = preprocessing.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:,:]

    best_features = preprocessing.feature_selection(X_train, 
                                                y_horizon.values.ravel(), 
                                                selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.LSTM_Model
    regressor.fit(X=X_train_selec.to_numpy().reshape(-1,1,n_feat), 
                  y=y_horizon.values.ravel())

    pred = prediction.recursive_forecast_np(y=y_train, 
                                            model=regressor, 
                                            window_length=window_length,
                                            feature_names=best_features,
                                            rolling_window=rolling_window,
                                            n_steps=points,
                                            freq=freq)

    if plot:
        plotting.plot_test_pred(y_test, pred)
    
    return(pred)
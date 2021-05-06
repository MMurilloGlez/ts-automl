"""Module containing the sklearn pipelines for time series forecasting


This module contains the sklearn pipelines for the different levels of forecast
'difficulty'. Three will be proposed, a fast, a balanced and a slow prediction,
each sacrificing processing time for forecasting accuracy.

"""
import numpy as np

from ts_automl import data_input
from ts_automl import preprocessing
from ts_automl import prediction
from ts_automl import plotting
from ts_automl import metrics


def fast_prediction(filename, freq, targetcol, datecol,
                    sep, decimal, date_format,
                    points=50, window_length=50, rolling_window=[5, 10, 20],
                    horizon=1, step=1, num_datapoints=2000,
                    features=['mean', 'std', 'max', 'min', 'minute'],
                    selected_feat=20, plot=True, error=['mse', 'mape'],
                    rel_metrics=True):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = preprocessing.ts_split(df, test_size=points)
    X_train = preprocessing.create_sample_features(y_train,
                                                   window_length=window_length,
                                                   features=features,
                                                   rolling_window=rolling_window)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_train = X_train.iloc[-num_datapoints:, :]

    y_horizon = preprocessing.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:, :]

    best_features = preprocessing.feature_selection(X_train,
                                                    y_horizon.values.ravel(),
                                                    selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.KNN_Model
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

    if error:
        abs_error = []
        for i in error:
            abs_error.append(metrics.switch_abs_error(i, y_test, pred))
        print(error)
        print(abs_error)

    if rel_metrics:
        naive = naive_prediction(filename=filename, freq=freq,
                                 targetcol=targetcol, datecol=datecol,
                                 sep=sep, decimal=decimal,
                                 date_format=date_format, points=points,
                                 num_datapoints=num_datapoints)
        r_error = metrics.relative_error(y_test, pred, naive)
        print('error relative to naïve prediction')
        print(r_error)

    return pred, abs_error, r_error


def balanced_prediction(filename, freq, targetcol, datecol,
                        sep, decimal, date_format,
                        points=50, window_length=100,
                        rolling_window=[5, 10, 20], horizon=1, step=1,
                        features=['mean', 'std', 'max', 'min', 'minute'],
                        selected_feat=40, num_datapoints=2000,
                        plot=True, error=['mse', 'mape'],
                        rel_metrics=True):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = preprocessing.ts_split(df, test_size=points)
    X_train = preprocessing.create_sample_features(y_train,
                                                   window_length=window_length,
                                                   features=features,
                                                   rolling_window=rolling_window)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]

    y_horizon = preprocessing.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:, :]

    best_features = preprocessing.feature_selection(X_train,
                                                    y_horizon.values.ravel(),
                                                    selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.LGB_Model
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

    if error:
        abs_error = []
        for i in error:
            abs_error.append(metrics.switch_abs_error(i, y_test, pred))
        print(error)
        print(abs_error)

    if rel_metrics:
        naive = naive_prediction(filename=filename, freq=freq,
                                 targetcol=targetcol, datecol=datecol,
                                 sep=sep, decimal=decimal,
                                 date_format=date_format, points=points,
                                 num_datapoints=num_datapoints)
        r_error = metrics.relative_error(y_test, pred, naive)
        print('error relative to naïve prediction')
        print(r_error)

    return pred,  abs_error, r_error


def slow_prediction(filename, freq, targetcol, datecol,
                    sep, decimal, date_format,
                    points=50, window_length=100, rolling_window=[5, 10, 20],
                    horizon=1, step=1,
                    features=['mean', 'std', 'max', 'min', 'minute'],
                    selected_feat=50, num_datapoints=2000,
                    plot=True, error=['mse', 'mape'],
                    rel_metrics=True):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = preprocessing.ts_split(df, test_size=points)
    y_train = y_train.iloc[-num_datapoints:, :]

    X_train = preprocessing.create_sample_features(y_train,
                                                   window_length=window_length,
                                                   features=features,
                                                   rolling_window=rolling_window)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]

    y_horizon = preprocessing.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:, :]

    best_features = preprocessing.feature_selection(X_train,
                                                    y_horizon.values.ravel(),
                                                    selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.LSTM_Model(n_feat=selected_feat)
    regressor.fit(x=X_train_selec.to_numpy().reshape(-1, 1, selected_feat),
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

    if error:
        abs_error = []
        for i in error:
            abs_error.append(metrics.switch_abs_error(i, y_test, pred))
        print(error)
        print(abs_error)

    if rel_metrics:
        naive = naive_prediction(filename=filename, freq=freq,
                                 targetcol=targetcol, datecol=datecol,
                                 sep=sep, decimal=decimal,
                                 date_format=date_format, points=points,
                                 num_datapoints=num_datapoints)
        r_error = metrics.relative_error(y_test, pred, naive)
        print('error relative to naïve prediction')
        print(r_error)

    return pred, abs_error, r_error


def naive_prediction(filename, freq, targetcol, datecol,
                     sep, decimal, date_format,
                     points=50, num_datapoints=2000, plot=False,
                     error=['mse', 'mape']):

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)
    df = df.squeeze()
    y_train, y_test = preprocessing.ts_split(df, test_size=points)
    y_train = y_train.iloc[-num_datapoints:]

    regressor = prediction.Naive_Model()

    regressor.fit(y_train)

    pred = regressor.predict(fh=np.arange(len(y_test)) + 1)

    if plot:
        plotting.plot_test_pred(y_test.to_frame(), pred)

    if error:
        abs_error = []
        for i in error:
            abs_error.append(metrics.switch_abs_error(i, y_test, pred))
        print('Naïve error metrics:')
        print(error)
        print(abs_error)

    return pred

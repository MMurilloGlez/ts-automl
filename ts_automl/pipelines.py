"""Module containing the sklearn pipelines for time series forecasting


This module contains the sklearn pipelines for the different levels of forecast
'difficulty'. Three will be proposed, a fast, a balanced and a slow prediction,
each sacrificing processing time for forecasting accuracy.

"""
import numpy as np

from ts_automl import data_input
from ts_automl import preprocessing as pre
from ts_automl import prediction
from ts_automl import plotting
from ts_automl import metrics


features = ["mean", "std", "max", "min", "quantile", "minute",
            "hour", "dayofweek", "day", "month", "weekend"]


def fast_prediction(filename, freq, targetcol, datecol,
                    sep, decimal, date_format,
                    points=50, window_length=100, rolling_window=[5, 10, 20],
                    horizon=1, step=1, num_datapoints=2000,
                    features=features,
                    selected_feat=50, plot=True, error=['mse', 'mape'],
                    rel_metrics=True, opt=False, opt_runs=10,
                    plot_train=True):

    if rel_metrics is not True:
        r_error = None

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = pre.ts_split(df, test_size=points)
    y_train = y_train.iloc[-num_datapoints:, :]
    X_train = pre.create_sample_feat(y_train,
                                     window_length=window_length,
                                     features=features,
                                     rolling_window=rolling_window)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_train = X_train.iloc[-num_datapoints:, :]

    y_horizon = pre.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:, :]

    best_features = pre.feature_selection(X_train,
                                          y_horizon.values.ravel(),
                                          selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor = prediction.KNN_Model
    regressor.fit(X=X_train_selec,
                  y=y_horizon.values.ravel())

    pred = prediction.rec_forecast(y=y_train,
                                   model=regressor,
                                   window_length=window_length,
                                   feature_names=best_features,
                                   rolling_window=rolling_window,
                                   n_steps=points,
                                   freq=freq)
    reg = regressor
    if opt:

        regressor_1_o = prediction.KNN_Model_Opt(opt_runs=opt_runs)
        regressor_1_o.fit(X=X_train_selec,
                          y=y_horizon.values.ravel())
        pred_1_o = prediction.rec_forecast(y=y_train,
                                           model=regressor_1_o,
                                           window_length=window_length,
                                           feature_names=best_features,
                                           rolling_window=rolling_window,
                                           n_steps=points,
                                           freq=freq)
        pred = pred_1_o
        reg = regressor_1_o

    if plot:
        plotting.plot_test_pred(y_test, pred)

    if plot_train:
        plotting.plot_train(y_train.iloc[-2000:, :])

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
        r_error = metrics.relative_error(y_test, pred, naive['pred'])
        print('error relative to naïve prediction')
        print(r_error)
    response = {'filename': filename,
                'features': features,
                'regressor': reg,
                'r_error': r_error,
                'abs_error': abs_error,
                'error': error,
                'pred': pred,
                'y_true': y_test,
                'naive': naive
                }
    return response


def balanced_prediction(filename, freq, targetcol, datecol,
                        sep, decimal, date_format,
                        points=50, window_length=100,
                        rolling_window=[5, 10, 20], horizon=1, step=1,
                        features=features,
                        selected_feat=50, num_datapoints=2000,
                        plot=True, error=['mse', 'mape'],
                        rel_metrics=True, opt=False, opt_runs=10,
                        plot_train=True):

    if rel_metrics is not True:
        r_error = None

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = pre.ts_split(df, test_size=points)
    y_train = y_train.iloc[-num_datapoints:, :]
    X_train = pre.create_sample_feat(y_train,
                                     window_length=window_length,
                                     features=features,
                                     rolling_window=rolling_window)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]

    y_horizon = pre.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:, :]

    best_features = pre.feature_selection(X_train,
                                          y_horizon.values.ravel(),
                                          selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor_1 = prediction.KNN_Model
    regressor_1.fit(X=X_train_selec,
                    y=y_horizon.values.ravel())

    pred_1 = prediction.rec_forecast(y=y_train,
                                     model=regressor_1,
                                     window_length=window_length,
                                     feature_names=best_features,
                                     rolling_window=rolling_window,
                                     n_steps=points,
                                     freq=freq)

    error_1 = metrics.switch_abs_error('mse', y_test, pred_1)

    regressor_2 = prediction.LGB_Model
    regressor_2.fit(X=X_train_selec,
                    y=y_horizon.values.ravel())

    pred_2 = prediction.rec_forecast(y=y_train,
                                     model=regressor_2,
                                     window_length=window_length,
                                     feature_names=best_features,
                                     rolling_window=rolling_window,
                                     n_steps=points,
                                     freq=freq)

    error_2 = metrics.switch_abs_error('mse', y_test, pred_2)

    if error_1 < error_2:
        print('Using KNN Prediction')
        if opt:
            print('Optimizing model')
            regressor_1_o = prediction.KNN_Model_Opt(opt_runs=opt_runs)
            regressor_1_o.fit(X=X_train_selec,
                              y=y_horizon.values.ravel())
            pred_1_o = prediction.rec_forecast(y=y_train,
                                               model=regressor_1_o,
                                               window_length=window_length,
                                               feature_names=best_features,
                                               rolling_window=rolling_window,
                                               n_steps=points,
                                               freq=freq)
            pred = pred_1_o
            reg = regressor_1_o
        else:
            pred = pred_1
            reg = regressor_1
    else:
        print('Using LightGBM prediction')
        if opt:
            print('Optimizing model')
            regressor_2_o = prediction.LGB_Model_Opt(opt_runs=opt_runs)
            regressor_2_o.fit(X=X_train_selec,
                              y=y_horizon.values.ravel())
            pred_2_o = prediction.rec_forecast(y=y_train,
                                               model=regressor_2_o,
                                               window_length=window_length,
                                               feature_names=best_features,
                                               rolling_window=rolling_window,
                                               n_steps=points,
                                               freq=freq)
            pred = pred_2_o
            reg = regressor_2_o
        else:
            pred = pred_2
            reg = regressor_2

    if plot:
        plotting.plot_test_pred(y_test, pred)

    if plot_train:
        plotting.plot_train(y_train)

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
        r_error = metrics.relative_error(y_test, pred, naive['pred'])
        print('error relative to naïve prediction')
        print(r_error)

    response = {'filename': filename,
                'features': features,
                'regressor': reg,
                'r_error': r_error,
                'abs_error': abs_error,
                'error': error,
                'pred': pred,
                'y_test': y_test,
                'naive': naive
                }
    return response


def slow_prediction(filename, freq, targetcol, datecol,
                    sep, decimal, date_format,
                    points=50, window_length=100, rolling_window=[5, 10, 20],
                    horizon=1, step=1,
                    features=features,
                    selected_feat=50, num_datapoints=2000,
                    plot=True, error=['mse', 'mape'],
                    rel_metrics=True, opt=False, opt_runs=30,
                    plot_train=True):

    if rel_metrics is not True:
        r_error = None

    df = data_input.read_data(filename=filename,
                              freq=freq,
                              targetcol=targetcol,
                              datecol=datecol,
                              sep=sep,
                              decimal=decimal,
                              date_format=date_format)

    y_train, y_test = pre.ts_split(df, test_size=points)
    y_train = y_train.iloc[-num_datapoints:, :]

    X_train = pre.create_sample_feat(y_train,
                                     window_length=window_length,
                                     features=features,
                                     rolling_window=rolling_window)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]

    y_horizon = pre.create_horizon(y_train, horizon)
    y_horizon = y_horizon.loc[X_train.index[0]:, :]

    best_features = pre.feature_selection(X_train,
                                          y_horizon.values.ravel(),
                                          selected_feat)

    X_train_selec = X_train.loc[:, best_features]

    regressor_1 = prediction.KNN_Model
    regressor_1.fit(X=X_train_selec,
                    y=y_horizon.values.ravel())

    pred_1 = prediction.rec_forecast(y=y_train,
                                     model=regressor_1,
                                     window_length=window_length,
                                     feature_names=best_features,
                                     rolling_window=rolling_window,
                                     n_steps=points,
                                     freq=freq)

    error_1 = metrics.switch_abs_error('mse', y_test, pred_1)

    regressor_2 = prediction.LGB_Model
    regressor_2.fit(X=X_train_selec,
                    y=y_horizon.values.ravel())

    pred_2 = prediction.rec_forecast(y=y_train,
                                     model=regressor_2,
                                     window_length=window_length,
                                     feature_names=best_features,
                                     rolling_window=rolling_window,
                                     n_steps=points,
                                     freq=freq)

    error_2 = metrics.switch_abs_error('mse', y_test, pred_2)

    regressor_3 = prediction.LSTM_Model(n_feat=selected_feat)
    regressor_3.fit(x=X_train_selec.to_numpy().reshape(-1, 1, selected_feat),
                    y=y_horizon.values.ravel())

    pred_3 = prediction.rec_forecast_np(y=y_train,
                                        model=regressor_3,
                                        window_length=window_length,
                                        feature_names=best_features,
                                        rolling_window=rolling_window,
                                        n_steps=points,
                                        freq=freq)

    error_3 = metrics.switch_abs_error('mse', y_test, pred_3)

    if (error_3 < error_2) & (error_3 < error_1):
        pred = pred_3
        reg = regressor_3
        print('Using LSTM prediction')
    elif(error_2 < error_1) & (error_2 < error_3):
        print('Using LightGBM prediction')
        if opt:
            print('Optimizing model')
            regressor_2_o = prediction.LGB_Model_Opt(opt_runs=opt_runs)
            regressor_2_o.fit(X=X_train_selec,
                              y=y_horizon.values.ravel())
            pred_2_o = prediction.rec_forecast(y=y_train,
                                               model=regressor_2_o,
                                               window_length=window_length,
                                               feature_names=best_features,
                                               rolling_window=rolling_window,
                                               n_steps=points,
                                               freq=freq)
            pred = pred_2_o
            reg = regressor_2_o
        else:
            pred = pred_2
            reg = regressor_2
    else:
        print('Using KNN Prediction')
        if opt:
            print('Optimizing model')
            regressor_1_o = prediction.KNN_Model_Opt(opt_runs=opt_runs)
            regressor_1_o.fit(X=X_train_selec,
                              y=y_horizon.values.ravel())
            pred_1_o = prediction.rec_forecast(y=y_train,
                                               model=regressor_1_o,
                                               window_length=window_length,
                                               feature_names=best_features,
                                               rolling_window=rolling_window,
                                               n_steps=points,
                                               freq=freq)
            pred = pred_1_o
            reg = regressor_1_o
        else:
            pred = pred_1
            reg = regressor_1

    if plot:
        plotting.plot_test_pred(y_test, pred)

    if plot_train:
        plotting.plot_train(y_train)

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
        r_error = metrics.relative_error(y_test, pred, naive['pred'])
        print('error relative to naïve prediction')
        print(r_error)

    response = {'filename': filename,
                'features': features,
                'regressor': reg,
                'r_error': r_error,
                'abs_error': abs_error,
                'error': error,
                'pred': pred,
                'y_test': y_test,
                'naive': naive
                }
    return response


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
    y_train, y_test = pre.ts_split(df, test_size=points)
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
        reg = regressor

    response = {'filename': filename,
                'features': None,
                'regressor': reg,
                'r_error': None,
                'abs_error': abs_error,
                'error': error,
                'pred': pred,
                'y_true': y_test
                }
    return response


class pipeline(object):
    def __init__(self, filename: str,
                 type: str = 'balanced', freq: str = "15T",
                 targetcol: str = "VALUE", datecol: str = "DATE",
                 sep: str = ";", decimal: str = ",",
                 date_format: str = "%d/%m/%Y %H:%M:%S.%f", features=features,
                 selected_feat: int = 50, plot: bool = True,
                 window_length: int = 100, horizon: int = 1,
                 rolling_window: "list[int]" = [5, 10, 20], points: int = 50,
                 error: "list[str]" = ['mse', 'mape'],
                 num_datapoints: int = 2000, plot_train: bool = True,
                 rel_metrics=True,
                 **kwargs):

        """
        Init method concerning all transformations prior to fitting the model


        The initiation of the class pbject will trigger the building of the
        dataset including lagged values and generated features. It will also
        build a test dataset to check accuracy of future fitted models.


        Parameters
        ----------

        filename: str
            file path of a csv or xls relative to python working path
        type: str ['fast','balanced','slow']:
            type of model to be fit.
        freq: str
            Frequency to interpolate the time series data
        targetcol: str
            Name of target column of data i.e. the values column
        datecol: str
            Name of the timestamp column of the data
        sep: str
            Separator for csv data
        decimal: str
            Decimal point or comma for data
        date_format: str
            Date format for the timestamps values
        features: list of str
            Features to be computed for the creation of te ddataset
        selected_feat: int
            Number of features to preserve from the ones created
        plot: bool
            Whether or not to plot the training data
        window_length: int
            Number of lags to include in dataset
        horizon: int
            Step to calculate in the future for prediction
        rolling_window: list of int
            Rolling window for which to calculate features
        points: int
            Number of points in the future to calculate prediction / test size
        error: list of str
            Metric(s) to calculate for characterizing predictions
        rel_metrics: bool
            Indicates if error relative to naive is to be calculated.
        num_datapoints: int
            Number of datapoints from dataset for which to generate predictions
        plot_train: bool
            Whether to plot the training time series

        """

        self.filename = filename
        self.freq = freq
        self.targetcol = targetcol
        self.datecol = datecol
        self.sep = sep
        self.decimal = decimal
        self.date_format = date_format
        self.window_length = window_length
        self.rolling_window = rolling_window
        self.plot = plot
        self.points = points

        self.type = type
        self.num_datapoints = num_datapoints
        self.error = error
        self.selected_feat = selected_feat
        self.rel_metrics = rel_metrics

        self.df = data_input.read_data(filename=filename,
                                       freq=freq,
                                       targetcol=targetcol,
                                       datecol=datecol,
                                       sep=sep,
                                       decimal=decimal,
                                       date_format=date_format)
        y_train, self.y_test = pre.ts_split(self.df,
                                            test_size=points)
        self.y_train = y_train.iloc[-num_datapoints:, :]
        X_train = pre.create_sample_feat(self.y_train,
                                         window_length=window_length,
                                         features=features,
                                         rolling_window=rolling_window)

        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_train = X_train.iloc[-num_datapoints:, :]
        self.X_train = X_train

        if plot_train:
            plotting.plot_train(self.y_train)

        y_horizon = pre.create_horizon(y_train, horizon)
        self.y_horizon = y_horizon.loc[X_train.index[0]:, :]

        best_features = pre.feature_selection(self.X_train,
                                              self.y_horizon.values.ravel(),
                                              selected_feat)

        X_train_selec = self.X_train.loc[:, best_features]
        self.X_train_selec = X_train_selec
        self.best_features = best_features

    def fit(self):
        """
        Fits the ML model(s) according to the type parameter.

        Will fit the models that correspond to the type parameter. ´fast´ will
        fit GLM and kNN, ´balanced´ will fit GLM, kNN and LightGB: ´slow´ will
        fit all of the above and LSTM.

        The optimum model will be saved under the ´model´ parameter of the
        class. WHile the test predictions for the optimum model will be saved
        under the ´test_pred´ parameter.

        This method doesn't accept parameters but rather inherits from the ones
        defined in __init__
        """
        self.modeltype = 'Pandas'
        regressor_1 = prediction.KNN_Model
        regressor_2 = prediction.LGB_Model
        regressor_3 = prediction.LSTM_Model(n_feat=self.selected_feat)

        if self.type == 'fast':

            regressor_1.fit(X=self.X_train_selec,
                            y=self.y_horizon.values.ravel())
            pred_1 = prediction.rec_forecast(y=self.y_train,
                                             model=regressor_1,
                                             window_length=self.window_length,
                                             feature_names=self.best_features,
                                             rolling_window=self.rolling_window,
                                             n_steps=self.points,
                                             freq=self.freq)

            error_1 = metrics.switch_abs_error('mse', self.y_test, pred_1)

            pred = pred_1
            reg = regressor_1

        if self.type == 'balanced':
            regressor_1.fit(X=self.X_train_selec,
                            y=self.y_horizon.values.ravel())
            pred_1 = prediction.rec_forecast(y=self.y_train,
                                             model=regressor_1,
                                             window_length=self.window_length,
                                             feature_names=self.best_features,
                                             rolling_window=self.rolling_window,
                                             n_steps=self.points,
                                             freq=self.freq)

            error_1 = metrics.switch_abs_error('mse', self.y_test, pred_1)

            regressor_2.fit(X=self.X_train_selec,
                            y=self.y_horizon.values.ravel())
            pred_2 = prediction.rec_forecast(y=self.y_train,
                                             model=regressor_2,
                                             window_length=self.window_length,
                                             feature_names=self.best_features,
                                             rolling_window=self.rolling_window,
                                             n_steps=self.points,
                                             freq=self.freq)
            error_2 = metrics.switch_abs_error('mse', self.y_test, pred_2)

            if error_2 > error_1:
                reg = regressor_1
                pred = pred_1

            else:
                reg = regressor_2
                pred = pred_2

        if self.type == 'slow':
            regressor_1.fit(X=self.X_train_selec,
                            y=self.y_horizon.values.ravel())
            pred_1 = prediction.rec_forecast(y=self.y_train,
                                             model=regressor_1,
                                             window_length=self.window_length,
                                             feature_names=self.best_features,
                                             rolling_window=self.rolling_window,
                                             n_steps=self.points,
                                             freq=self.freq)

            error_1 = metrics.switch_abs_error('mse', self.y_test, pred_1)

            regressor_2.fit(X=self.X_train_selec,
                            y=self.y_horizon.values.ravel())
            pred_2 = prediction.rec_forecast(y=self.y_train,
                                             model=regressor_2,
                                             window_length=self.window_length,
                                             feature_names=self.best_features,
                                             rolling_window=self.rolling_window,
                                             n_steps=self.points,
                                             freq=self.freq)
            error_2 = metrics.switch_abs_error('mse', self.y_test, pred_2)

            regressor_3.fit(x=self.X_train_selec.to_numpy().reshape(-1, 1, self.selected_feat),
                            y=self.y_horizon.values.ravel())
            pred_3 = prediction.rec_forecast_np(y=self.y_train,
                                                model=self.regressor_3,
                                                window_length=self.window_length,
                                                feature_names=self.best_features,
                                                rolling_window=self.rolling_window,
                                                n_steps=self.points,
                                                freq=self.freq)
            error_3 = metrics.switch_abs_error('mse', self.y_test, pred_3)

            if (error_3 < error_2) & (error_3 < error_1):
                reg = regressor_3
                pred = pred_3
                self.modeltype = 'Numpy'

            elif (error_2 > error_1) & (error_3 > error_1):
                reg = regressor_1
                pred = pred_1

            else:
                reg = regressor_2
                pred = pred_2

        if self.plot:
            self.plot_test = plotting.plot_test_pred(self.y_test, pred)
            self.plot_test

        if self.error:
            abs_error = []
            for i in self.error:
                abs_error.append(metrics.switch_abs_error(i, self.y_test, pred))
            print(self.error)
            print(abs_error)
            self.abs_error = abs_error

        if self.rel_metrics:
            naive = naive_prediction(filename=self.filename, freq=self.freq,
                                     targetcol=self.targetcol,
                                     datecol=self.datecol,
                                     sep=self.sep, decimal=self.decimal,
                                     date_format=self.date_format,
                                     points=self.points,
                                     num_datapoints=self.num_datapoints)
            r_error = metrics.relative_error(self.y_test, pred, naive['pred'])
            print('error relative to naïve prediction')
            print(r_error)
            self.naive = naive
            self.r_error = r_error

        self.model = reg

        response = {'filename': self.filename,
                    'features': features,
                    'regressor': reg,
                    'r_error': r_error,
                    'abs_error': abs_error,
                    'error': self.error,
                    'pred': pred,
                    'y_test': self.y_test,
                    'naive': naive
                    }

        self.fit_response = response

        return response

    def fit_opt(self, opt_runs):
        """
        Fits the ML model(s) with optimization

        Fits wither the kNN or LightGBM models using HyperOpt as an optimizer,
        as before, ´test_pred´ will correspond to the predicted values for the
        test dataset, while the model will be saved under the ´model´ parameter
        of the class.

        To be run after normal fit

        Parameters
        ----------
        opt_runs: int /50/
            Number of passes or optimizations to run before finishing.

        """

        self.opt_runs = opt_runs

        regressor_1 = prediction.KNN_Model
        regressor_1.fit(X=self.X_train_selec,
                        y=self.y_horizon.values.ravel())

        pred_1 = prediction.rec_forecast(y=self.y_train,
                                         model=regressor_1,
                                         window_length=self.window_length,
                                         feature_names=self.best_features,
                                         rolling_window=self.rolling_window,
                                         n_steps=self.points,
                                         freq=self.freq)

        error_1 = metrics.switch_abs_error('mse', self.y_test, pred_1)

        regressor_2 = prediction.LGB_Model
        regressor_2.fit(X=self.X_train_selec,
                        y=self.y_horizon.values.ravel())

        pred_2 = prediction.rec_forecast(y=self.y_train,
                                         model=regressor_2,
                                         window_length=self.window_length,
                                         feature_names=self.best_features,
                                         rolling_window=self.rolling_window,
                                         n_steps=self.points,
                                         freq=self.freq)

        error_2 = metrics.switch_abs_error('mse', self.y_test, pred_2)

        if error_1 < error_2:
            print('Using KNN Prediction')

            print('Optimizing model')
            regressor_1_o = prediction.KNN_Model_Opt(opt_runs=opt_runs)
            regressor_1_o.fit(X=self.X_train_selec,
                              y=self.y_horizon.values.ravel())
            pred_1_o = prediction.rec_forecast(y=self.y_train,
                                               model=regressor_1_o,
                                               window_length=self.window_length,
                                               feature_names=self.best_features,
                                               rolling_window=self.rolling_window,
                                               n_steps=self.points,
                                               freq=self.freq)
            pred = pred_1_o
            reg = regressor_1_o
        else:
            print('Using LightGBM prediction')

            print('Optimizing model')
            regressor_2_o = prediction.LGB_Model_Opt(opt_runs=opt_runs)
            regressor_2_o.fit(X=self.X_train_selec,
                              y=self.y_horizon.values.ravel())
            pred_2_o = prediction.rec_forecast(y=self.y_train,
                                               model=regressor_2_o,
                                               window_length=self.window_length,
                                               feature_names=self.best_features,
                                               rolling_window=self.rolling_window,
                                               n_steps=self.points,
                                               freq=self.freq)
            pred = pred_2_o
            reg = regressor_2_o

        self.test_pred = pred
        self.model = reg

        if self.plot:
            self.plot_test = plotting.plot_test_pred(self.y_test, pred)
            self.plot_test

        if self.error:
            abs_error = []
            for i in self.error:
                abs_error.append(metrics.switch_abs_error(i, self.y_test, pred))
            print(self.error)
            print(abs_error)
            self.abs_error = abs_error

        if self.rel_metrics:
            naive = naive_prediction(filename=self.filename, freq=self.freq,
                                     targetcol=self.targetcol,
                                     datecol=self.datecol,
                                     sep=self.sep, decimal=self.decimal,
                                     date_format=self.date_format,
                                     points=self.points,
                                     num_datapoints=self.num_datapoints)
            r_error = metrics.relative_error(self.y_test, pred, naive['pred'])
            print('error relative to naïve prediction')
            print(r_error)
            self.naive = naive
            self.r_error = r_error

        response = {'filename': self.filename,
                    'features': features,
                    'regressor': reg,
                    'r_error': r_error,
                    'abs_error': abs_error,
                    'error': self.error,
                    'pred': pred,
                    'y_test': self.y_test,
                    'naive': naive
                    }

        self.fit_opt_response = response

        return response

    def predict(self, num_points=50):
        """
        Predicts using pre-fitted model, optionally with new data

        Once the model has been fit, predicts for num_steps using

        Parameters
        ----------
        num_points: int /50/
            Number of points (steps) to predict into the future

        Returns
        -------
            pred_r: list of float
                Predictions generated from the training data

        """

        if self.modeltype == 'Numpy':
            pred_r = prediction.rec_forecast_np(y=self.df,
                                                model=self.model,
                                                window_length=self.window_length,
                                                feature_names=self.best_features,
                                                rolling_window=self.rolling_window,
                                                n_steps=num_points,
                                                freq=self.freq)
        else:
            pred_r = prediction.rec_forecast(y=self.df,
                                             model=self.model,
                                             window_length=self.window_length,
                                             feature_names=self.best_features,
                                             rolling_window=self.rolling_window,
                                             n_steps=num_points,
                                             freq=self.freq)

        self.pred_r = pred_r
        return pred_r

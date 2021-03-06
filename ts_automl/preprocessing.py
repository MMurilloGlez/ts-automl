"""A module containing all preprocessing steps for time series analysis


Module containing all necessary steps to convert an univariate input time
series with a constant frequency into a multivariate dataframe with lags and
features, so that conventional regression techniques can be applied.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skits.preprocessing import HorizonTransformer
import lightgbm as lgb

scaler = StandardScaler()


def ts_split(data, test_size=100):
    """split training data into train and val"""

    """
    Splits input dataframe into train and test. Works for both univariate and
    multivariable dataframes, in case the split is done after generating
    the features and lags.

    Parameters
    ----------
    data: pd.Dataframe
        Input dataframe containing the time series. Index must be a datetime
        object.
    val_size: float {0.1, other}
        Number of values dedicated to the validation portion of the dataset.
        Can either be a proportion of dataset size or a set number of points.

    Returns
    -------
    pd.Dataframe x2
        Two dataframes, each containing the proportion of the series dedicated
        to the train and validation of the model respectively.

    """

    ts_train, ts_test = train_test_split(data, test_size=test_size,
                                         shuffle=False)
    return(ts_train, ts_test)

# -----------------------------------------------------------------------------


class Lags(TransformerMixin):
    """Class containing methods to create the lagged time series"""

    """
    A class that defines a transformer to create the lagged time series. Based
    on the TransformerMixin class from the scikit-learn library. so it supports
    the fit, transform and fit_transform arguments.

    Parameters
    ----------
    lag: int {2, other}
        Number of lags to be computed for each step in the time series.
    step: int {1, other}
        Number of steps in between each lag, expressed in the frequency of the
        time series.
    dropna: bool {True, False}
        Defines wether to include or not the initial NA values in the lagged
        time series. True by default. Will cause errors if set to False.

    Returns
    -------
    pd.Dataframe
        Containing the initial time series with the lagged features.
    """
    def __init__(self, lag=2, step=1, dropna=True):
        self.lag = lag
        self.dropna = dropna
        self.step = step
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.iloc[:, 0]
        if type(X) is pd.DataFrame:
            new_dict = {}
            for col_name in X:
                for n in range(1, self.lag+1, self.step):
                    new_dict['%s_lag%d' %(col_name, n)] = X[col_name].shift(n)
            res = pd.DataFrame(new_dict, index=X.index)

        elif type(X) is pd.Series:
            the_range = range(0, self.lag+1, self.step)
            res = pd.concat([X.shift(i) for i in the_range], axis=1)
            res.columns = ['lag_%d' %i for i in the_range]
        else:
            raise TypeError(('Input data must be in the form of Pandas' +
                             ' series or dataframe' +
                             'but was %d').format(type(X)))
            print('Only works for DataFrame or Series')
            return None
        if self.dropna:
            res = res.dropna()
            res = res[res.columns[::-1]]
            return res
        else:
            res = res[res.columns[::-1]]
            return res


def lags_sample(df, window_length=5, step=1):
    X = Lags(lag=window_length, step=step).transform(df)
    X = X.iloc[:, :-1]
    return X


def minute_sample(X):
    """OneHot encoding for the minute in hour feature"""

    enc = OneHotEncoder(sparse='False',
                        categories=[list(range(0, 60))])
    array = enc.fit_transform(np.array(X.index.minute).reshape(-1, 1))
    array = array.toarray()
    columns = ['minute_'+str(i) for i in range(0, 60)]
    return pd.DataFrame(array, index=X.index, columns=columns)


def hour_sample(X):
    """Onehot encoding for the hour of day feature"""
    enc = OneHotEncoder(sparse='False',
                        categories=[list(range(0, 24))])
    array = enc.fit_transform(np.array(X.index.hour).reshape(-1, 1))
    array = array.toarray()
    columns = ['hour_'+str(i) for i in range(0, 24)]
    return pd.DataFrame(array, index=X.index, columns=columns)


def dayofweek_sample(X):
    """OneHot encoding for the day of week feature"""
    enc = OneHotEncoder(sparse='False', drop='first',
                        categories=[list(range(0, 8))])
    array = enc.fit_transform(np.array(X.index.dayofweek).reshape(-1, 1))
    array = array.toarray()
    columns = ['dayofweek_'+str(i) for i in range(1, 8)]
    return pd.DataFrame(array, index=X.index, columns=columns)


def day_sample(X):
    """OneHot encoding for the day of month feature"""
    enc = OneHotEncoder(sparse='False', drop='first',
                        categories=[list(range(0, 32))])
    array = enc.fit_transform(np.array(X.index.day).reshape(-1, 1)).toarray()
    columns = ['day_'+str(i) for i in range(1, 32)]
    return pd.DataFrame(array, index=X.index, columns=columns)


def month_sample(X):
    """OneHot encoding for month of year feature"""
    enc = OneHotEncoder(sparse='False', drop='first',
                        categories=[list(range(0, 13))])
    array = enc.fit_transform(np.array(X.index.month).reshape(-1, 1))
    array = array.toarray()
    columns = ['month_'+str(i) for i in range(1, 13)]
    return pd.DataFrame(array, index=X.index, columns=columns)


def weekend_sample(X):
    """OneHot encoding for weekend feature (Binary)"""
    saturday = X.index.dayofweek == 5
    sunday = X.index.dayofweek == 6
    enc = OneHotEncoder(sparse='False', drop='first',
                        categories=[list(range(0, 2))])
    array = enc.fit_transform(np.logical_or(saturday, sunday).reshape(-1, 1))
    array = array.toarray()
    columns = ['weekend_'+str(i) for i in range(1, 2)]
    return pd.DataFrame(array, index=X.index, columns=columns)


def mean_sample(X, rolling_window=2):
    X_rol_mean = pd.DataFrame(X.iloc[:, :].rolling(rolling_window).mean())
    X_rol_mean.rename(columns=lambda x: x.replace('lag', 'mean' + '_' +
                      str(rolling_window)),
                      inplace=True)
    return X_rol_mean


def std_sample(X, rolling_window=2):
    X_rol_std = pd.DataFrame(X.iloc[:, :].rolling(rolling_window).std())
    X_rol_std.rename(columns=lambda x: x.replace('lag', 'std' + '_' +
                     str(rolling_window)),
                     inplace=True)
    return X_rol_std


def min_sample(X, rolling_window=2):
    X_rol_min = pd.DataFrame(X.iloc[:, :].rolling(rolling_window).min())
    X_rol_min.rename(columns=lambda x: x.replace('lag', 'min' + '_' +
                     str(rolling_window)),
                     inplace=True)
    return X_rol_min


def max_sample(X, rolling_window=2):
    X_rol_max = pd.DataFrame(X.iloc[:, :].rolling(rolling_window).max())
    X_rol_max.rename(columns=lambda x: x.replace('lag', 'max' + '_' +
                     str(rolling_window)),
                     inplace=True)
    return X_rol_max


def quantile_sample(X, rolling_window=2):
    X_q = pd.DataFrame(X.iloc[:, :].rolling(rolling_window).quantile(0.5))
    X_q.rename(columns=lambda x: x.replace('lag', 'quantile' + '_' +
               str(rolling_window)),
               inplace=True)
    return X_q


# -----------------------------------------------------------------------------


def create_sample_feat(X, window_length=5, features=[],
                       rolling_window=[2]):
    lags_X = lags_sample(X, window_length=window_length, step=1)
    df = lags_X.copy()
    for f in features:
        if rolling_window:
            for rol in rolling_window:
                aux = switch_sample_features(f, X, lags_X, rol)
                df = pd.concat([df, aux], axis=1).dropna()
        else:
            aux = switch_sample_features(f, X, lags_X, [])
            df = pd.concat([df, aux], axis=1).dropna()
    return df


def switch_sample_features(value, X, lags_X, rolling_window):
    """Switch for selecting sample features to add to the initial time series


    A switch, that takes as input the feature to be calculated

    Parameters
    ----------
    value: list of str

    X: pd.Dataframe

    lags_X: list of int

    rolling_window: list of int


    Returns
    -------
    function call
        Function call to corresponding sample feature creation
    """
    return {
        'mean': lambda: mean_sample(lags_X, rolling_window),
        'std': lambda: std_sample(lags_X, rolling_window),
        'max': lambda: max_sample(lags_X, rolling_window),
        'min': lambda: min_sample(lags_X, rolling_window),
        'quantile': lambda: quantile_sample(lags_X, rolling_window),
        'minute': lambda: minute_sample(X),
        'hour': lambda: hour_sample(X),
        'dayofweek': lambda: dayofweek_sample(X),
        'day': lambda: day_sample(X),
        'month': lambda: month_sample(X),
        'weekend': lambda: weekend_sample(X)

    }.get(value)()


# -----------------------------------------------------------------------------

def create_horizon(df, horizon):
    y = df.copy()
    ht = HorizonTransformer(horizon=horizon+1)
    y_horizon = ht.fit_transform(y.iloc[:, 0])
    y_horizon = pd.DataFrame(y_horizon[:-horizon, :], index=y.index[horizon:])
    name = 't+'+str(horizon)
    y_horizon = pd.DataFrame({name: y_horizon.iloc[:, -1]})

    return y_horizon


def feature_selection(X, y, num_features):
    clf = lgb.LGBMRegressor(n_estimators=40).fit(X, y)
    best_indx_col = clf.feature_importances_.argsort()[-num_features:]
    best_features = list(X.columns[best_indx_col])
    return order_by_other_list(X.columns, best_features)


def order_by_other_list(list_order, list_tobe_order):
    d = {k: v for v, k in enumerate(list_order)}
    list_tobe_order.sort(key=d.get)
    return list_tobe_order

"""A module containing all preprocessing steps for time series analysis"""

"""

Module containing all necessary steps to convert an univariate input time series
with a constant frequency into a multivariate dataframe with lags and features,
so that conventional regression techniques can be applied.

Parameters
----------
data: pd.Dataframe
    Input time series data with constant frequency. interpolated if necessary by
    the data import functions.
val_split: float {0.1, other}
    Validation split size. Can be either a number indicating number of samples
    for validation or a decimal, indicating proportion of values for validation.
lag_num: int
    Number of lags to compute from the time series
features: list of str
    Features to compute for each lag or set of lags
rolling_window: int
    Size of rolling window for features such as moving average

Returns
-------
pd.Dataframe x2
    Two dataframes containing the generated features and lags. One containing 
    the validation data and one containing the train data.

Raises
------

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import scipy

def ts_split(data, val_size=0.1):
    """split training data into train and val"""

    """
    Splits input dataframe into train and test. Works for both univariate and
    multivariable dataframes, in case the split is done after generating the 
    features and lags.

    Parameters
    ----------
    data: pd.Dataframe
        Input dataframe containing the time series. Index must be a datetime
        object.
    val_size: float {0.1, other}
        Number of values dedicated to the validation portion of the dataset. Can
        either be a proportion of dataset size or a set number of points.
    
    Returns
    -------
    pd.Dataframe x2
        Two dataframes, each containing the proportion of the series dedicated 
        to the train and validation of the model respectively.
    
    """

    ts_train, ts_val = train_test_split(data, test_size=val_size, shuffle=False)
    return(ts_train, ts_test)

# Extract Features -------------------------------------------------------------

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
    

#-------------------------------------------------------------------------------

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
        X = X.iloc[:,0]                                                         
        if type(X) is pd.DataFrame:                                             
            new_dict={}                                                        
            for col_name in X:                                                 
                for l in range(1,self.lag+1, self.step):                    
                    new_dict['%s_lag%d' %(col_name,l)]=X[col_name].shift(l) 
            res=pd.DataFrame(new_dict,index=X.index)

        elif type(X) is pd.Series:
            the_range=range(0,self.lag+1, self.step)
            res=pd.concat([X.shift(i) for i in the_range],axis=1)
            res.columns=['lag_%d' %i for i in the_range]
        else:
            raise TypeError('Input data must be in the form of Pandas series' +
            'or dataframe but was %d'.format(type(X)))   
            print('Only works for DataFrame or Series')
            return None
        if self.dropna:
            res = res.dropna()
            res = res[res.columns[::-1]] 
            return res
        else:
            res = res[res.columns[::-1]]
            return res

def lags_sample(df, window_length = 5, step = 1):
    X = Lags(lag = window_length, step=step).transform(df) 
    X = X.iloc[:, :-1]
    return X   


def minute_sample(X):
    """OneHot encoding for the minute in hour feature"""

    enc = OneHotEncoder(sparse='False', drop='first',
                        categories=[list(range(0,61))])
    array = enc.fit_transform(np.array(X.index.minute).reshape(-1,1)).toarray()
    columns = ['minute_'+str(i) for i in range(1,61)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def hour_sample(X):
    """Onehot encoding for the hour of day feature"""
    enc = OneHotEncoder(sparse='False', drop='first', 
                        categories=[list(range(0,25))])
    array = enc.fit_transform(np.array(X.index.hour).reshape(-1,1)).toarray()
    columns = ['hour_'+str(i) for i in range(1,25)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def dayofweek_sample(X):
    """OneHot encoding for the day of week feature"""
    enc = OneHotEncoder(sparse='False', drop='first', 
                        categories=[list(range(0,8))])
    array = enc.fit_transform(np.array(X.index.dayofweek).reshape(-1,1))
    array = array.toarray()
    columns = ['dayofweek_'+str(i) for i in range(1,8)]
    return pd.DataFrame(array, index = X.index, columns = columns)

def day_sample(X):
    """OneHot encoding for the day of month feature"""
    enc = OneHotEncoder(sparse='False', drop='first',
                        categories=[list(range(0,32))])
    array = enc.fit_transform(np.array(X.index.day).reshape(-1,1)).toarray()
    columns = ['day_'+str(i) for i in range(1,32)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def month_sample(X):
    """OneHot encoding for month of year feature"""
    enc = OneHotEncoder(sparse='False', drop='first', 
                        categories=[list(range(0,13))])
    array = enc.fit_transform(np.array(X.index.month).reshape(-1,1)).toarray()
    columns = ['month_'+str(i) for i in range(1,13)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def quarter_sample(X):
    """OneHot encoding for quarter of year feature"""
    enc = OneHotEncoder(sparse='False', drop='first', 
                        categories=[list(range(0,5))])
    array = enc.fit_transform(np.array(X.index.quarter).reshape(-1,1)).toarray()
    columns = ['quarter_'+str(i) for i in range(1,5)]
    return pd.DataFrame(array, index = X.index, columns = columns)

def weekofyear_sample(X):
    """OneHot encoding for week number feature"""
    enc = OneHotEncoder(sparse='False', drop='first', 
                        categories=[list(range(0,53))])
    array = enc.fit_transform(np.array(X.index.isocalendar().week).reshape(-1,1))
    array = array.toarray()
    columns = ['weekofyear_'+str(i) for i in range(1,53)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def weekend_sample(X):
    """OneHot encoding for weekend feature (Binary)"""
    saturday = X.index.dayofweek == 5
    sunday = X.index.dayofweek == 6
    enc = OneHotEncoder(sparse='False', drop='first', 
                        categories=[list(range(0,2))])
    array = enc.fit_transform(np.logical_or(saturday, sunday).reshape(-1,1))
    array = array.toarray()
    columns = ['weekend_'+str(i) for i in range(1,2)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def mean_sample(X, rolling_window = 2):
    X_rol_mean = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).mean()) 
    X_rol_mean.rename(columns=lambda x: x.replace('lag', 'mean'+'_'+
                      str(rolling_window)), 
                      inplace=True)
    return X_rol_mean


def std_sample(X, rolling_window = 2):
    X_rol_std = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).std()) 
    X_rol_std.rename(columns=lambda x: x.replace('lag', 'std'+'_'+
                     str(rolling_window)), 
                     inplace=True)
    return X_rol_std


def min_sample(X, rolling_window = 2):
    X_rol_min = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).min())
    X_rol_min.rename(columns=lambda x: x.replace('lag', 'min'+'_'+
                     str(rolling_window)), 
                     inplace=True)
    return X_rol_min


def max_sample(X, rolling_window = 2):
    X_rol_max = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).max())
    X_rol_max.rename(columns=lambda x: x.replace('lag', 'max'+'_'+
                     str(rolling_window)), 
                     inplace=True)
    return X_rol_max


def quantile_sample(X, rolling_window = 2):
    X_q = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).quantile(0.5))
    X_q.rename(columns=lambda x: x.replace('lag', 'quantile'+'_'+
               str(rolling_window)), 
               inplace=True)
    return X_q


def iqr_sample(series, rolling_window = 2):
    r_in=[]
    for i in list(range(len(series)-rolling_window+1)):
        r_in.append(scipy.stats.iqr(series.iloc[i:rolling_window+i,:], axis=0))
    r_in = pd.DataFrame(r_in)
    r_in.index = series.iloc[rolling_window-1:,:].index
    r_in.columns = series.columns
    r_in.rename(columns=lambda x: x.replace('lag', 'iqr'+'_'+
                str(rolling_window)), 
                inplace=True)
    return r_in

#-------------------------------------------------------------------------------

def switch_sample_features(value, X, lags_X, rolling_window):
    """Switch for selecting sample features to add to the initial time series"""

    """
    A switch, that takes as input the 
    
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
        'mean': lambda : mean_sample(lags_X, rolling_window),
        'std': lambda : std_sample(lags_X, rolling_window),
        'max': lambda : max_sample(lags_X, rolling_window),
        'min': lambda : min_sample(lags_X, rolling_window),
        'quantile': lambda : quantile_sample(lags_X, rolling_window),
        'iqr': lambda : iqr_sample(lags_X, rolling_window),
        #-----------------------------------------
        'minute': lambda : minute_sample(X),
        'hour': lambda : hour_sample(X),
        'dayofweek': lambda : dayofweek_sample(X),
        'day': lambda : day_sample(X),
        'month': lambda : month_sample(X),
        'quarter': lambda : quarter_sample(X),
        'weekofyear': lambda : weekofyear_sample(X),
        'weekend': lambda : weekend_sample(X)
        
    }.get(value)()


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


def switch_optimization(value, regressor):
    return {
        'tpe': lambda : tpe_optimization(),
        'pso': lambda : pso_optimization(),
    }.get(value)()

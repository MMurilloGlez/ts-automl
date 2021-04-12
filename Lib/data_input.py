"""

A module to input data from a csv file, correctly trasform the data within
and save the time series in an univariate dataframe with a datetime index
This module is by default tuned to time series output by IDBoxRT, but 
parameters may be changed accordingly to allow for other time series.

Parameters
----------
filename: str
    Name of the file or path from which to read data
targetcol: str {"VALUE", other}, optional
    Target value column name for input file
datecol: str {"DATE", other}, optional
    Date column name for input file.
sep: str {";", other}, optional
    Column separator for csv files.
decimal: str {",", other}, optional
    decimal place indicator for data values.
date_format: str {"%d/%m/%Y %H:%M:%S.%f", other}, optional
    format of the time series date column.
freq: str {"1T", other}, optional
    frequency at which to read the time series.

Returns
-------
pd.Dataframe
    Univariate time series interpolated to a given frequency

Raises
------

ValueError
    when the expected frequency is smaller than the one given by the
    input file data.
NotImplementedError
    when the filetype or file extension doesn´t conform to the ones implemented.

"""

import numpy as np
import pandas as pd

def read_data(filename, targetcol='VALUE', datecol='DATE', sep=';',
              decimal=',', date_format="%d/%m/%Y %H:%M:%S.%f", freq='1T'):

    """Read data from file and parse it"""

    """
    
    Reads data from csv or xls file, parsing all date values while doing so.
    Calls parsedates to parse the file and interpolate to the given frequency

    Returns
    -------

    pd.Dataframe
        Univariate dataframe containing the interpolated time series.
    
    """

    if filename[-3:] == 'csv' or filename[-3:] == 'txt':
        file_ = pd.read_table(filename, sep=sep, decimal=decimal,
                              parse_dates=False)
        file_parsed = parsedates(file_, date_format=date_format, freq=freq,
                                 datecol=datecol)
    elif filename[-3:] == 'xls' or filename[-3:] == 'lsx':
        file_ = pd.read_excel(filename, parse_dates=False)
        file_parsed = parsedates(file_, date_format=date_format, freq=freq,
                                 datecol=datecol)
    else:
        raise NotImplementedError("This file type is not supported")

    return(file_parsed[targetcol].to_frame())


def parsedates(file_, date_format, freq, datecol):

    """Parse dates in time series. interpolate to given frequency"""

    """
    Parse the datetime data from the input time series. Interpolate values
    of said time series to given frequency freq.

    Returns
    -------
    
    pd.Dataframe
        Univariate dataframe containing the interpolated time series.
    
    """

    freq = pd.to_timedelta(freq)
    datetime_i = pd.to_datetime(file_[datecol], format=date_format)
    print(type(datetime_i.iloc[-1]))
    if freq < (datetime_i.iloc[1] - datetime_i.iloc[0]):
        raise ValueError('Expected frequency smaller than dataset information')

    file_.index = datetime_i
    desired_index = pd.date_range(start=datetime_i.iloc[0],
                                  end=datetime_i.iloc[-1], freq=freq)
    file_int = file_.reindex(file_.index.union(desired_index))
    file_int = file_int.interpolate(method='time').reindex(desired_index)

    return(file_int)


import pandas as pd                                     ## Importacion de librerias
import numpy as np
import scipy
import pywt                                             ## Pywavelets. wavelet transforms in python

def lag_feature(data, num):                             ## Lag formado por los ultimos 'num' elementos
    return data[-num]

#-----------------------                                ## Encoding one hot, 1 si yes, 0 si el resto. Tenemos en cuenta el minuto $VAL$?¿
def minute_feature(date, num):                          ## Extraer minuto de datetimestr
    minute = date.minute
    if minute == num:
        return 1
    else:
        return 0                                        ## Si no se resuelve un numero de minuto, se devuelve 0

    
def hour_feature(date, num):                            ## Extraer hora de 
    hour = date.hour
    if hour == num:
        return 1
    else:
        return 0


def dayofweek_feature(date, num):                       ## Extraer dia de la semana
    dayofweek = date.dayofweek
    if dayofweek == num:
        return 1
    else:
        return 0

    
def day_feature(date, num):                             ## extraer dia de mes
    day = date.day
    if day == num:
        return 1
    else:
        return 0
    

def month_feature(date, num):                           ## extraer mes
    month = date.month
    if month == num:
        return 1
    else:
        return 0

    
def quarter_feature(date, num):                         ## extraer quarter q1 q2 q3 q4
    quarter = date.quarter
    if quarter == num:
        return 1
    else:
        return 0
    

def weekofyear_feature(date, num):                      ## extraer numero de semana 1 a 52
    weekofyear = date.weekofyear
    if weekofyear == num:
        return 1
    else:
        return 0

    
def weekend_feature(date, num):                         ## es fin de semana o no?
    dayofweek = date.dayofweek
    if dayofweek == 5 or dayofweek == 6:
        return 1
    else:
        return 0
    
#-----------------------

def mean_feature(data, rolling_window, num):            ## Media de la rolling window
    if num == 1:                                        ## si el lag es 1 se calcula para ese lag
        return np.mean(data[-rolling_window:])
    else:
        return np.mean(data[-(rolling_window-1+num):-(num-1)])
    
    
def std_feature(data, rolling_window, num):
    if num == 1:
        return np.std(data[-rolling_window:], ddof=1)
    else:
        return np.std(data[-(rolling_window-1+num):-(num-1)], ddof=1)
    
    
def max_feature(data, rolling_window, num):
    if num == 1:
        return np.max(data[-rolling_window:])
    else:
        return np.max(data[-(rolling_window-1+num):-(num-1)])

    
def min_feature(data, rolling_window, num):
    if num == 1:
        return np.min(data[-rolling_window:])
    else:
        return np.min(data[-(rolling_window-1+num):-(num-1)])
    

def quantile_feature(data, rolling_window, num):
    if num == 1:
        return np.quantile(data[-rolling_window:], 0.5)
    else:
        return np.quantile(data[-(rolling_window-1+num):-(num-1)], 0.5)
    
    
def iqr_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.iqr(data[-rolling_window:], rng = [25, 75])
    else:
        return scipy.stats.iqr(data[-(rolling_window-1+num):-(num-1)], rng = [25, 75])

    
def entropy_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.entropy(data[-rolling_window:])
    else:
        return scipy.stats.entropy(data[-(rolling_window-1+num):-(num-1)])
    
    
def trimmean_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.trim_mean(data[-rolling_window:], 0.02)
    else:
        return scipy.stats.trim_mean(data[-(rolling_window-1+num):-(num-1)], 0.02)
    
    
def variation_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.variation(data[-rolling_window:])
    else:
        return scipy.stats.variation(data[-(rolling_window-1+num):-(num-1)])
    
    
def hmean_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.hmean(data[-rolling_window:])
    else:
        return scipy.stats.hmean(data[-(rolling_window-1+num):-(num-1)])
    
    
def gmean_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.gmean(data[-rolling_window:])
    else:
        return scipy.stats.gmean(data[-(rolling_window-1+num):-(num-1)])
    
    
def mad_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.median_abs_deviation(data[-rolling_window:])
    else:
        return scipy.stats.median_abs_deviation(data[-(rolling_window-1+num):-(num-1)])
    
    
# def gstd_feature(data, rolling_window, num):
#     if num == 1:
#         return scipy.stats.gstd(np.abs(data[-rolling_window:]))
#     else:
#         return scipy.stats.gstd(np.abs(data[-(rolling_window-1+num):-(num-1)]))
    
    
def tvar_feature(data, rolling_window, num):                                ## Tail value at risk
    if num == 1:
        return scipy.stats.tvar(data[-rolling_window:])
    else:
        return scipy.stats.tvar(data[-(rolling_window-1+num):-(num-1)])
    
    
def kurtosis_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.kurtosis(data[-rolling_window:])
    else:
        return scipy.stats.kurtosis(data[-(rolling_window-1+num):-(num-1)])
    
    
def sem_feature(data, rolling_window, num):
    if num == 1:
        return scipy.stats.sem(data[-rolling_window:])
    else:
        return scipy.stats.sem(data[-(rolling_window-1+num):-(num-1)]) 
    
    
def wav_feature(data, rolling_window, num):                                 ## Continuous wavelet transform
    if num == 1:
        term, coeff = pywt.cwt(data[-rolling_window:], np.arange(1,2), "morl", method = "conv")
        return abs(term[0][0])
    else:
        term, coeff = pywt.cwt(data[-(rolling_window-1+num):-(num-1)], np.arange(1,2), wavelet = "morl", method = "conv")
        return abs(term[0][0])

import pandas as pd
import numpy as np
import scipy
import pywt

from sklearn.base import TransformerMixin                                               ## Mixin class for all transformers in scikit-learn. Fits to X and y with fit_params. Returns new x
from sklearn.preprocessing import OneHotEncoder                                         ## Encodes categorical features into one-hot automatically


class Lags(TransformerMixin):                                                           ## Clase de lags, hereda de la clase TransformerMixin
    def __init__(self, lag=2, step=1, dropna=True):                                     ## por defecto 2 lags y de 1 en 1
        self.lag = lag
        self.dropna = dropna
        self.step = step
        pass
    def fit(self, X, y=None):                                                           ## fit se devuelve a si misma
        return self
    def transform(self, X, y=None):                                                     ## Transformacion
        X = X.iloc[:,0]                                                                 ## Convertimos a un df unidimensional (or series for that matter)
        if type(X) is pd.DataFrame:                                                     ## si es dataframe
            new_dict={}                                                                 ## Creamos un dict vacio
            for col_name in X:                                                          ## por cada nombre de columna en los datos iniciales
                #new_dict[col_name]=X[col_name]
                # create lagged Series
                for l in range(1,self.lag+1, self.step):                                ## por cada lag que haya que crear con el step que se necesite
                    new_dict['%s_lag%d' %(col_name,l)]=X[col_name].shift(l)             ## creamos los lags con formato numero _lag
            res=pd.DataFrame(new_dict,index=X.index)                                    ## creamos dataframe nuevo con estas cosas

        elif type(X) is pd.Series:                                                      ## si es serie
            the_range=range(0,self.lag+1, self.step)                                    ## de 0 a numero de lags
            res=pd.concat([X.shift(i) for i in the_range],axis=1)                       ## concatenamos los lags a cada fila del nuevo dataframe 
            res.columns=['lag_%d' %i for i in the_range]                                ## como nombre de columna tal
        else:
            raise TypeError('Input data must be in the form of Pandas series or dataframe but was %d'.format(type(X)))              #################### Añadido error
            print('Only works for DataFrame or Series')
            return None
        if self.dropna:                                                                 ## si hay que quitar na
            res = res.dropna()                                                          ## quitamos na
            res = res[res.columns[::-1]]                                                ## res = todo menos la ultima columna 
            return res
        else:
            res = res[res.columns[::-1]]                                                ## res = todo menos la ultima columna
            return res

def lags_sample(df, window_length = 5, step = 1):                                       ## lags de un sample
    X = Lags(lag = window_length, step=step).transform(df)                              ## lags dentro de lags para crear
    X = X.iloc[:, :-1]                                                                  ## nos quedamos todo menos la ultima columna
    return X   

#------------------------------                                                         ## Seccion de autoencoders, crea un autoencoder segun los datos iniciales, ya sean por minutos horas, segundo quarters etc...

def minute_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,61))])
    array = enc.fit_transform(np.array(X.index.minute).reshape(-1,1)).toarray()
    columns = ['minute_'+str(i) for i in range(1,61)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def hour_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,25))])
    array = enc.fit_transform(np.array(X.index.hour).reshape(-1,1)).toarray()
    columns = ['hour_'+str(i) for i in range(1,25)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def dayofweek_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,8))])
    array = enc.fit_transform(np.array(X.index.dayofweek).reshape(-1,1)).toarray()
    columns = ['dayofweek_'+str(i) for i in range(1,8)]
    return pd.DataFrame(array, index = X.index, columns = columns)

def day_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,32))])
    array = enc.fit_transform(np.array(X.index.day).reshape(-1,1)).toarray()
    columns = ['day_'+str(i) for i in range(1,32)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def month_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,13))])
    array = enc.fit_transform(np.array(X.index.month).reshape(-1,1)).toarray()
    columns = ['month_'+str(i) for i in range(1,13)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def quarter_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,5))])
    array = enc.fit_transform(np.array(X.index.quarter).reshape(-1,1)).toarray()
    columns = ['quarter_'+str(i) for i in range(1,5)]
    return pd.DataFrame(array, index = X.index, columns = columns)

def weekofyear_sample(X):
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,53))])
    array = enc.fit_transform(np.array(X.index.isocalendar().week).reshape(-1,1)).toarray()
    columns = ['weekofyear_'+str(i) for i in range(1,53)]
    return pd.DataFrame(array, index = X.index, columns = columns)


def weekend_sample(X):
    saturday = X.index.dayofweek == 5
    sunday = X.index.dayofweek == 6
    enc = OneHotEncoder(sparse='False', drop='first', categories=[list(range(0,2))])
    array = enc.fit_transform(np.logical_or(saturday, sunday).reshape(-1,1)).toarray()
    columns = ['weekend_'+str(i) for i in range(1,2)]
    return pd.DataFrame(array, index = X.index, columns = columns)

#------------------------------                                                         ## Seccion de formulas de samples. se realizan sobre una rolling window de 2 por defecto, salvo si lo dice el usuario.

def mean_sample(X, rolling_window = 2):
    X_rol_mean = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).mean()) 
    X_rol_mean.rename(columns=lambda x: x.replace('lag', 'mean'+'_'+str(rolling_window)), inplace=True)
    return X_rol_mean


def std_sample(X, rolling_window = 2):
    X_rol_std = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).std()) 
    X_rol_std.rename(columns=lambda x: x.replace('lag', 'std'+'_'+str(rolling_window)), inplace=True)
    return X_rol_std


def min_sample(X, rolling_window = 2):
    X_rol_min = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).min())
    X_rol_min.rename(columns=lambda x: x.replace('lag', 'min'+'_'+str(rolling_window)), inplace=True)
    return X_rol_min


def max_sample(X, rolling_window = 2):
    X_rol_max = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).max())
    X_rol_max.rename(columns=lambda x: x.replace('lag', 'max'+'_'+str(rolling_window)), inplace=True)
    return X_rol_max


def quantile_sample(X, rolling_window = 2):
    X_q = pd.DataFrame(X.iloc[:,:].rolling(rolling_window).quantile(0.5))
    X_q.rename(columns=lambda x: x.replace('lag', 'quantile'+'_'+str(rolling_window)), inplace=True)
    return X_q


def iqr_sample(series, rolling_window = 2):
    r_in=[]
    for i in list(range(len(series)-rolling_window+1)):
        r_in.append(scipy.stats.iqr(series.iloc[i:rolling_window+i,:], axis=0))
    r_in = pd.DataFrame(r_in)
    r_in.index = series.iloc[rolling_window-1:,:].index
    r_in.columns = series.columns
    r_in.rename(columns=lambda x: x.replace('lag', 'iqr'+'_'+str(rolling_window)), inplace=True)
    return r_in


def entropy_sample(series, rolling_window=2):
    ent=[]
    for i in list(range(len(series)-rolling_window+1)):
        ent.append(scipy.stats.entropy(series.iloc[i:rolling_window+i,:]))
    ent = pd.DataFrame(ent)
    ent.index = series.iloc[rolling_window-1:,:].index
    ent.columns = series.columns
    ent.rename(columns=lambda x: x.replace('lag', 'entropy'+'_'+str(rolling_window)), inplace=True)
    return ent


def trimmean_sample(series, rolling_window = 2):
    tmean=[]
    for i in list(range(len(series)-rolling_window+1)):
        tmean.append(scipy.stats.trim_mean(series.iloc[i:rolling_window+i,:], 0.02))
    tmean = pd.DataFrame(tmean)
    tmean.index = series.iloc[rolling_window-1:,:].index
    tmean.columns = series.columns
    tmean.rename(columns=lambda x: x.replace('lag', 'trimmean'+'_'+str(rolling_window)), inplace=True)
    return tmean


def variation_sample(series, rolling_window=2):
    variation=[]
    for i in list(range(len(series)-rolling_window+1)):
        variation.append(scipy.stats.variation(series.iloc[i:rolling_window+i,:]))
    variation = pd.DataFrame(variation)
    variation.index = series.iloc[rolling_window-1:,:].index
    variation.columns = series.columns
    variation.rename(columns=lambda x: x.replace('lag', 'variation'+'_'+str(rolling_window)), inplace=True)
    return variation


def hmean_sample(series, rolling_window=2):
    hmean=[]
    for i in list(range(len(series)-rolling_window+1)):
        hmean.append(scipy.stats.hmean(series.iloc[i:rolling_window+i,:]))
    hmean = pd.DataFrame(hmean)
    hmean.index = series.iloc[rolling_window-1:,:].index
    hmean.columns = series.columns
    hmean.rename(columns=lambda x: x.replace('lag', 'hmean'+'_'+str(rolling_window)), inplace=True)
    return hmean


def gmean_sample(series, rolling_window=2):
    gmean=[]
    for i in list(range(len(series)-rolling_window+1)):
        gmean.append(scipy.stats.gmean(series.iloc[i:rolling_window+i,:]))
    gmean = pd.DataFrame(gmean)
    gmean.index = series.iloc[rolling_window-1:,:].index
    gmean.columns = series.columns
    gmean.rename(columns=lambda x: x.replace('lag', 'gmean'+'_'+str(rolling_window)), inplace=True)
    return gmean


def mad_sample(series, rolling_window=2):
    m_abs=[]
    for i in list(range(len(series)-rolling_window+1)):
        m_abs.append(scipy.stats.median_abs_deviation(series.iloc[i:rolling_window+i,:]))
    m_abs = pd.DataFrame(m_abs)
    m_abs.index = series.iloc[rolling_window-1:,:].index
    m_abs.columns = series.columns
    m_abs.rename(columns=lambda x: x.replace('lag', 'mad'+'_'+str(rolling_window)), inplace=True)
    return m_abs


# def gstd_sample(series, rolling_window=2):
#     std=[]
#     for i in list(range(len(series)-rolling_window+1)):
#         std.append(scipy.stats.gstd(np.abs(series.iloc[i:rolling_window+i,:])))
#     std = pd.DataFrame(std)
#     std.index = series.iloc[rolling_window-1:,:].index
#     std.columns = series.columns
#     std.rename(columns=lambda x: x.replace('lag', 'gstd'+'_'+str(rolling_window)), inplace=True)
#     return std


def tvar_sample(series, rolling_window=2):
    t_var=[]
    for i in list(range(len(series)-rolling_window+1)):
        t_var.append(scipy.stats.tvar(series.iloc[i:rolling_window+i,:]))
    t_var = pd.DataFrame(t_var)
    t_var.index = series.iloc[rolling_window-1:,:].index
    t_var.columns = series.columns
    t_var.rename(columns=lambda x: x.replace('lag', 'tvar'+'_'+str(rolling_window)), inplace=True)
    return t_var


def kurtosis_sample(series, rolling_window=2):
    kurt=[]
    for i in list(range(len(series)-rolling_window+1)):
        kurt.append(scipy.stats.kurtosis(series.iloc[i:rolling_window+i,:]))
    kurt = pd.DataFrame(kurt)
    kurt.index = series.iloc[rolling_window-1:,:].index
    kurt.columns = series.columns
    kurt.rename(columns=lambda x: x.replace('lag', 'kurtosis'+'_'+str(rolling_window)), inplace=True)
    return kurt


def sem_sample(series, rolling_window=2):
    std_e=[]
    for i in list(range(len(series)-rolling_window+1)):
        std_e.append(scipy.stats.sem(series.iloc[i:rolling_window+i,:]))
    std_e = pd.DataFrame(std_e)
    std_e.index = series.iloc[rolling_window-1:,:].index
    std_e.columns = series.columns
    std_e.rename(columns=lambda x: x.replace('lag', 'sem'+'_'+str(rolling_window)), inplace=True)
    return std_e


def wav_sample(series, rolling_window=2):
    wav = []
    for i in list(range(len(series)-rolling_window+1)):
        term, coeff = pywt.cwt(series.iloc[i:rolling_window+i,:], np.arange(1,2), wavelet = "morl", method = "conv", axis=0)
        wav.append(abs(term[0][0]))
    wav = pd.DataFrame(wav)
    wav.index = series.iloc[rolling_window-1:,:].index
    wav.columns = series.columns
    wav.rename(columns=lambda x: x.replace('lag', 'wav'+'_'+str(rolling_window)), inplace=True)
    return wav
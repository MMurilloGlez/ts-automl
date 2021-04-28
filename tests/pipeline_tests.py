import os
import sys
import numpy as np
import pandas as pd

from timeit import default_timer as timer

import data_input
import preprocessing
import feature_selection
import prediction
import plotting

start = timer()

# Change working directory to the one containing the script

os.chdir(os.path.dirname(sys.argv[0]))

# Read Seat.csv data

df = data_input.read_data(filename='Seat.csv',
                          freq='15T',
                          targetcol='GAS DIRECTO [kWh]',
                          datecol='MSJO_DATUM',
                          sep=',',
                          decimal='.',
                          date_format="%d/%m/%Y %H:%M")

df = df.iloc[-2000:,:]
# Parameters

points = 50
window_length = 100
rolling_window = [5,10,20]
horizon = 1
step = 1
freq = '15T'

# Generate training features for data.

y_train, y_test = preprocessing.ts_split(df, test_size=50)

end = timer()

print("ELAPSED TIME: " + str(end-start))

features = ['mean', 'std', 'max', 'min', 'minute']

X_train = preprocessing.create_sample_features(y_train, 
                                               window_length=window_length, 
                                               features=features, 
                                               rolling_window=rolling_window)

X_train = X_train.loc[:,~X_train.columns.duplicated()]
X_train = X_train.iloc[-2000:,:]

y_horizon = preprocessing.create_horizon(y_train, horizon).loc[X_train.index[0]:,:]

# Feature selection

selected_feat = 40
best_features = preprocessing.feature_selection(X_train, 
                                                y_horizon.values.ravel(), 
                                                selected_feat)

X_train_selec = X_train.loc[:, best_features]

end = timer()

print("ELAPSED TIME: " + str(end-start))

# Predictions

## Funcionan knn y lgb
## keras funciona con recursive_forecast_np y to_numpy.reshape(-1,1)

lgbregressor = prediction.LGB_Model
lgbregressor.fit(X=X_train_selec, 
                 y=y_horizon.values.ravel())

end = timer()

print("ELAPSED TIME: " + str(end-start))


# A partir de aqui en el predict


pred = prediction.recursive_forecast(y=y_train, 
                          model=lgbregressor, 
                          window_length=window_length,
                          feature_names=best_features,
                          rolling_window=rolling_window,
                          n_steps=points,
                          freq=freq)

end = timer()

print("ELAPSED TIME: " + str(end-start))


# A partir de aqui el plot

plotting.plot_test_pred(y_test, pred)
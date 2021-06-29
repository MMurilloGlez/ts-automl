# ts-automl

### An automated machine learning library for time series forecasting in python

[![codecov](https://codecov.io/gh/MMurilloGlez/ts-automl/branch/master/graph/badge.svg?token=N85DT683O3)](https://codecov.io/gh/MMurilloGlez/ts-automl)   [![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FMMurilloGlez%2Fts-automl.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FMMurilloGlez%2Fts-automl?ref=badge_shield)    [![DOI](https://zenodo.org/badge/364524469.svg)](https://zenodo.org/badge/latestdoi/364524469)

## Installation

Create a python 3.8 virtual environment into which to install the library

```python
conda create -n ts_automl_test python=3.8
```

clone the github branch to download the installation files

```
git clone https://github.com/MMurilloGlez/ts-automl.git ts-automl
```
activate your environment and install wheel

```python
conda activate ts_automl_test
pip install wheel
```
install the library using pip install

```python
pip install './ts-automl/dist/ts_automl-0.1.4-py3-none-any.whl'
```
installation will take a while depending on the number of dependencies already on your system



## Usage


import the prediction models and execute the scripts (changing anything needed for the particular series)

```python
from ts_automl.pipelines import Pipeline
```
* for slow prediction (LSTM + LightGBM + KNN):
```python

model=Pipeline(filename='./example.csv',
               type='slow'
	       freq='15T', 
               targetcol='TARGET', 
               datecol='DATES', 
               sep=',', 
               decimal='.', 
               date_format="%d/%m/%Y %H:%M")
```

* for balanced prediction (LightGBM + KNN):
```python
model=Pipeline(filename='./example.csv',
               type='balanced'
	       freq='15T', 
               targetcol='TARGET', 
               datecol='DATES', 
               sep=',', 
               decimal='.', 
               date_format="%d/%m/%Y %H:%M")
```


* for fast prediction (KNN Model):
```python
model=Pipeline(filename='./example.csv',
               type='slow'
	       freq='15T', 
               targetcol='TARGET', 
               datecol='DATES', 
               sep=',', 
               decimal='.', 
               date_format="%d/%m/%Y %H:%M")
```

* for naïve prediction (mean of last 50 values) (not recommended except for comparison with other models): 
```python
from ts_automl.pipelines import naive_prediction
naive_prediction(filename='./example.csv', 
		freq='15T', 
                targetcol='TARGET', 
                datecol='DATES', 
                sep=',', 
                decimal='.', 
                date_format="%d/%m/%Y %H:%M")
```

Once a Pipeline has been initialized with a given model and a given dataset, the next step is to fit the model to the dataset
```python
model.fit()
```
Optionally optimization on the fitted models can be applied using Hyperopt:
```python
model.fit_opt()
```
After fitting the models a prediction can be made using the predict method of the class
```python
model.predict()
```
## Accessing data from the Pipeline classes
Details of the fit validation error, the training data or the error relative to the naïve prediction can be found inside the Pipeline class object, by accessing the following parameters:
```python
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
        df: pd.DataFrame
            Full generated dtaframe of the input time series with lags and features
        y_train, y_test: pd.Series
            Time series containing the training and test data for the series respectively
        X_Train_selec: pd.DataFrame
            DataFrame containing the best features according to the selected_feat argument
        best_features: List of str
            List containing names of the best features in the dataframe.
        
        naive: List of float
            Prediction for test uding the Naive model (if rel_metrics=True)
        model: object
            Model fitted to training data

        opt_runs: int
            Number of optimization passes to run on the hyperopt version of a given regressor
```
## Optional parameters

* for fast, balanced and slow prediction:
```python
points: int 
        Size of the horizon to predict in the future
window_length: int
        Number of points in the past to obtain lags for
rolling_window: list of int
        rolling window size to calculate the features
selected_feat: int
        Number of features to retain in feature selection.
num_datapoints: int
        Size of the training set for the models
plot: bool
        Whether to plot or not the results
error: list of str ['mape','mse','r2', 'exp_var']
        Error metrics to calculate with respect to the test data 
rel_error: bool
        Whether or not to calculate the error relative to naive forecast

```


* for naïve prediction:

```python
points: int 
        Size of the horizon to predict in the future
num_datapoints: int
        Size of the training set for the models
plot: bool
        Whether to plot or not the results
error: list of str ['mape','mse','r2', 'exp_var']
        Error metrics to calculate with respect to the test data 
```



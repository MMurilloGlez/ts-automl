# ts-automl

### An automated machine learning library for time series forecasting in python

## Installation

Create a python 3.8 virtual environment into which to install the library

```python
conda create -n ts_automl_test python=3.8
```

clone the github branch to download the installation files

```
git clone https://gitlab.corp.cic.es/CIC/IDbox/idbox-ml-i-d/ts-automl.git ts-automl
cd ts-automl
```
activate your environment and install wheel

```python
conda activate ts_automl_test
pip install wheel
```
install the library using pip install

```python
pip install './ts-automl/dist/ts_automl-0.1.0-py3-none-any.whl'
```
installation will take a while depending on the number of dependencies already on your system



## Usage

import the prediction models and execute the scripts (changing anything needed for the particular series)


* for slow prediction (LSTM Model):
```python
from ts_automl.pipelines import slow_prediction
slow_prediction(filename='./example.csv', 
		freq='15T', 
                targetcol='TARGET', 
                datecol='DATES', 
                sep=',', 
                decimal='.', 
                date_format="%d/%m/%Y %H:%M")
```

* for balanced prediction (LightGBM Model):
```python
from ts_automl.pipelines import balanced_prediction
balanced_prediction(filename='./example.csv', 
		    freq='15T', 
            	    targetcol='TARGET', 
            	    datecol='DATES', 
            	    sep=',', 
                    decimal='.', 
            	    date_format="%d/%m/%Y %H:%M")
```

* for fast prediction (KNN Model):
```python
from ts_automl.pipelines import fast_prediction
fast_prediction(filename='./example.csv', 
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

## Optional parameters

* for fast, balanced and slow prediction:
```python


```


* for naïve prediction:

```python


```



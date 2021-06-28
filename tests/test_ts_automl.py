from ts_automl import __version__
from ts_automl.pipelines import fast_prediction
from ts_automl.pipelines import balanced_prediction
from ts_automl.pipelines import slow_prediction
from ts_automl.pipelines import pipeline
from ts_automl.data_input import read_data
from ts_automl.preprocessing import create_sample_feat
from ts_automl import api
from fastapi.testclient import TestClient


def test_version():
    """
    Test for validating the package version installed.

    Will pass if version is same as the one expected by the test.
    """

    assert __version__ == '0.1.3'

# API Tests


client = TestClient(api.app)


def api_test_root():
    response = client.get("/")
    assert response.status_code == 200


def api_test_training():

    "Test the training data upload endpoint"

    params = (
        ('freq', '15T'),
        ('targetcol', 'VALUE'),
        ('datecol', 'DATE'),
        ('dateformat', '%d/%m/%Y %H:%M:%S.%f'),
        ('sep', ';'),
        ('decimal', ','),
        ('points', '50'),
        ('window_length', '100'),
        ('rolling_window', ['5', '10', '20']),
        ('horizon', '1'),
        ('step', '1'),
        ('num_datapoints', '2000'),
        ('features', ['mean', 'std', 'max', 'min', 'minute']),
        ('selected_feat', '20'),
        ('plot', 'true'),
        ('error', ["mse", "rmse",
                   "mape", "rsquare",
                   "exp_var"]),
        ('rel_error', 'true'),
        ('opt', 'false'),
        ('opt_runs', '10'),
        ('fit_type', 'fast'),
    )

    filename = "./tests/test_series/Serie4.csv"
    files = {"file": ("Serie4.csv", open(filename, "rb"),
             "application/vnd.ms-excel")}

    response = client.post("/UploadTraining/",
                           params=params,
                           files=files)

    assert response.status_code == 200


def api_test_fit():
    "Test the model fit endpoint"

    response = client.get("/Fit/")
    assert response.status_code == 200

def api_test_predict():
    "Tests the model predict endpoint"
    response = client.get("/Predict/")
    assert response.status_code == 200

# Prediction tests


def test_build_feat():
    """Test building all different features for time series"""

    df = read_data(filename='./tests/test_series/Serie4.csv',
                   freq='5T',
                   targetcol='VALUE',
                   datecol='DATE',
                   sep=';',
                   decimal=',',
                   date_format="%d/%m/%Y %H:%M:%S.%f")

    features = ["mean", "std", "max", "min", "quantile", "minute",
                "hour", "dayofweek", "day", "month", "weekend"]

    y_train = df
    X_train = create_sample_feat(y_train,
                                 window_length=10,
                                 features=features,
                                 rolling_window=[5])

    assert X_train is not None


def test_load_data():
    """Tests loading data from file using class object"""

    global model
    model = pipeline(filename='./tests/test_series/Serie1.csv',
                     type='fast',
                     freq='10T'
                     )

    assert model.df is not None


def test_fit_model():
    """Tests that the model can support the fit method"""

    model.fit()

    assert model.r_error < 100

def test_predict_model():
    "Tests that the model's predict method works"
    
    pred = model.predict()
    assert pred is not None

def test_fast_1():
    """
    Test for fast prediction time, using knn, series 1

    Uses a reference time series from test_series path, launches a prediction
    model on it using the respective model, and compares the result with the
    naive prediction for the same series. If the prediction mse is lower in
    comparison with the naive model, the test will pass
    """

    test_f_1 = fast_prediction(filename='./tests/test_series/Serie1.csv',
                               freq='10T',
                               targetcol='VALUE',
                               datecol='DATE',
                               sep=';',
                               decimal=',',
                               date_format="%d/%m/%Y %H:%M:%S.%f",
                               plot=False)

    assert test_f_1['r_error'] < 100


def test_fast_2():
    """Test for fast prediction time, using knn, series 2"""
    test_f_2 = fast_prediction(filename='./tests/test_series/Serie2.csv',
                               freq='1T',
                               targetcol='VALUE',
                               datecol='DATE',
                               sep=';',
                               decimal=',',
                               date_format="%d/%m/%Y %H:%M:%S.%f",
                               plot=False,
                               opt=False)

    assert test_f_2['r_error'] < 100


def test_fast_3():
    """Test for fast prediction time, using knn, series 3"""
    test_f_3 = fast_prediction(filename='./tests/test_series/Serie3.csv',
                               freq='15T',
                               targetcol='INSTALACIONES [kWh]',
                               datecol='MSJO_DATUM',
                               sep=';',
                               decimal='.',
                               date_format="%d/%m/%Y %H:%M",
                               plot=False)

    assert test_f_3['r_error'] < 100


def test_fast_4():
    """Test for fast prediction time, using knn, series 4"""
    test_f_4 = fast_prediction(filename='./tests/test_series/Serie4.csv',
                               freq='5T',
                               targetcol='VALUE',
                               datecol='DATE',
                               sep=';',
                               decimal=',',
                               date_format="%d/%m/%Y %H:%M:%S.%f",
                               plot=False)

    assert test_f_4['r_error'] < 100


def test_bal_1():
    """Test for balanced prediction time, using lightgbm, series 1"""
    test_b_1 = balanced_prediction(filename='./tests/test_series/Serie1.csv',
                                   freq='10T',
                                   targetcol='VALUE',
                                   datecol='DATE',
                                   sep=';',
                                   decimal=',',
                                   date_format="%d/%m/%Y %H:%M:%S.%f",
                                   plot=False)
    assert test_b_1['r_error'] < 100


def test_bal_2():
    """Test for balanced prediction time, using lightgbm, series 1"""
    test_b_2 = balanced_prediction(filename='./tests/test_series/Serie2.csv',
                                   freq='5T',
                                   targetcol='VALUE',
                                   datecol='DATE',
                                   sep=';',
                                   decimal=',',
                                   date_format="%d/%m/%Y %H:%M:%S.%f",
                                   plot=False,
                                   opt=False,
                                   opt_runs=10)

    assert test_b_2['r_error'] < 100


def test_bal_3():
    """Test for balanced prediction time, using lightgbm, series 3"""
    test_b_3 = balanced_prediction(filename='./tests/test_series/Serie3.csv',
                                   freq='15T',
                                   targetcol='INSTALACIONES [kWh]',
                                   datecol='MSJO_DATUM',
                                   sep=';',
                                   decimal='.',
                                   date_format="%d/%m/%Y %H:%M",
                                   plot=False)

    assert test_b_3['r_error'] < 100


def test_bal_4():
    """Test for balanced prediction time, using lightgbm, series 4"""
    test_b_3 = balanced_prediction(filename='./tests/test_series/Serie4.csv',
                                   freq='5T',
                                   targetcol='VALUE',
                                   datecol='DATE',
                                   sep=';',
                                   decimal=',',
                                   date_format="%d/%m/%Y %H:%M:%S.%f",
                                   plot=False)

    assert test_b_3['r_error'] < 100


def test_slow_1():
    """Test for slow prediction time, using keras, series 1"""
    test_s_1 = slow_prediction(filename='./tests/test_series/Serie1.csv',
                               freq='10T',
                               targetcol='VALUE',
                               datecol='DATE',
                               sep=';',
                               decimal=',',
                               date_format="%d/%m/%Y %H:%M:%S.%f",
                               plot=False)

    assert test_s_1['r_error'] < 100


def test_slow_2():
    """Test for slow prediction time, using keras, series 2"""
    test_s_2 = slow_prediction(filename='./tests/test_series/Serie2.csv',
                               freq='1T',
                               targetcol='VALUE',
                               datecol='DATE',
                               sep=';',
                               decimal=',',
                               date_format="%d/%m/%Y %H:%M:%S.%f",
                               plot=False)

    assert test_s_2['r_error'] < 100


def test_slow_3():
    """Test for slow prediction time, using keras, series 3"""
    test_s_3 = slow_prediction(filename='./tests/test_series/Serie3.csv',
                               freq='15T',
                               targetcol='INSTALACIONES [kWh]',
                               datecol='MSJO_DATUM',
                               sep=';',
                               decimal='.',
                               date_format="%d/%m/%Y %H:%M",
                               plot=False)

    assert test_s_3['r_error'] < 100


def test_slow_4():
    """Test for fast prediction time, using keras, series 4"""
    test_s_4 = slow_prediction(filename='./tests/test_series/Serie4.csv',
                               freq='5T',
                               targetcol='VALUE',
                               datecol='DATE',
                               sep=';',
                               decimal=',',
                               date_format="%d/%m/%Y %H:%M:%S.%f",
                               plot=False)

    assert test_s_4['r_error'] < 100

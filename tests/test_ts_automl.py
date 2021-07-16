from ts_automl import __version__
from ts_automl.pipelines import Pipeline
from ts_automl.data_input import read_data
from ts_automl.preprocessing import create_sample_feat
from ts_automl import api
from fastapi.testclient import TestClient


def test_version():
    """
    Test for validating the package version installed.

    Will pass if version is same as the one expected by the test.
    """

    assert __version__ == '0.2.0'

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
    model = Pipeline(filename='./tests/test_series/Serie1.csv',
                     type='fast',
                     freq='10T'
                     )

    assert model.df is not None


def test_fit_model():
    """Tests that the model can support the fit method"""

    model.fit()

    assert model.r_error is not None


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

    model_1 = Pipeline(filename='./tests/test_series/Serie1.csv',
                       type='fast',
                       freq='10T'
                       )
    model_1.fit()

    assert model_1.r_error < 100


def test_fast_2():
    """Test for fast prediction time, using knn, series 2"""

    model_2 = Pipeline(filename='./tests/test_series/Serie2.csv',
                       type='fast',
                       freq='1T')
    model_2.fit()

    assert model_2.r_error < 100


def test_fast_3():
    """Test for fast prediction time, using knn, series 3"""

    model_3 = Pipeline(filename='./tests/test_series/Serie3.csv',
                       type='fast',
                       freq='15T',
                       targetcol='INSTALACIONES [kWh]',
                       datecol='MSJO_DATUM',
                       sep=';',
                       decimal='.',
                       date_format="%d/%m/%Y %H:%M",
                       plot=True)
    model_3.fit()
    assert model_3.r_error < 100


def test_fast_4():
    """Test for fast prediction time, using knn, series 4"""

    model_4 = Pipeline(filename='./tests/test_series/Serie4.csv',
                       type='fast',
                       freq='5T'
                       )
    model_4.fit()

    assert model_4.r_error < 100


def test_bal_1():
    """Test for balanced prediction time, using lightgbm, series 1"""
    model_1 = Pipeline(filename='./tests/test_series/Serie1.csv',
                       type='balanced',
                       freq='10T'
                       )
    model_1.fit()

    assert model_1.r_error < 100


def test_bal_2():
    """Test for balanced prediction time, using lightgbm, series 1"""
    model_2 = Pipeline(filename='./tests/test_series/Serie2.csv',
                       type='balanced',
                       freq='1T')
    model_2.fit()

    assert model_2.r_error < 100


def test_bal_3():
    """Test for balanced prediction time, using lightgbm, series 3"""
    model_3 = Pipeline(filename='./tests/test_series/Serie3.csv',
                       type='balanced',
                       freq='15T',
                       targetcol='INSTALACIONES [kWh]',
                       datecol='MSJO_DATUM',
                       sep=';',
                       decimal='.',
                       date_format="%d/%m/%Y %H:%M",
                       plot=True)
    model_3.fit()
    assert model_3.r_error < 100


def test_bal_4():
    """Test for balanced prediction time, using lightgbm, series 4"""
    model_4 = Pipeline(filename='./tests/test_series/Serie4.csv',
                       type='balanced',
                       freq='5T'
                       )
    model_4.fit()

    assert model_4.r_error < 100


def test_slow_1():
    """Test for slow prediction time, using keras, series 1"""
    model_1 = Pipeline(filename='./tests/test_series/Serie1.csv',
                       type='slow',
                       freq='10T'
                       )
    model_1.fit()

    assert model_1.r_error < 100


def test_slow_2():
    """Test for slow prediction time, using keras, series 2"""
    model_2 = Pipeline(filename='./tests/test_series/Serie2.csv',
                       type='slow',
                       freq='1T')
    model_2.fit()

    assert model_2.r_error < 100


def test_slow_3():
    """Test for slow prediction time, using keras, series 3"""
    model_3 = Pipeline(filename='./tests/test_series/Serie3.csv',
                       type='slow',
                       freq='15T',
                       targetcol='INSTALACIONES [kWh]',
                       datecol='MSJO_DATUM',
                       sep=';',
                       decimal='.',
                       date_format="%d/%m/%Y %H:%M",
                       plot=True)
    model_3.fit()
    assert model_3.r_error < 100


def test_slow_4():
    """Test for fast prediction time, using keras, series 4"""
    model_4 = Pipeline(filename='./tests/test_series/Serie4.csv',
                       type='slow',
                       freq='5T'
                       )
    model_4.fit()

    assert model_4.r_error < 100

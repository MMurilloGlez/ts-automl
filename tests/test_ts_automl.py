from ts_automl import __version__
from ts_automl.pipelines import fast_prediction
from ts_automl.pipelines import balanced_prediction
from ts_automl.pipelines import slow_prediction


def test_version():
    """
    Test for validating the package version installed.

    Will pass if version is same as the one expected by the test.
    """

    assert __version__ == '0.1.1'


def test_fast_1():
    """
    Test for fast prediction time, using knn, series 1

    Uses a reference time series from test_series path, launches a prediction
    model on it using the respective model, and compares the result with the
    naive prediction for the same series. If the prediction mse is lower in
    comparison with the naive model, the test will pass
    """

    p, e, r = fast_prediction(filename='./tests/test_series/Serie1.csv',
                              freq='10T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

    assert r < 100


def test_fast_2():
    """Test for fast prediction time, using knn, series 2"""
    p, e, r = fast_prediction(filename='./tests/test_series/Serie2.csv',
                              freq='1T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

    assert r < 100


def test_fast_3():
    """Test for fast prediction time, using knn, series 3"""
    p, e, r = fast_prediction(filename='./tests/test_series/Serie3.csv',
                              freq='15T',
                              targetcol='INSTALACIONES [kWh]',
                              datecol='MSJO_DATUM',
                              sep=';',
                              decimal='.',
                              date_format="%d/%m/%Y %H:%M",
                              plot=False)

    assert r < 100


def test_fast_4():
    """Test for fast prediction time, using knn, series 4"""
    p, e, r = fast_prediction(filename='./tests/test_series/Serie4.csv',
                              freq='5T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

    assert r < 100


def test_fast_5():
    """Test for fast prediction time, using knn, series 5"""
    p, e, r = fast_prediction(filename='./tests/test_series/Serie5.csv',
                              freq='20T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)


def test_bal_1():
    """Test for balanced prediction time, using lightgbm, series 1"""
    p, e, r = balanced_prediction(filename='./tests/test_series/Serie1.csv',
                                  freq='10T',
                                  targetcol='VALUE',
                                  datecol='DATE',
                                  sep=';',
                                  decimal=',',
                                  date_format="%d/%m/%Y %H:%M:%S.%f",
                                  plot=False)
    assert r < 100


def test_bal_2():
    """Test for balanced prediction time, using lightgbm, series 1"""
    p, e, r = balanced_prediction(filename='./tests/test_series/Serie2.csv',
                                  freq='5T',
                                  targetcol='VALUE',
                                  datecol='DATE',
                                  sep=';',
                                  decimal=',',
                                  date_format="%d/%m/%Y %H:%M:%S.%f",
                                  plot=False)

    assert r < 100


def test_bal_3():
    """Test for balanced prediction time, using lightgbm, series 3"""
    p, e, r = balanced_prediction(filename='./tests/test_series/Serie3.csv',
                                  freq='15T',
                                  targetcol='INSTALACIONES [kWh]',
                                  datecol='MSJO_DATUM',
                                  sep=';',
                                  decimal='.',
                                  date_format="%d/%m/%Y %H:%M",
                                  plot=False)

    assert r < 100


def test_bal_4():
    """Test for balanced prediction time, using lightgbm, series 4"""
    p, e, r = balanced_prediction(filename='./tests/test_series/Serie4.csv',
                                  freq='5T',
                                  targetcol='VALUE',
                                  datecol='DATE',
                                  sep=';',
                                  decimal=',',
                                  date_format="%d/%m/%Y %H:%M:%S.%f",
                                  plot=False)

    assert r < 100


def test_bal_5():
    """Test for balanced prediction time, using lightgbm, series 5"""
    p, e, r = balanced_prediction(filename='./tests/test_series/Serie5.csv',
                                  freq='20T',
                                  targetcol='VALUE',
                                  datecol='DATE',
                                  sep=';',
                                  decimal=',',
                                  date_format="%d/%m/%Y %H:%M:%S.%f",
                                  plot=False)

    assert r < 100


def test_slow_1():
    """Test for slow prediction time, using keras, series 1"""
    p, e, r = slow_prediction(filename='./tests/test_series/Serie1.csv',
                              freq='10T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

    assert r < 100


def test_slow_2():
    """Test for slow prediction time, using keras, series 2"""
    p, e, r = slow_prediction(filename='./tests/test_series/Serie2.csv',
                              freq='1T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

    assert r < 100


def test_slow_3():
    """Test for slow prediction time, using keras, series 3"""
    p, e, r = slow_prediction(filename='./tests/test_series/serie3.csv',
                              freq='15T',
                              targetcol='INSTALACIONES [kWh]',
                              datecol='MSJO_DATUM',
                              sep=';',
                              decimal='.',
                              date_format="%d/%m/%Y %H:%M",
                              plot=False)

    assert r < 100


def test_slow_4():
    """Test for fast prediction time, using keras, series 4"""
    p, e, r = slow_prediction(filename='./tests/test_series/Serie4.csv',
                              freq='5T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

    assert r < 100


def test_slow_5():
    """Test for slow prediction time, using keras, series 5"""
    p, e, r = slow_prediction(filename='./tests/test_series/Serie5.csv',
                              freq='20T',
                              targetcol='VALUE',
                              datecol='DATE',
                              sep=';',
                              decimal=',',
                              date_format="%d/%m/%Y %H:%M:%S.%f",
                              plot=False)

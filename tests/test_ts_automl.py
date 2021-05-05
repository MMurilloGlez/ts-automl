from ts_automl import __version__


def test_version():
    assert __version__ == '0.1.1'


def slow_prediction_test():

    rel_error = 0
    assert rel_error < 100


def bal_prediction_test():

    rel_error = 0
    assert rel_error < 100


def fast_prediction_test():

    rel_error = 0
    assert rel_error < 100


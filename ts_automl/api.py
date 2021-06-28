from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.encoders import jsonable_encoder
from enum import Enum
from ts_automl.data_input import read_data
from ts_automl.pipelines import fast_prediction, slow_prediction
from ts_automl.pipelines import balanced_prediction, pipeline
from json import dump, load
import os
from argparse import Namespace


class Pred_time(str, Enum):
    fast = 'fast'
    balanced = 'balanced'
    slow = 'slow'


class Errors(str, Enum):
    mse = 'mse'
    rmse = 'rmse'
    mape = 'mape'
    rsquare = 'rsquare'
    exp_var = 'exp_var'


class Feat(str, Enum):
    mean = 'mean'
    std = 'std'
    max = 'max'
    min = 'min'
    quantile = 'quantile'
    iqr = 'iqr'
    minute = 'minute'
    hour = 'hour'
    dayofweek = 'dayofweek'
    day = 'day'
    month = 'month'
    quarter = 'quarter'
    weekofyear = 'weekofyear'
    weekend = 'weekend'


path = ''
model = None

app = FastAPI(title="TS-AutoML",
              description="Automated machine learning for time series" +
                          " prediction",
              version="0.1.3",
              debug=True)


@app.get("/")
def read_root():
    return {"TS-AutoML API"}


@app.post("/UploadTraining/")
async def upload_csv(file: UploadFile = File(...),
                     freq: Optional[str] = Query('15T'),
                     targetcol: Optional[str] = Query('VALUE'),
                     datecol: Optional[str] = Query('DATE'),
                     dateformat: Optional[str] = Query('%d/%m/%Y %H:%M:%S.%f'),
                     sep: Optional[str] = Query(';'),
                     decimal: Optional[str] = Query(','),
                     points: Optional[int] = Query(50),
                     window_length: Optional[int] = Query(100),
                     rolling_window: Optional[List[int]] = Query([10]),
                     horizon: Optional[int] = Query(1),
                     step: Optional[int] = Query(1),
                     num_datapoints: Optional[int] = Query(2000),
                     features: Optional[List[Feat]] = Query([Feat.mean,
                                                             Feat.std]),
                     selected_feat: Optional[int] = Query(20),
                     plot: Optional[bool] = Query(True),
                     error: Optional[List[Errors]] = Query([Errors.mse,
                                                            Errors.mape]),
                     rel_error: Optional[bool] = Query(True),
                     opt: Optional[bool] = Query(False),
                     opt_runs: Optional[int] = Query(10),
                     type: Optional[Pred_time] = Query([Pred_time.fast])
                     ):
    try:
        os.mkdir("./train_data")
        print(os.getcwd())
    except Exception:
        print(Exception)

    filename = os.getcwd()+"/train_data/" + file.filename.replace(" ", "-")

    with open(filename, 'wb+') as f:
        f.write(file.file.read())
        f.close()

    response = {"filename": filename,
                "type": type,
                "datecol": datecol,
                "targetcol": targetcol,
                "dateformat": dateformat,
                "sep": sep,
                "decimal": decimal,
                "freq": freq,
                "points": points,
                "plot": plot,
                "error": error,
                "rel_error": rel_error,
                "features": features,
                "selected_feat": selected_feat,
                "window_length": window_length,
                "rolling_window": rolling_window,
                "horizon": horizon,
                "opt": opt,
                "opt_runs": opt_runs,
                "num_datapoints": num_datapoints
                }
    server_response = jsonable_encoder(response)

    global model
    model = pipeline(filename='./tests/test_series/Serie1.csv',
                     type=type, targetcol=targetcol, datecol=datecol,
                     plot=plot, points=points, error=error, rel_error=rel_error,
                     features=features, selected_feat=selected_feat,
                     window_length=window_length, rolling_window=rolling_window,
                     horizon=horizon, opt=opt, opt_runs=opt_runs,
                     num_datapoints=num_datapoints, sep=sep, decimal=decimal,
                     dateformat=dateformat, freq=freq
                     )


    response['data'] = model.df.to_json()

    metadata_file = filename[:-4]+"_metadata.txt"
    global path
    path = metadata_file

    with open(metadata_file, 'w') as f:
        dump(response, f)
        f.close()

    return(server_response)


@app.get('/Fit/')
async def model_fitting():

    if path == '':
        response = 'No training data uploaded.'
    else:
        if model.opt:
            model_fit = model.fit_opt()
            response = model_fit
        else:
            model_fit = model.fit()
            response = model_fit

    return{response}


@app.get('/Predict/')
async def model_predict(horizon: int = Query(50)):
    if model.abs_error is None:
        response = 'Use /Fit/ method first'
    elif model.abs_error is not None:
        prediction = model.predict(num_points=horizon)
        response = prediction
    return(response)

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.encoders import jsonable_encoder
from enum import Enum
from ts_automl.data_input import read_data
from ts_automl.pipelines import fast_prediction, slow_prediction
from ts_automl.pipelines import balanced_prediction
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

app = FastAPI(debug=True)


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
                     fit_type: Optional[Pred_time] = Query([Pred_time.fast])
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
                "datecol": datecol,
                "targetcol": targetcol,
                "dateformat": dateformat,
                "sep": sep,
                "decimal": decimal,
                "freq": freq,
                "points": points,
                "fit_type": fit_type,
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

    df = read_data(filename,
                   targetcol,
                   datecol,
                   sep,
                   decimal,
                   dateformat,
                   freq
                   )
    response['data'] = df.to_json()

    metadata_file = filename[:-4]+"_metadata.txt"
    global path
    path = metadata_file

    with open(metadata_file, 'w') as f:
        dump(response, f)
        f.close()

    return(server_response)


@app.get('/Fit/')
async def model_fit():
    global path, model
    if path == '':
        response = 'No training data uploaded.'
    else:
        f = load(open(path))
        f = Namespace(**f)
        if f.fit_type == 'fast':
            fast_prediction(f.filename, f.freq, f.targetcol, f.datecol,
                            f.sep, f.decimal, f.dateformat)
        elif f.fit_type == 'balanced':
            balanced_prediction(f.filename, f.freq, f.targetcol, f.datecol,
                                f.sep, f.decimal, f.dateformat)
        else:
            slow_prediction(f.filename, f.freq, f.targetcol, f.datecol,
                            f.sep, f.decimal, f.dateformat)
        response = 'fit successful'

    return{response}


@app.get('/Predict/')
async def model_predict(horizon: int = Query(50)):

    return{'This feature is not yet working'}

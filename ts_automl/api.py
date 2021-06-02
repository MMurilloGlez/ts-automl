from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Query
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
                     freq: str = Form('15T'),
                     targetcol: str = Form('VALUE'),
                     datecol: str = Form('DATE'),
                     date_format: str = Form('%d/%m/%Y %H:%M:%S.%f'),
                     sep: str = Form(';'),
                     decimal: str = Form(','),
                     points: int = Form(50),
                     window_length: int = Form(100),
                     rolling_window: List[int] = Form([10]),
                     horizon: int = Form(1),
                     step: int = Form(1),
                     num_datapoints: int = Form(2000),
                     features: List[Feat] = Query([Feat.mean, Feat.std]),
                     selected_feat: int = Form(20),
                     plot: Optional[bool] = Form(True),
                     error: List[Errors] = Query([Errors.mse, Errors.mape]),
                     rel_error: Optional[bool] = Form(True),
                     opt: Optional[bool] = Form(False),
                     opt_runs: Optional[int] = Form(10),
                     fit_type: Pred_time = Form([Pred_time.fast])
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
                "date_format": date_format,
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
                   date_format,
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
                            f.sep, f.decimal, f.date_format)
        elif f.fit_type == 'balanced':
            balanced_prediction(f.filename, f.freq, f.targetcol, f.datecol,
                                f.sep, f.decimal, f.date_format)
        else:
            slow_prediction(f.filename, f.freq, f.targetcol, f.datecol,
                            f.sep, f.decimal, f.date_format)
        response = 'fit successful'

    return{response}


@app.get('/Predict/')
async def model_predict(horizon: int = Query(50)):
    return{'jujuju'}

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.encoders import jsonable_encoder
from enum import Enum
import os


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


app = FastAPI()


@app.get("/")
def read_root():
    return {"TS-AutoML API"}


@app.get('/ModelFit/')
def model_fit(predictor: str = 'majakaa'):
    return{'jejeje'}


@app.get('/Predict/')
def model_predict():
    return{'jujuju'}


@app.post("/UploadTraining/")
async def upload_csv(file: UploadFile = File(...),
                     freq: str = Form('15T'),
                     targetcol: str = Form('DATE'),
                     datecol: str = Form('VALUE'),
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
                     fit_type: List[Pred_time] = Form([Pred_time.fast])
                     ):
    try:
        os.mkdir("./train_data")
        print(os.getcwd())
    except Exception:
        print(Exception)
    filename = os.getcwd()+"/train_data/"+file.filename.replace(" ", "-")
    with open(filename, 'wb+') as f:
        f.write(file.file.read())
        f.close()
    response = jsonable_encoder({"Train_data_path": filename,
                                 "Date column name": datecol,
                                 "Target column name": targetcol,
                                 "Date Format": date_format,
                                 "Field Separator": sep,
                                 "Decimal place": decimal,
                                 "Frequency": freq,
                                 "points": points,
                                 "Model": fit_type,
                                 "plot": plot,
                                 "Error Metrics": error,
                                 "Relative error": rel_error,
                                 "Feature list": features,
                                 "Features to keep": selected_feat,
                                 "Window length": window_length,
                                 "Rolling window": rolling_window,
                                 "Horizon": horizon,
                                 "Optimization": opt,
                                 "Number of optimization runs": opt_runs,
                                 "Number of datapoints": num_datapoints
                                 })

    return(response)

from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
import os

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
async def upload_csv(freq: str = Form(...),
                     targetcol: str = Form(...),
                     datecol: str = Form(...),
                     date_format: str = Form(...),
                     points: int = Form(50),
                     window_length: int = Form(100),
                     rolling_window: List[int] = Form([10]),
                     horizon: int = Form(1),
                     step: int = Form(1),
                     num_datapoints: int = Form(2000),
                     features: List[str] = Form(['mean', 'std', 'max', 'min']),
                     selected_feat: int = Form(20),
                     plot: Optional[bool] = Form(True),
                     error: List[str] = Form(['mse', 'mape']),
                     opt: Optional[bool] = Form(False),
                     opt_runs: Optional[int] = Form(10),
                     file: UploadFile = File(...)
                     ):
    print(file.file)
    try:
        os.mkdir("./train_data")
        print(os.getcwd())
    except Exception:
        print(Exception)
    filename = os.getcwd()+"/train_data/"+file.filename.replace(" ", "-")
    with open(filename, 'wb+') as f:
        f.write(file.file.read())
        f.close()
    file = jsonable_encoder({"Train_data_path": filename})
    return(file)

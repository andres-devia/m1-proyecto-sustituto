import json 
from fastapi import FastAPI, UploadFile
from utils import preprocess_train
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model = None

@app.get('/')
def hello_world():
  return {
    'message': 'Hello world'
  }

class Pacientdata:
  age: int
  gender: str
  hormonal_changes: str
  family_history: str
  race_ethnicity: str
  body_weight: str
  calcium_intake: str
  vitamin_d_intake: str
  physical_activity: str
  smoking: str
  alcohol_consumption: str
  medical_conditions: str
  medications: str
  prior_fractures: str

@app.post('/train')
def train(file: UploadFile = ...):

  df = pd.read_csv(file.file)
  pipe, accuracy = preprocess_train(df)
  model = pipe
  
  return {
    'message': 'El modelo se entreno con exito',
    'accuracy': accuracy
  }

@app.post('/predict')
def predict(data: Pacientdata = ...):
  data.dict = data.__dict__

  df = pd.DataFrame(data.dict)

  return {
    'message': 'Hello world'
  }


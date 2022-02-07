from typing import Optional

from fastapi import FastAPI
from fastapi import Body, Query ,Path
from pydantic import BaseModel
import numpy as np
import joblib

class Item(BaseModel):
    Sabor: int
    Ambiente: int
    Servicio: int
    Variedad_menu: int
    Vale_precio_pagado: int

svmodel = open("SVModel.pkl","rb")
model = joblib.load(svmodel) 

app = FastAPI()

@app.get('/')
async def index():
  return {"text":"Hello from FastAPI"}

@app.post('/predict')
def predict_function(item: Item):
  # sabor ambiente servicio variedad vale_precio
   prediction = model.predict(np.array([item.Sabor,item.Ambiente,item.Servicio,item.Variedad_menu,item.Vale_precio_pagado]).reshape(1, -1))
   print(prediction[0])
   return {"data": str(prediction[0])}
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

SVCmodel = open("./Models/SVC.pkl","rb")
Treemodel = open("./Models/Arboles.pkl","rb")
KNNmodel = open("./Models/KNN.pkl","rb")

modelSVC = joblib.load(SVCmodel) 
modelTree = joblib.load(Treemodel) 
modelKNN = joblib.load(KNNmodel) 

app = FastAPI()

@app.get('/')
async def index():
  return {"text":"Hello from FastAPI"}

@app.post('/predict')
def predict_function(item: Item):
  # sabor ambiente servicio variedad vale_precio
   predictionSVC = modelSVC.predict(np.array([item.Sabor,item.Ambiente,item.Servicio,item.Variedad_menu,item.Vale_precio_pagado]).reshape(1, -1))
   predictionTree = modelTree.predict(np.array([item.Sabor,item.Ambiente,item.Servicio,item.Variedad_menu,item.Vale_precio_pagado]).reshape(1, -1))
   predictionKNN = modelKNN.predict(np.array([item.Sabor,item.Ambiente,item.Servicio,item.Variedad_menu,item.Vale_precio_pagado]).reshape(1, -1))
   print(predictionSVC[0])
   return {"data": {
                    "svc":str(predictionSVC[0]),
                    "tree": str(predictionTree[0]),
                    "knn": str(predictionKNN[0]),
                    }
          }
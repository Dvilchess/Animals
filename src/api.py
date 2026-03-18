import io
import os
import numpy as np
from pathlib import Path

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "models/modelo_animales.keras")
IMG_SIZE = (100, 100)

CLASES = [
    "cane", "cavallo", "elefante", "farfalla", "gallina",
    "gatto", "mucca", "pecora", "ragno", "scoiattolo"
]

TRADUCCION = {
    "cane": "Perro", "cavallo": "Caballo", "elefante": "Elefante",
    "farfalla": "Mariposa", "gallina": "Gallina", "gatto": "Gato",
    "mucca": "Vaca", "pecora": "Oveja", "ragno": "Araña", "scoiattolo": "Ardilla"
}

app = FastAPI(title="🐾 Clasificador de Animales", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

model = None

@app.on_event("startup")
async def cargar_modelo():
    global model
    if Path(MODEL_PATH).exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Modelo cargado")
    else:
        print(f"⚠️  Modelo no encontrado en {MODEL_PATH}")

@app.get("/")
def root():
    return {"mensaje": "🐾 Clasificador de Animales API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": model is not None}

@app.get("/clases")
def get_clases():
    return {"clases": [{"italiano": c, "espanol": TRADUCCION[c]} for c in CLASES]}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Solo JPG o PNG")

    contenido = await file.read()
    img = Image.open(io.BytesIO(contenido)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predicciones = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(predicciones))
    confianza = float(predicciones[idx])

    top3_idx = np.argsort(predicciones)[-3:][::-1]
    top3 = [{"animal": CLASES[i], "espanol": TRADUCCION[CLASES[i]], "confianza": f"{float(predicciones[i])*100:.1f}%"} for i in top3_idx]

    return {"animal": CLASES[idx], "espanol": TRADUCCION[CLASES[idx]], "confianza": f"{confianza*100:.1f}%", "top3": top3}
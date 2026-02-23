# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 21:38:06 2026

@author: subha
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model once at startup
model = tf.keras.models.load_model("face_mask_model.h5")

app = FastAPI(title="Face Mask Detection API")

# Image size (change if your model uses different size)
IMG_SIZE = 224

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    
    # If image has 4 channels (RGBA), convert to RGB
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "Face Mask Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]

        if prediction > 0.5:
            label = "Mask"
            confidence = float(prediction)
        else:
            label = "No Mask"
            confidence = float(1 - prediction)

        return JSONResponse({
            "prediction": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
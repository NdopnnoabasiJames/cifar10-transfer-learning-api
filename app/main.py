from fastapi import FastAPI, UploadFile, File
from model_loader import model, CLASS_NAMES
from utils import preprocess_image
from PIL import Image
import numpy as np

app = FastAPI(title="CIFAR10 Transfer Learning API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image = Image.open(file.file).convert("RGB")
    processed = preprocess_image(image)

    predictions = model.predict(processed)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return {
        "prediction": CLASS_NAMES[class_index],
        "confidence": round(confidence, 4)
    }
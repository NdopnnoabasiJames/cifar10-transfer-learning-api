from fastapi import FastAPI, UploadFile, File
from model_loader import model, CLASS_NAMES
from utils import preprocess_image
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CIFAR10 Transfer Learning API")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"Prediction request received: {file.filename}")

        image = Image.open(file.file).convert("RGB")
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return {
            "prediction": CLASS_NAMES[class_index],
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": "An error occurred during prediction."}
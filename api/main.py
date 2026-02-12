"""
FastAPI application for CIFAR-10 classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.main import CIFAR10Predictor
from app.utils import CIFAR10_CLASSES


app = FastAPI(
    title="CIFAR-10 Transfer Learning API",
    description="API for CIFAR-10 image classification using transfer learning",
    version="0.1.0"
)

# Initialize predictor (update model path as needed)
# predictor = CIFAR10Predictor(model_path="models/best_model.pth")


@app.get("/")
def read_root():
    """
    Root endpoint
    """
    return {
        "message": "CIFAR-10 Transfer Learning API",
        "version": "0.1.0",
        "classes": CIFAR10_CLASSES
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict CIFAR-10 class for uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with class and confidence scores
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        # prediction = predictor.predict(image)
        
        # Placeholder response
        return JSONResponse(content={
            "filename": file.filename,
            "message": "Prediction endpoint - model integration pending"
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/classes")
def get_classes():
    """
    Get list of CIFAR-10 classes
    """
    return {"classes": CIFAR10_CLASSES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

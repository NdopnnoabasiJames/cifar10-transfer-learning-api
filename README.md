# CIFAR-10 Transfer Learning API

A RESTful API for CIFAR-10 image classification using transfer learning with PyTorch.

## Project Structure

```
cifar10-transfer-learning-api/
│
├── app/
│   ├── __init__.py
│   ├── model.py          # Model architecture
│   ├── train.py          # Training script
│   ├── inference.py      # Load + predict logic
│   └── utils.py          # Preprocessing helpers
│
├── api/
│   └── main.py           # FastAPI application
│
├── models/
│   └── (saved models go here)
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## CIFAR-10 Classes

The model classifies images into 10 categories:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cifar10-transfer-learning-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training

Train the model on CIFAR-10 dataset:

```bash
python app/train.py
```

The trained model will be saved in the `models/` directory.

## Running the API

Start the FastAPI server:

```bash
python api/main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check
- `GET /classes` - Get list of CIFAR-10 classes
- `POST /predict` - Upload an image for classification

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Get classes
curl http://localhost:8000/classes

# Predict image class
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black app/ api/ tests/
```

Lint code:
```bash
flake8 app/ api/ tests/
```

## License

MIT License

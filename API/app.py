import os
import time
import pickle
import pandas as pd
import mlflow
import dagshub
import numpy as np
import yaml
import onnx
import onnxruntime as rt
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_handling import data_preprocessing
from src.logger import logger
from sklearn.feature_extraction.text import TfidfTransformer


app = FastAPI()

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create a custom registry
registry = CollectorRegistry()

# Define custom metrics
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

class ModelLoader:
    def __init__(self):
        """Initialize model and vectorizer"""
        try:
            # Load vectorizer
            self.vectorizer = self.load_vectorizer()
            
            # Load ONNX model
            self.model = self.load_model()
            self.inspect_model()
            
            logger.info("✅ Model and vectorizer loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize model loader: {e}")
            logger.warning("Continuing with dummy model for testing purposes")
            self.vectorizer = None
            self.model = None
        
    def inspect_model(self):
        """Inspect ONNX model details"""
        try:
            # Get input details
            input_details = self.model.get_inputs()
            output_details = self.model.get_outputs()

            logger.info("ONNX Model Input Details:")
            for input_detail in input_details:
                logger.info(f"Input Name: {input_detail.name}")
                logger.info(f"Input Type: {input_detail.type}")
                logger.info(f"Input Shape: {input_detail.shape}")

            logger.info("\nONNX Model Output Details:")
            for output_detail in output_details:
                logger.info(f"Output Name: {output_detail.name}")
                logger.info(f"Output Type: {output_detail.type}")
                logger.info(f"Output Shape: {output_detail.shape}")

        except Exception as e:
            logger.error(f"Error inspecting ONNX model: {e}")


    def load_vectorizer(self, vectorizer_path='models/vectorizer.pkl'):
        """Load and restore the vectorizer"""
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)

            logger.info(f" Vectorizer Loaded. Vocabulary size: {len(vectorizer.vocabulary_)}")
            return vectorizer
        except Exception as e:
            logger.error(f" Error loading vectorizer: {e}")
            return None

    

    def load_model(self, model_path='models/model.onnx'):
        """Load  model"""
        try:
            onnx_model = onnx.load(model_path)
            
            session = rt.InferenceSession(model_path)
            
            logger.info(f"🔍 ONNX Model loaded from {model_path}")
            return session
        except Exception as e:
            logger.error(f"❌ Error loading ONNX model: {e}")
            raise

    

    def predict(self, text):
        """Predict the class of the input text with comprehensive debugging"""
        try:
            
            #dummy try because earlier model loading was failing sometimes
            if self.model is None:
                logger.info("Using dummy model for prediction")
                return 1 
                
            # Cleaning the text
            normalized_text = self.normalize_text(text)

            # Transform text using vectorizer
            features = self.vectorizer.transform([normalized_text])

            features_array = features.toarray()
            limited_features = features_array[:, :].astype(np.float32)
            
            # Prediction
            input_name = self.model.get_inputs()[0].name

            output_names = [
                output.name for output in self.model.get_outputs()
            ]

            results = self.model.run(output_names, {input_name: limited_features})

            label = results[0][0] 
            probabilities = results[1]  

            logger.info(f"Predicted label: {label}")
            logger.info(f"Prediction probabilities: {probabilities}")

            return int(label)

        except Exception as e:
            logger.error(f"Detailed Prediction Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return a default value in case of error
            return 0

    def normalize_text(self, text):
        """Enhanced text normalization with logging"""
        try:
            try:
                params = yaml.safe_load(open('params.yaml'))
                
                preprocessor = data_preprocessing.PreProcess(train_path='', test_path='')
                
                normalized_text = preprocessor.preprocess_text(text, params['data_preprocessing'])
                
               
                return normalized_text
            except Exception as inner_e:
                logger.warning(f"Error in preprocessing: {inner_e}, using simple normalization")
                return text.lower().strip() #just fall back step 
        except Exception as e:
            logger.error(f"Error in normalize_text: {e}")
            return text

model_loader = ModelLoader()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home route that renders the index page"""
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    
    try:
        response = templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "result": None,
                "title": "Text Classification App",
                "message": "Enter text to classify"
            }
        )
        REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
        return response
    except Exception as e:
        logger.error(f"Template  error: {e}")
        return HTMLResponse(content="""
            <h1>Text Classification App</h1>
            <p style="color: red">Error: Template not found or invalid</p>
            <p>Please ensure 'index.html' exists in the templates directory</p>
        """, status_code=500)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    """Predict route for text classification"""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        prediction = model_loader.predict(text)
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": prediction,
                "text": text,  
                "title": "Prediction Result"
            }
        )
    except Exception as e:
        logger.error(f" Prediction error: {e}")
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Sorry, we encountered an error. Please try again later.",
                "text": text,  
                "message": "Error",
                "title": "Error"
            }
        )

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics"""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    import uvicorn
    logger.info(" Starting application")
    uvicorn.run(app, host="0.0.0.0", port=5000)

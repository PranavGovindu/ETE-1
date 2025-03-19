import dagshub.auth
import onnx
from skl2onnx import convert_sklearn
import onnxruntime as rt
from src.logger import logger
import yaml
import pandas as pd
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class ModelEval():
    def __init__(self):
        dagshub_token = os.getenv("DAGSHUB_AUTH_TOKEN")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_AUTH_TOKEN environment variable is not set")
        
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        mlflow_tracking_url = os.getenv("MLFLOW_TRACKING_URI")
        mlflow.set_tracking_uri(mlflow_tracking_url)
        dagshub.auth.add_app_token(dagshub_token)
        
    def load_params(self, params_path):
        """Load model hyperparameters from YAML file."""
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            logger.info(f"Parameters loaded from {params_path}")
            return params
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            raise e
        
    def load_model(self, model_path):
        """Load ONNX model."""
        try:
            model = onnx.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
        
    def load_data(self, path):
        """Load CSV data."""
        try:
            data = pd.read_csv(path)
            logger.info(f"Data loaded from {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e
    
    def evaluate_model(self, model, X_test, Y_test):
        """Evaluate the model on test data for sentiment analysis with labels -1, 0, 1.
        Args:
            model: ONNX model.
            X_test: Test data.
            Y_test: True labels (-1, 0, 1).
        """
        try:
            sess = rt.InferenceSession(model.SerializeToString())
            input_name = sess.get_inputs()[0].name
            
            X_test_float32 = X_test.astype(np.float32)
            
            # Run inference
            pred_onnx = sess.run(None, {input_name: X_test_float32})[0]
            
            # Calculate metrics
            accuracy = accuracy_score(Y_test, pred_onnx)
            
            # For sentiment analysis with 3 classes, we use macro averaging
            precision = precision_score(Y_test, pred_onnx, average='macro', zero_division=0)
            recall = recall_score(Y_test, pred_onnx, average='macro', zero_division=0)
            f1 = f1_score(Y_test, pred_onnx, average='macro', zero_division=0)
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
            
            logger.info(f"Model evaluation completed. Metrics: {metrics}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise e
    
    def save_metrics(self, metrics, metrics_path):
        """Save model evaluation metrics."""
        try:
            with open(metrics_path, 'w') as file:
                json.dump(metrics, file, indent=4)
            logger.info(f"Metrics saved at {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise e
    
    def save_model_info(self, file_path, model_path, run_id):
        """Save ONNX model."""
        try:
            model_info = {'run_id': run_id, 'model_path': model_path}
            with open(file_path, 'w') as f:
                json.dump(model_info, f, indent=4) 
            
            logger.info(f"Model info saved at {file_path}")
        except Exception as e:
            logger.error(f"Error saving model info: {e}")
            raise e
        
def main():
    try:
        ev = ModelEval()
        params = ev.load_params("params.yaml")

        experiment_name = params['base']['experiment_name']
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        # Start a run
        with mlflow.start_run() as run:
            model_path = params.get('paths', {}).get('model_path', 'models/logistic_regression.onnx')
            test_data_path = params.get('paths', {}).get('test_data_path', 'data/processed/test_tfidf.csv')
            metrics_path = params.get('paths', {}).get('metrics_path', 'reports/metrics.json')
            model_info_path = params.get('paths', {}).get('model_info_path', 'reports/model_info.json')
            
            model = ev.load_model(model_path)
            data = ev.load_data(test_data_path)

            X_test = data.iloc[:, :-1].values
            Y_test = data.iloc[:, -1].values

            metrics = ev.evaluate_model(model, X_test, Y_test)
            ev.save_metrics(metrics, metrics_path)

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model with signature
            input_example = X_test[:1].astype(np.float32)
            mlflow.onnx.log_model(model, "model", input_example=input_example)

            ev.save_model_info(model_info_path, model_path, run.info.run_id)

            mlflow.log_artifact(metrics_path)

    except Exception as e:
        logger.error(f"Error occurred while evaluating model: {e}")
        raise e

if __name__ == "__main__":
    main()
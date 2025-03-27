import os
import yaml
import pandas as pd
import numpy as np
import onnx
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from src.logger import logger

class ModelBuild:
    
    def load_data(self, path):
        """Load CSV data."""
        try:
            data = pd.read_csv(path)
            logger.info(f"Data loaded from {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e
    
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
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, C: float, solver: str, penalty: str):
        """Train the Logistic Regression model ."""
        try:
        

            clf = LogisticRegression(
                C=C,
                solver=solver,
                penalty=penalty,
            )
            clf.fit(X_train, y_train)
            logger.info('Model training completed')
            return clf
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e

    def save_model(self, model, model_path):
        """ Save Logistic Regression model in ONNX format."""
        try:
            if not hasattr(model, "coef_"):
                raise ValueError("Invalid model: Model must be trained before saving.")

            initial_type = [("input", FloatTensorType([None, model.coef_.shape[1]]))]

            onnx_model = convert_sklearn(model, initial_types=initial_type)

            # Save ONNX model
            onnx.save(onnx_model, model_path)

            logger.info(f"Model saved successfully in ONNX format at {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise e

def main():
    """Main function to execute the ML pipeline."""
    try:
        model_builder = ModelBuild()

        data_path = "data/processed/train_tfidf.csv"
        data = model_builder.load_data(data_path)
        X_train = data.iloc[:, :-1].values
        y_train = data.iloc[:, -1].values
        
        print(X_train.shape)
        print(y_train.shape)
        print("Number of features:", X_train.shape[1])

        params_path = "params.yaml"
        params=model_builder.load_params(params_path)

        trained_model = model_builder.train_model(
            X_train,
            y_train,params["model_building"]["hyperparameters"]["logistic_regression"]["C"],
            params["model_building"]["hyperparameters"]["logistic_regression"]["solver"],
            params["model_building"]["hyperparameters"]["logistic_regression"]["penalty"])

        # Save model in ONNX format
        model_path = "models/model.onnx"
        model_builder.save_model(trained_model, model_path)

        logger.info("Model Building done successfully")

    except Exception as e:
        logger.error(f"Error occured while building model: {e}")

if __name__ == "__main__":
    main()

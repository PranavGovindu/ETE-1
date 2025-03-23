from src.logger import logger
import mlflow
import os
from dotenv import load_dotenv
import json
import yaml
from mlflow.tracking import MlflowClient

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_AUTH_TOKEN")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow_tracking_url = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(mlflow_tracking_url)

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
   
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict, model_alias: str, status_tag: str):
    """Register the model to the MLflow Model Registry using aliases instead of stages."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
       
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        
        client = mlflow.tracking.MlflowClient()
        
        client.set_registered_model_alias(
            name=model_name,
            alias="staging",
            version=model_version.version
        )
        
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key="status",
            value="staging"
        )
       
        logger.debug(f'Model {model_name} version {model_version.version} registered and aliased to "staging"')
        
        return model_version  # Return the same object as before for compatibility
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        params = yaml.safe_load(open("params.yaml"))["model_registry"]
        model_name = params["model_name"]
        model_info_path = 'reports/figures/model_info.json'
        model_info = load_model_info(model_info_path)
        
        # Extract parameters, with backward compatibility
        model_alias = params["model_alias"]
        status_tag = params["model_stage"]
        
        # Register a new model version
        version = register_model(model_name, model_info, model_alias, status_tag)
        print(f"Successfully registered model version {version}")
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
import os
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.logger import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.connections import s3_connection
from dotenv import load_dotenv

load_dotenv()

class Data_Ingestion():
    def __init__(self, params_path='params.yaml'):
        self.params = self.load_params(params_path)
        self.sample_size = self.params["data_ingestion"]["sample_size"]
        self.random_state = self.params["base"]["random_state"]
        self.test_size = self.params["data_ingestion"]["test_size"]
       
    def load_params(self, params_path:str):
        """Loads params file
        Args:
            params_path (str): contains params
        Returns:
            yaml of params
        """
        try:
            with open(params_path) as f:
                params = yaml.safe_load(f)
                logger.debug("Params loaded successfully from %s", params_path)
                return params
        except Exception as e:
            logger.error("Error loading params: %s", e)    
            raise
       
    def load_data(self):
        """
        Loading data from source
        Returns:
            dataframe
        """
        try:
            df = pd.read_csv(self.data_path).dropna()
            if self.sample_size < len(df):
                df = df.sample(n=self.sample_size, random_state=self.random_state)
            logger.debug(f"Data loaded successfully  of shape ={df.shape}")
            return df
           
        except Exception as e:
            logger.error("Error in loading files: %s", e)
            raise
    
    def split_data(self, df):
        """
        Split data into train and test sets
        Args:
            df (pd.DataFrame): Input dataframe
        Returns:
            tuple: (train_df, test_df)
        """
        try:
            train_df, test_df = train_test_split(
                df, 
                test_size=self.test_size, 
                random_state=self.random_state
            )
            logger.debug("Data split successfully: %d training samples, %d test samples", 
                         len(train_df), len(test_df))
            return train_df, test_df
        except Exception as e:
            logger.error("Error splitting data: %s", e)
            raise
       
    def save_data(self, train_df:pd.DataFrame, test_df:pd.DataFrame, data_path:str):
        """After splitting data is stored in the given datapaths
        Args:
            train_df (pd.DataFrame): train df
            test_df (pd.DataFrame): test df
            data_path (str): location
        """
        try:
            raw_data_path = os.path.join(data_path, 'raw')
            os.makedirs(raw_data_path, exist_ok=True)
            train_df.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
            test_df.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
            logger.debug('Train and test data saved to %s', raw_data_path)
        except Exception as e:
            logger.error('Unexpected error occurred while saving the data: %s', e)
            raise
           
def main():
    """Main function to run data ingestion process"""
    try:
        s3= s3_connection.s3_operations(bucket_name='ete-1', aws_access_key=os.getenv('AWS_ACCESS_ID'), aws_secret_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
        data_ingestion = Data_Ingestion()
        df = s3.fetch_file_from_s3("Twitter_Data.csv")
        train_df, test_df = data_ingestion.split_data(df)
        data_ingestion.save_data(train_df, test_df, 'data')
        logger.info("Data ingestion completed successfully")
    except Exception as e:
        logger.error("Data ingestion failed: %s", e)

if __name__ == "__main__":
    main()
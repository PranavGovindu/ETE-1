import pandas as pd
import yaml
import re
import nltk
from nltk.corpus import stopwords
import os
from src.logger import logger
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.logger import logger

class PreProcess():
    
    def __init__(self,train_path:str,test_path:str):
        self.train_path=train_path
        self.test_path=test_path
        
    
    
    def preprocess_data(self):
        """Preprocess train and test data"""
        try:
            with open('params.yaml') as params_file:
                    params = yaml.safe_load(params_file)

            preprocess_params = params['data_preprocessing']

            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            train_df.columns=train_df.columns.str.strip()
            test_df.columns=test_df.columns.str.strip()
            train_df = train_df.dropna(subset=['clean_text']).astype(str)
            test_df = test_df.dropna(subset=['clean_text']).astype(str)


            train_df['clean_text'] = train_df['clean_text'].apply(
                lambda x: self.preprocess_text(x, preprocess_params)
            )
            test_df['clean_text'] = test_df['clean_text'].apply(
                lambda x: self.preprocess_text(x, preprocess_params)
            )
            logger.info("data preprocessed")
            return train_df,test_df

        
        except Exception as e:
            logger.error("error occured pre processing text in pre processing file :%s",e)
            raise e

    def preprocess_text(self,text, params):
        """Clean and preprocess text data"""
        try:  
            if params['lowercase']:
                text = text.lower()

            if params['remove_punctuation']:
                text = re.sub(r'[^\w\s]', '', text)

            if params['remove_stopwords']:
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words('english'))
                tokens = text.split()
                text = ' '.join([word for word in tokens if word not in stop_words])
            

            return text
        
        except Exception as e:
            logger.error("error while pre processing text :%s",e)
            raise e

def main():
    
    
    with open('params.yaml') as params_file:
        params = yaml.safe_load(params_file)
    
    
    pre=PreProcess(params['data_preprocessing']['train_path'],params['data_preprocessing']['test_path'])
    
    train_preprocessed,test_preprocessed=pre.preprocess_data()
    
    train_preprocessed.to_csv('data/interim/train_preprocessed.csv', index=False)
    test_preprocessed.to_csv('data/interim/test_preprocessed.csv', index=False)
    logger.info("Preprocessed data saved succesfully")

if __name__ == "__main__":
    main()
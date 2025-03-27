import pandas as pd
from src.logger import logger
import yaml
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import numpy as np

class FeatureEng:
    def __init__(self):
        self.vectorizer = None
        self.feature_selector = None

    def load_params(self, params_path):
        """Load parameters from YAML file."""
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            logger.info(f"Parameters loaded from {params_path}")
            return params
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            raise e

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            df.fillna('', inplace=True)
            logger.info('Data loaded and NaNs filled from %s', file_path)
            return df
        except Exception as e:
            logger.error('Error loading data: %s', e)
            raise

    def apply_tfidf(self, train_data, test_data, max_features, ngram_range):
        """Apply TF-IDF Vectorizer with feature selection"""
        try:
            logger.info("Applying TF-IDF...")
            
            self.vectorizer = TfidfVectorizer(
                max_features=5000,  
                ngram_range=ngram_range
            )
            
            X_train = train_data['clean_text'].values
            y_train = train_data['category'].values
            X_test = test_data['clean_text'].values
            y_test = test_data['category'].values
           
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            
            self.feature_selector = SelectKBest(score_func=f_classif, k=max_features)
            X_train_selected = self.feature_selector.fit_transform(
                X_train_tfidf.toarray(), 
                y_train
            )
            
            X_test_tfidf = self.vectorizer.transform(X_test)
            X_test_selected = X_test_tfidf[:, self.feature_selector.get_support(indices=True)].toarray()
            
            train_df = pd.DataFrame(X_train_selected)
            train_df['label'] = y_train
            
            test_df = pd.DataFrame(X_test_selected)
            test_df['label'] = y_test
           
            os.makedirs('models', exist_ok=True)
            pickle.dump(self.vectorizer, open('models/vectorizer.pkl', 'wb'))
            
            logger.info(f'Vectorizer saved to models/vectorizer.pkl')
            
            logger.info(f'TF-IDF applied. Selected {max_features} top features')
           
            return train_df, test_df
        except Exception as e:
            logger.error(f"Error during TF-IDF transformation: {e}")
            raise e
           
    def save_data(self, df, file_path):
        """Save the dataframe to a CSV file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise e

def main():
    try:
        feature_eng = FeatureEng()
       
        params = feature_eng.load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])
       
        train_data = feature_eng.load_data('./data/interim/train_preprocessed.csv')
        test_data = feature_eng.load_data('./data/interim/test_preprocessed.csv')
       
        # Apply TF-IDF and feature selection
        train_df, test_df = feature_eng.apply_tfidf(train_data, test_data, max_features, ngram_range)
       
        # Save processed data
        feature_eng.save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        feature_eng.save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
       
        logger.info("Feature engineering completed successfully")
       
    except Exception as e:
        logger.error(f"Failed to complete feature engineering: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
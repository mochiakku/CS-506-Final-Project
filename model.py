import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, Tuple

class HousingPricePredictionPipeline:
    def __init__(self):
        """Initialize the pipeline with necessary components."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.linear_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Warning: Could not initialize NLTK components. Error: {e}")
            self.sia = None

    def parse_walkscore(self, walkscore_str: str) -> Tuple[float, float, float]:
        """Parse walkscore string into separate numeric scores."""
        try:
            if pd.isna(walkscore_str):
                return 0, 0, 0
                
            scores = {'walk': 0, 'transit': 0, 'bike': 0}
            parts = str(walkscore_str).split(';')
            
            for part in parts:
                part = part.strip()
                if 'Walk:' in part:
                    scores['walk'] = float(re.search(r'(\d+)/100', part).group(1))
                elif 'Transit:' in part:
                    scores['transit'] = float(re.search(r'(\d+)/100', part).group(1))
                elif 'Bike:' in part:
                    scores['bike'] = float(re.search(r'(\d+)/100', part).group(1))
                    
            return scores['walk'], scores['transit'], scores['bike']
        except Exception as e:
            print(f"Error parsing walkscore: {e}")
            return 0, 0, 0

    def extract_text_features(self, text: str) -> tuple:
        """Extract sentiment features from text."""
        if pd.isna(text):
            return 0, 0
        
        text = str(text)
        sentiment = self.sia.polarity_scores(text)['compound'] if self.sia else 0
        subjectivity = TextBlob(text).sentiment.subjectivity
        return sentiment, subjectivity


def main():
    return

if __name__ == "__main__":
    main()
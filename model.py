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
    
    def optimize_random_forest(self):
        """Optimize Random Forest hyperparameters using GridSearchCV."""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        self.rf_model = grid_search.best_estimator_
        
        return grid_search.best_params_, grid_search.best_score_

    def load_and_preprocess_data(self, filepath: str) -> None:
        """Load and preprocess the housing data."""
        try:
            print(f"\nLoading data from {filepath}...")
            self.data = pd.read_csv(filepath)
            
            # Remove parking space listings and outliers
            self.data = self.data[self.data['sq_ft'].notna()]
            self.data = self.data[self.data['price'] < self.data['price'].quantile(0.99)]
            self.data = self.data[self.data['sq_ft'] < self.data['sq_ft'].quantile(0.99)]
            
            print(f"After removing outliers: {len(self.data)} properties")
            
            # Parse walkscores
            walk_scores = self.data['walkscore'].apply(self.parse_walkscore)
            self.data['walk_score'] = [score[0] for score in walk_scores]
            self.data['transit_score'] = [score[1] for score in walk_scores]
            self.data['bike_score'] = [score[2] for score in walk_scores]
            
            # Extract text features
            sentiment_subj = [self.extract_text_features(desc) for desc in self.data['description']]
            self.data['description_sentiment'] = [s[0] for s in sentiment_subj]
            self.data['description_subjectivity'] = [s[1] for s in sentiment_subj]
            
            # Select features for modeling
            features = ['beds', 'baths', 'sq_ft', 'walk_score', 'transit_score', 'bike_score',
                       'description_sentiment', 'description_subjectivity']
            
            # Prepare data for modeling
            X = self.data[features]
            y = self.data['price']
            
            # Handle missing values
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            print("\nData preprocessing completed successfully")
            print(f"Final dataset shape: {self.data.shape}")
            self.print_data_summary()
            
        except Exception as e:
            print(f"Error in data preprocessing: {e}")
            raise

    def print_data_summary(self):
        """Print summary statistics of the dataset."""
        print("\nDataset Summary:")
        print(f"Total properties: {len(self.data)}")
        print(f"Average price: ${self.data['price'].mean():,.2f}")
        print(f"Median price: ${self.data['price'].median():,.2f}")
        print(f"Average square footage: {self.data['sq_ft'].mean():,.2f}")
        print(f"Average bedrooms: {self.data['beds'].mean():.1f}")
        print(f"Average bathrooms: {self.data['baths'].mean():.1f}")
        print(f"Average walk score: {self.data['walk_score'].mean():.1f}")
        print(f"Average transit score: {self.data['transit_score'].mean():.1f}")
        print(f"Average bike score: {self.data['bike_score'].mean():.1f}")


def main():
    return

if __name__ == "__main__":
    main()
"""
Author: Mohammed Islam
Date: 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


class AmericanAirlinesPricePredictor:
    def __init__(self):
        self.model = None
        self.features = [
            'daysInAdvance',  # Days between search and flight
            #'seatsRemaining',  # Remaining seats
            'numberOfConnections',  # Direct or connecting flight
            'travelDistanceMinutes',  # Flight duration
        ]
        self.is_trained = False

    def preprocess(self, df):
        """Simplify data for modeling."""
        df = df.copy()
        df['searchDate'] = pd.to_datetime(df['searchDate'])
        df['flightDate'] = pd.to_datetime(df['flightDate'])
        df['daysInAdvance'] = (df['flightDate'] - df['searchDate']).dt.days
        # Weekend flag (True if Saturday or Sunday)
        df['weekendFlight'] = df['flightDate'].dt.dayofweek >= 5
        return df

    def create_target(self, df):
        """Target: 1 if price > day's average, else 0."""
        daily_avg = df.groupby('flightDate')['baseFare'].mean()
        df['daily_avg'] = df['flightDate'].map(daily_avg)
        df['above_avg'] = (df['baseFare'] > df['daily_avg']).astype(int)
        return df

    def train(self, df):
        """Train a model to predict above/below average price."""
        df = self.preprocess(df)
        df = self.create_target(df)
        X = df[self.features]
        y = df['above_avg']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        preds = self.model.predict(X_test)
        auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        acc = accuracy_score(y_test, preds)

        print(f"Model trained: Accuracy {acc:.2%}, AUC {auc:.2%}")

    def predict(self, days_in_advance, num_connections, duration):
        """Predict for a single flight scenario."""
        if not self.is_trained:
            raise Exception("Train the model first!")

        # Arrange into DataFrame for sklearn
        row = pd.DataFrame([[days_in_advance, num_connections, duration]],
                           columns=self.features)
        prob = self.model.predict_proba(row)[0]
        prediction = 'Above Average' if prob[1] > 0.5 else 'Below Average'
        confidence = f"{prob[1]:.1%} the price will go up"
        print(f"Prediction: {prediction} ({prob[1]:.1%} the price will go up)")
        return confidence, prediction


# Example usage (simple demo)
if __name__ == "__main__":
    df = pd.read_csv('JFK_ORD_truncated_16.csv')
    predictor = AmericanAirlinesPricePredictor()
    predictor.train(df)
    # Test: 5 days in advance, 4 seats left, 1 connection, 180 min flight
    predictor.predict(days_in_advance=100, num_connections=2, duration=300)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

class StockPredictor:
    def __init__(self, preprocessed_data: pd.DataFrame):
        """
        Initialize stock predictor with preprocessed data.
        
        Args:
            preprocessed_data (pd.DataFrame): Preprocessed data for prediction
        """
        self.data = preprocessed_data
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self, time_steps: int = 5) -> tuple:
        """
        Prepare features and target for time series prediction.
        
        Args:
            time_steps (int, optional): Number of previous time steps to use. Defaults to 5.
        
        Returns:
            tuple: Feature and target arrays
        """
        # Select relevant features
        feature_columns = [
            col for col in self.data.columns 
            if col.startswith('sentiment_feature_') or 
               col.startswith('normalized_')
        ]
        
        # Sort data by timestamp if available
        if 'created_utc' in self.data.columns:
            self.data = self.data.sort_values('created_utc')
        
        # Create sliding window features
        X, y = [], []
        for i in range(len(self.data) - time_steps):
            # Extract time series window
            window = self.data.iloc[i:i+time_steps][feature_columns]
            X.append(window.values.flatten())
            
            # Predict next time step's target
            next_row = self.data.iloc[i+time_steps]
            y.append(next_row['normalized_score'])
        
        return np.array(X), np.array(y)

    def train_model(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train Random Forest Regressor with cross-validation.
        
        Args:
            test_size (float, optional): Proportion of test data. Defaults to 0.2.
            random_state (int, optional): Random seed. Defaults to 42.
        
        Returns:
            dict: Model performance metrics and model itself
        """
        X, y = self.prepare_features()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Initialize and train model
        self.model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5, 
            random_state=random_state
        )
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Performance metrics
        metrics = {
            'model': self.model,
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cross_val_scores': cross_val_score(self.model, X_scaled, y, cv=5)
        }
        
        return metrics

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model.
        
        Args:
            input_data (np.ndarray): Input features for prediction
        
        Returns:
            np.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Scale input data
        input_scaled = self.scaler.transform(input_data)
        
        return self.model.predict(input_scaled)
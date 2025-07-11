import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MLPredictor:
    """
    Machine Learning predictor for MLB player statistics
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def predict_player_stats(self, player_data: pd.DataFrame, player_type: str) -> dict:
        """Predict player statistics using ML models"""
        try:
            if player_data.empty:
                return {}
            
            # Prepare features
            features = self._prepare_features(player_data, player_type)
            
            if features.empty:
                return {}
            
            predictions = {}
            
            if player_type.lower() == 'batter':
                # Predict batting stats
                stats_to_predict = ['batting_avg', 'home_runs', 'rbi', 'ops']
                
                for stat in stats_to_predict:
                    if stat in player_data.columns:
                        pred_value = self._predict_stat(features, player_data[stat], stat)
                        predictions[stat] = pred_value
            else:
                # Predict pitching stats
                stats_to_predict = ['era', 'whip', 'strikeouts', 'wins']
                
                for stat in stats_to_predict:
                    if stat in player_data.columns:
                        pred_value = self._predict_stat(features, player_data[stat], stat)
                        predictions[stat] = pred_value
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {}
    
    def _prepare_features(self, player_data: pd.DataFrame, player_type: str) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            # Create time-based features
            if 'date' in player_data.columns:
                player_data['date'] = pd.to_datetime(player_data['date'])
                player_data['days_since_start'] = (player_data['date'] - player_data['date'].min()).dt.days
                player_data['day_of_week'] = player_data['date'].dt.dayofweek
                player_data['month'] = player_data['date'].dt.month
            
            # Create rolling averages as features
            numeric_cols = player_data.select_dtypes(include=[np.number]).columns
            
            feature_cols = []
            for col in numeric_cols:
                if col not in ['date', 'days_since_start', 'day_of_week', 'month']:
                    # Create rolling features
                    player_data[f'{col}_rolling_5'] = player_data[col].rolling(5, min_periods=1).mean()
                    player_data[f'{col}_rolling_10'] = player_data[col].rolling(10, min_periods=1).mean()
                    player_data[f'{col}_trend'] = player_data[col].diff()
                    
                    feature_cols.extend([f'{col}_rolling_5', f'{col}_rolling_10', f'{col}_trend'])
            
            # Add time features
            if 'days_since_start' in player_data.columns:
                feature_cols.extend(['days_since_start', 'day_of_week', 'month'])
            
            # Select features that exist
            available_features = [col for col in feature_cols if col in player_data.columns]
            
            if not available_features:
                return pd.DataFrame()
            
            features = player_data[available_features].fillna(0)
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _predict_stat(self, features: pd.DataFrame, target: pd.Series, stat_name: str) -> float:
        """Predict a specific statistic"""
        try:
            if len(features) < 10:  # Not enough data for ML
                return self._simple_prediction(target)
            
            # Prepare data
            X = features.fillna(0)
            y = target.fillna(target.mean())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Try multiple models
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'lr': LinearRegression()
            }
            
            best_model = None
            best_score = float('-inf')
            
            for name, model in models.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    score = r2_score(y_test, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        self.scalers[stat_name] = scaler
                        
                except Exception:
                    continue
            
            if best_model is None:
                return self._simple_prediction(target)
            
            # Make prediction on latest data
            latest_features = X.iloc[-1:].fillna(0)
            latest_scaled = self.scalers[stat_name].transform(latest_features)
            prediction = best_model.predict(latest_scaled)[0]
            
            # Store model performance
            self.model_performance[stat_name] = {
                'r2_score': best_score,
                'model_type': type(best_model).__name__
            }
            
            # Apply reasonable bounds based on stat type
            prediction = self._apply_stat_bounds(prediction, stat_name)
            
            return prediction
            
        except Exception as e:
            print(f"Error predicting {stat_name}: {e}")
            return self._simple_prediction(target)
    
    def _simple_prediction(self, target: pd.Series) -> float:
        """Simple prediction using recent trend"""
        try:
            if len(target) < 3:
                return target.mean()
            
            # Use weighted average of recent games
            recent_games = min(10, len(target))
            recent_values = target.tail(recent_games)
            
            # Give more weight to recent games
            weights = np.exp(np.linspace(0, 1, len(recent_values)))
            weights = weights / weights.sum()
            
            prediction = np.average(recent_values, weights=weights)
            
            return prediction
            
        except Exception:
            return target.mean()
    
    def _apply_stat_bounds(self, prediction: float, stat_name: str) -> float:
        """Apply realistic bounds to predictions"""
        bounds = {
            'batting_avg': (0.100, 0.450),
            'home_runs': (0, 70),
            'rbi': (0, 150),
            'ops': (0.400, 1.400),
            'era': (1.00, 8.00),
            'whip': (0.80, 2.00),
            'strikeouts': (0, 300),
            'wins': (0, 25)
        }
        
        if stat_name in bounds:
            min_val, max_val = bounds[stat_name]
            prediction = max(min_val, min(max_val, prediction))
        
        return prediction
    
    def get_feature_importance(self, stat_name: str) -> dict:
        """Get feature importance for a specific statistic"""
        if stat_name in self.feature_importance:
            return self.feature_importance[stat_name]
        return {}
    
    def get_model_performance(self, stat_name: str) -> dict:
        """Get model performance metrics"""
        if stat_name in self.model_performance:
            return self.model_performance[stat_name]
        return {}
    
    def predict_team_performance(self, team_data: pd.DataFrame) -> dict:
        """Predict team-level performance metrics"""
        try:
            predictions = {}
            
            if team_data.empty:
                return predictions
            
            # Group by player and predict individual stats
            player_predictions = {}
            
            for player_name, player_data in team_data.groupby('player_name'):
                player_type = player_data['player_type'].iloc[0] if 'player_type' in player_data.columns else 'batter'
                
                player_pred = self.predict_player_stats(player_data, player_type)
                player_predictions[player_name] = player_pred
            
            # Aggregate to team level
            if player_predictions:
                team_batting_avg = np.mean([pred.get('batting_avg', 0.250) 
                                          for pred in player_predictions.values()])
                team_era = np.mean([pred.get('era', 4.00) 
                                  for pred in player_predictions.values()])
                
                predictions = {
                    'team_batting_avg': team_batting_avg,
                    'team_era': team_era,
                    'projected_wins': max(0, min(162, 81 + (team_batting_avg - 0.250) * 100 - (team_era - 4.00) * 10))
                }
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting team performance: {e}")
            return {}
    
    def calculate_prediction_confidence(self, stat_name: str, prediction: float, historical_data: pd.Series) -> float:
        """Calculate confidence in prediction"""
        try:
            if stat_name not in self.model_performance:
                return 0.6  # Default confidence
            
            # Base confidence on model performance
            r2_score = self.model_performance[stat_name].get('r2_score', 0.5)
            base_confidence = max(0.5, min(0.95, r2_score))
            
            # Adjust based on data variability
            if len(historical_data) > 5:
                cv = historical_data.std() / historical_data.mean()
                variability_adjustment = max(0.8, 1.0 - cv)
                base_confidence *= variability_adjustment
            
            # Adjust based on prediction reasonableness
            if len(historical_data) > 0:
                recent_avg = historical_data.tail(10).mean()
                if abs(prediction - recent_avg) / recent_avg > 0.3:  # 30% difference
                    base_confidence *= 0.8
            
            return max(0.5, min(0.95, base_confidence))
            
        except Exception:
            return 0.6

#!/usr/bin/env python3
"""
Advanced Prediction Framework
Leverages your comprehensive Statcast dataset for sophisticated modeling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from historical_data_manager import HistoricalDataManager
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictionFramework:
    """
    Sophisticated prediction engine leveraging comprehensive Statcast data
    """
    
    def __init__(self):
        self.historical_manager = HistoricalDataManager()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def build_comprehensive_features(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Build sophisticated feature set from Statcast data
        """
        features = pd.DataFrame()
        
        # Basic performance metrics
        features['recent_batting_avg'] = player_data['hits'] / player_data['at_bats'].replace(0, 1)
        features['recent_home_run_rate'] = player_data['home_runs'] / player_data['at_bats'].replace(0, 1)
        
        # Advanced Statcast features (when available)
        if 'launch_angle_avg' in player_data.columns:
            features['avg_launch_angle'] = player_data['launch_angle_avg'].fillna(0)
            features['launch_angle_consistency'] = player_data.groupby('player_name')['launch_angle_avg'].transform('std').fillna(0)
        
        if 'hard_hit_rate' in player_data.columns:
            features['hard_hit_rate'] = player_data['hard_hit_rate'].fillna(0)
            features['hard_hit_trend'] = player_data.groupby('player_name')['hard_hit_rate'].transform(lambda x: x.diff().fillna(0))
        
        if 'barrel_rate' in player_data.columns:
            features['barrel_rate'] = player_data['barrel_rate'].fillna(0)
        
        # Rolling performance windows
        for window in [5, 10, 15]:
            features[f'rolling_{window}_avg'] = player_data.groupby('player_name')['hits'].transform(
                lambda x: x.rolling(window, min_periods=1).sum() / 
                player_data.groupby('player_name')['at_bats'].transform(
                    lambda y: y.rolling(window, min_periods=1).sum().replace(0, 1)
                )
            )
        
        # Momentum indicators
        features['recent_form'] = player_data.groupby('player_name')['hits'].transform(
            lambda x: x.rolling(7, min_periods=1).sum()
        )
        
        # Consistency metrics
        features['performance_variance'] = player_data.groupby('player_name')['hits'].transform('std').fillna(0)
        
        # Days since last game (rest factor)
        features['rest_days'] = player_data.groupby('player_name')['game_date'].transform(
            lambda x: x.diff().dt.days.fillna(1)
        )
        
        # Season context
        features['season'] = player_data['season']
        features['games_played'] = player_data.groupby(['player_name', 'season']).cumcount() + 1
        
        return features
    
    def train_hitting_prediction_model(self):
        """
        Train sophisticated hitting prediction models using your Statcast data
        """
        print("🤖 Training Advanced Hitting Prediction Model")
        print("=" * 50)
        
        session = self.historical_manager.get_session()
        
        # Load comprehensive dataset
        query = """
        SELECT player_name, game_date, season, at_bats, hits, home_runs,
               launch_angle_avg, hard_hit_rate, barrel_rate
        FROM historical_player_performance 
        WHERE at_bats > 0
        ORDER BY player_name, game_date
        """
        
        df = pd.read_sql(query, session.bind)
        session.close()
        
        print(f"📊 Loaded {len(df):,} game records for model training")
        
        if len(df) < 100:
            print("⚠️ Insufficient data for advanced modeling")
            return
        
        # Build features
        features = self.build_comprehensive_features(df)
        
        # Target variables
        targets = {
            'batting_avg': df['hits'] / df['at_bats'].replace(0, 1),
            'home_run_prob': df['home_runs'] / df['at_bats'].replace(0, 1),
            'hits_next_game': df.groupby('player_name')['hits'].shift(-1).fillna(0)
        }
        
        # Train models for each target
        for target_name, target_values in targets.items():
            print(f"\n🎯 Training {target_name} model...")
            
            # Remove rows with missing targets
            valid_idx = ~target_values.isna()
            X = features[valid_idx].fillna(0)
            y = target_values[valid_idx]
            
            if len(X) < 50:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Try multiple models
            models_to_try = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Ridge': Ridge(alpha=1.0)
            }
            
            best_model = None
            best_score = -np.inf
            
            for model_name, model in models_to_try.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                avg_score = cv_scores.mean()
                
                print(f"   • {model_name}: R² = {avg_score:.3f} (±{cv_scores.std():.3f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
            
            # Train best model
            best_model.fit(X_train_scaled, y_train)
            
            # Test performance
            y_pred = best_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"   ✅ Best model test R²: {test_r2:.3f}, RMSE: {test_rmse:.3f}")
            
            # Store model
            self.models[target_name] = best_model
            self.scalers[target_name] = scaler
            
            # Feature importance (for tree-based models)
            if hasattr(best_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[target_name] = importance
                print(f"   📈 Top features: {', '.join(importance.head(3)['feature'].tolist())}")
        
        print(f"\n✅ Advanced prediction models trained successfully!")
        self._evaluate_model_quality()
    
    def _evaluate_model_quality(self):
        """
        Evaluate the quality and reliability of trained models
        """
        print(f"\n📊 Model Quality Assessment:")
        print("=" * 40)
        
        for target_name in self.models.keys():
            if target_name in self.feature_importance:
                importance_df = self.feature_importance[target_name]
                top_features = importance_df.head(5)
                
                print(f"\n🎯 {target_name.replace('_', ' ').title()} Model:")
                print("   Top predictive features:")
                for _, row in top_features.iterrows():
                    print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        statcast_features = ['hard_hit_rate', 'avg_launch_angle', 'barrel_rate']
        basic_features = ['recent_batting_avg', 'rolling_10_avg', 'recent_form']
        
        print(f"\n🚀 Your Statcast Advantage:")
        print("   Advanced metrics available for enhanced predictions:")
        print("   • Exit velocity and launch angle data")
        print("   • Hard-hit rate and barrel classification")
        print("   • Multi-dimensional performance analysis")
        print("   • Professional-grade feature engineering")
    
    def predict_player_performance(self, player_name: str, recent_games: int = 15) -> dict:
        """
        Generate comprehensive predictions for a player
        """
        if not self.models:
            return {"error": "Models not trained yet"}
        
        session = self.historical_manager.get_session()
        
        # Get recent player data
        query = """
        SELECT * FROM historical_player_performance 
        WHERE player_name = :player_name 
        ORDER BY game_date DESC 
        LIMIT :limit
        """
        
        player_df = pd.read_sql(query, session.bind, params={
            'player_name': player_name, 
            'limit': recent_games
        })
        session.close()
        
        if len(player_df) == 0:
            return {"error": f"No data found for {player_name}"}
        
        # Build features for latest data
        features = self.build_comprehensive_features(player_df)
        latest_features = features.iloc[-1:].fillna(0)
        
        predictions = {}
        
        for target_name, model in self.models.items():
            if target_name in self.scalers:
                scaler = self.scalers[target_name]
                X_scaled = scaler.transform(latest_features)
                pred_value = model.predict(X_scaled)[0]
                
                # Convert to probability/percentage where appropriate
                if 'prob' in target_name or 'rate' in target_name:
                    predictions[target_name] = max(0, min(1, pred_value))
                else:
                    predictions[target_name] = max(0, pred_value)
        
        return {
            'player': player_name,
            'predictions': predictions,
            'data_quality': 'Enhanced with Statcast metrics' if player_df['hard_hit_rate'].sum() > 0 else 'Basic stats only',
            'games_analyzed': len(player_df)
        }

def main():
    """
    Demonstrate the advanced prediction framework
    """
    framework = AdvancedPredictionFramework()
    framework.train_hitting_prediction_model()

if __name__ == "__main__":
    main()
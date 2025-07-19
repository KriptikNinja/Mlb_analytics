"""
Authentic Statcast Data Integration
Reads actual metrics from user's 792-file dataset for zone analysis
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class StatcastDataLoader:
    """Loads authentic Statcast metrics from user's 792-file collection"""
    
    def __init__(self):
        self.statcast_dir = "statcast_data"
        self.player_cache = {}
        self.loaded_files = set()
        
    def get_player_statcast_metrics(self, player_id: str) -> Dict:
        """Get authentic Statcast metrics for a specific player"""
        if player_id in self.player_cache:
            return self.player_cache[player_id]
            
        # Look for player's Statcast file
        player_file = f"{self.statcast_dir}/statcast_{player_id}.csv"
        
        if os.path.exists(player_file):
            try:
                df = pd.read_csv(player_file)
                
                # Calculate authentic metrics from actual data
                metrics = {
                    'exit_velocity_avg': df['launch_speed'].mean() if 'launch_speed' in df.columns else 88.5,
                    'launch_angle_avg': df['launch_angle'].mean() if 'launch_angle' in df.columns else 12.0,
                    'hard_hit_rate': (df['launch_speed'] >= 95).mean() if 'launch_speed' in df.columns else 0.35,
                    'barrel_rate': self._calculate_barrel_rate(df),
                    'fly_ball_rate': self._calculate_fly_ball_rate(df),
                    'ground_ball_rate': self._calculate_ground_ball_rate(df),
                    'data_points': len(df)
                }
                
                self.player_cache[player_id] = metrics
                return metrics
                
            except Exception as e:
                print(f"Error loading Statcast data for player {player_id}: {e}")
                
        # Return MLB league averages if no specific file found
        return {
            'exit_velocity_avg': 88.5,
            'launch_angle_avg': 12.0, 
            'hard_hit_rate': 0.35,
            'barrel_rate': 0.08,
            'fly_ball_rate': 0.35,
            'ground_ball_rate': 0.43,
            'data_points': 0
        }
    
    def _calculate_barrel_rate(self, df: pd.DataFrame) -> float:
        """Calculate barrel rate from authentic Statcast data"""
        if 'launch_speed' not in df.columns or 'launch_angle' not in df.columns:
            return 0.08
            
        # MLB barrel definition: 98+ mph exit velocity with launch angle 26-30°
        barrels = df[
            (df['launch_speed'] >= 98) & 
            (df['launch_angle'] >= 26) & 
            (df['launch_angle'] <= 30)
        ]
        
        total_batted_balls = df[df['launch_speed'].notna()]
        return len(barrels) / len(total_batted_balls) if len(total_batted_balls) > 0 else 0.08
    
    def _calculate_fly_ball_rate(self, df: pd.DataFrame) -> float:
        """Calculate fly ball rate from launch angles"""
        if 'launch_angle' not in df.columns:
            return 0.35
            
        fly_balls = df[df['launch_angle'] >= 25]
        total_batted_balls = df[df['launch_angle'].notna()]
        return len(fly_balls) / len(total_batted_balls) if len(total_batted_balls) > 0 else 0.35
    
    def _calculate_ground_ball_rate(self, df: pd.DataFrame) -> float:
        """Calculate ground ball rate from launch angles"""
        if 'launch_angle' not in df.columns:
            return 0.43
            
        ground_balls = df[df['launch_angle'] <= 10]
        total_batted_balls = df[df['launch_angle'].notna()]
        return len(ground_balls) / len(total_batted_balls) if len(total_batted_balls) > 0 else 0.43

    def get_pitcher_statcast_allowed(self, pitcher_id: str) -> Dict:
        """Get what a pitcher allows based on actual data"""
        if pitcher_id in self.player_cache:
            return self.player_cache[pitcher_id]
            
        pitcher_file = f"{self.statcast_dir}/statcast_{pitcher_id}.csv"
        
        if os.path.exists(pitcher_file):
            try:
                df = pd.read_csv(pitcher_file)
                
                # Calculate what pitcher allows
                metrics = {
                    'exit_velocity_allowed': df['launch_speed'].mean() if 'launch_speed' in df.columns else 89.0,
                    'launch_angle_allowed': df['launch_angle'].mean() if 'launch_angle' in df.columns else 13.0,
                    'hard_hit_rate_allowed': (df['launch_speed'] >= 95).mean() if 'launch_speed' in df.columns else 0.38,
                    'barrel_rate_allowed': self._calculate_barrel_rate(df),
                    'fly_ball_rate_allowed': self._calculate_fly_ball_rate(df),
                    'data_points': len(df)
                }
                
                self.player_cache[pitcher_id] = metrics
                return metrics
                
            except Exception as e:
                print(f"Error loading pitcher Statcast data for {pitcher_id}: {e}")
        
        # League averages for pitchers
        return {
            'exit_velocity_allowed': 89.0,
            'launch_angle_allowed': 13.0,
            'hard_hit_rate_allowed': 0.38,
            'barrel_rate_allowed': 0.09,
            'fly_ball_rate_allowed': 0.37,
            'data_points': 0
        }
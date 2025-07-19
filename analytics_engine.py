import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AnalyticsEngine:
    """
    Core analytics engine for MLB data analysis
    """
    
    def __init__(self):
        self.streak_thresholds = {
            'hot': 0.7,   # 70th percentile
            'cold': 0.3   # 30th percentile
        }
        
    def calculate_rolling_averages(self, player_data: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate rolling averages for player stats"""
        if player_data.empty:
            return pd.DataFrame()
        
        try:
            # Ensure date column is datetime
            if 'date' in player_data.columns:
                player_data['date'] = pd.to_datetime(player_data['date'])
                player_data = player_data.sort_values('date')
            
            # Calculate rolling averages for numeric columns
            numeric_cols = player_data.select_dtypes(include=[np.number]).columns
            rolling_data = player_data.copy()
            
            for col in numeric_cols:
                if col != 'date':
                    rolling_data[f'{col}_rolling'] = player_data[col].rolling(
                        window=window, min_periods=1
                    ).mean()
            
            return rolling_data
            
        except Exception as e:
            print(f"Error calculating rolling averages: {e}")
            return pd.DataFrame()
    
    def detect_hot_cold_streaks(self, team_data: pd.DataFrame, window: int, stat_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Detect hot and cold streaks for players"""
        if team_data.empty:
            return [], []
        
        try:
            hot_players = []
            cold_players = []
            
            # Map stat type to column name
            stat_mapping = {
                'Batting Average': 'batting_avg',
                'Home Runs': 'home_runs',
                'RBIs': 'rbi',
                'ERA': 'era',
                'WHIP': 'whip'
            }
            
            stat_col = stat_mapping.get(stat_type, 'batting_avg')
            
            # Group by player and analyze streaks
            for player_name, player_data in team_data.groupby('player_name'):
                if stat_col not in player_data.columns:
                    continue
                
                # Calculate recent performance
                recent_data = player_data.tail(window)
                if len(recent_data) < 3:  # Need minimum data
                    continue
                
                recent_avg = recent_data[stat_col].mean()
                season_avg = player_data[stat_col].mean()
                
                # Calculate percentile rank
                percentile = stats.percentileofscore(player_data[stat_col], recent_avg) / 100
                
                # Determine streak type
                if percentile >= self.streak_thresholds['hot']:
                    streak_desc = f"Hot streak: {recent_avg:.3f} avg over last {window} games"
                    hot_players.append({
                        'name': player_name,
                        'recent_avg': recent_avg,
                        'season_avg': season_avg,
                        'percentile': percentile,
                        'streak_description': streak_desc,
                        'games': len(recent_data)
                    })
                elif percentile <= self.streak_thresholds['cold']:
                    streak_desc = f"Cold streak: {recent_avg:.3f} avg over last {window} games"
                    cold_players.append({
                        'name': player_name,
                        'recent_avg': recent_avg,
                        'season_avg': season_avg,
                        'percentile': percentile,
                        'streak_description': streak_desc,
                        'games': len(recent_data)
                    })
            
            # Sort by percentile
            hot_players.sort(key=lambda x: x['percentile'], reverse=True)
            cold_players.sort(key=lambda x: x['percentile'])
            
            return hot_players, cold_players
            
        except Exception as e:
            print(f"Error detecting streaks: {e}")
            return [], []
    
    def analyze_matchup(self, batter_name: str, pitcher_name: str, matchup_data: pd.DataFrame) -> Dict:
        """Analyze batter vs pitcher matchup"""
        try:
            analysis = {
                'batter': batter_name,
                'pitcher': pitcher_name,
                'matchup_score': 0.5,  # Default neutral score
                'confidence': 0.6,
                'recommendations': []
            }
            
            if matchup_data.empty:
                analysis['recommendations'].append("No historical data available for this matchup")
                return analysis
            
            # Calculate historical performance
            total_abs = len(matchup_data)
            hits = matchup_data['hit'].sum() if 'hit' in matchup_data.columns else 0
            hrs = matchup_data['home_run'].sum() if 'home_run' in matchup_data.columns else 0
            strikeouts = matchup_data['strikeout'].sum() if 'strikeout' in matchup_data.columns else 0
            
            if total_abs > 0:
                avg = hits / total_abs
                hr_rate = hrs / total_abs
                k_rate = strikeouts / total_abs
                
                # Calculate matchup score based on batter performance
                base_score = avg  # Start with batting average
                
                # Adjust for power
                if hr_rate > 0.05:  # Above average HR rate
                    base_score += 0.1
                
                # Adjust for strikeouts
                if k_rate > 0.25:  # High strikeout rate
                    base_score -= 0.1
                
                # Normalize to 0-1 range
                analysis['matchup_score'] = max(0, min(1, base_score))
                
                # Increase confidence with more data
                analysis['confidence'] = min(0.9, 0.5 + (total_abs / 50))
                
                # Generate recommendations
                if avg > 0.300:
                    analysis['recommendations'].append(f"Batter has strong history: {avg:.3f} average")
                elif avg < 0.200:
                    analysis['recommendations'].append(f"Pitcher has dominated: {avg:.3f} allowed")
                
                if hrs > 0:
                    analysis['recommendations'].append(f"Batter has {hrs} home runs in this matchup")
                
                if k_rate > 0.3:
                    analysis['recommendations'].append(f"High strikeout rate: {k_rate:.1%}")
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing matchup: {e}")
            return {
                'batter': batter_name,
                'pitcher': pitcher_name,
                'matchup_score': 0.5,
                'confidence': 0.5,
                'recommendations': ["Error analyzing matchup data"]
            }
    
    def find_great_matchups(self, games: List[Dict], min_confidence: float) -> List[Dict]:
        """Find great matchups for a given day"""
        try:
            great_matchups = []
            
            for game in games:
                # DISABLED: Fake matchup generation
                # This was generating completely synthetic lineups and stats
                # Need to integrate real MLB roster data instead
                print("⚠️  Advanced matchup analysis disabled - requires authentic roster data")
                continue
            
            # Sort by score and confidence
            great_matchups.sort(key=lambda x: (x['score'], x['confidence']), reverse=True)
            
            return great_matchups[:10]  # Return top 10
            
        except Exception as e:
            print(f"Error finding great matchups: {e}")
            return []
    
    def _generate_matchup_reasoning(self, score: float, confidence: float) -> str:
        """Generate reasoning for matchup recommendation"""
        reasons = []
        
        if score > 0.8:
            reasons.append("Exceptional historical performance")
        elif score > 0.7:
            reasons.append("Strong favorable matchup")
        
        if confidence > 0.85:
            reasons.append("high sample size")
        elif confidence > 0.75:
            reasons.append("good data reliability")
        
        # Add contextual reasons
        context_reasons = [
            "batter's hot streak vs pitcher's recent struggles",
            "favorable strike zone matchup",
            "batter excels against this pitch type",
            "pitcher vulnerable to this batter's approach",
            "recent form strongly favors batter"
        ]
        
        reasons.append(context_reasons[0] if context_reasons else "Standard analysis")
        
        return "Strong matchup due to " + ", ".join(reasons)
    
    def calculate_zone_advantage(self, batter_zones: pd.DataFrame, pitcher_zones: pd.DataFrame) -> Dict:
        """Calculate zone-by-zone advantage for batter vs pitcher"""
        try:
            if batter_zones.empty or pitcher_zones.empty:
                return {}
            
            advantages = {}
            
            for zone in range(1, 10):
                batter_perf = batter_zones[batter_zones['zone'] == zone]['batting_avg'].iloc[0] if not batter_zones[batter_zones['zone'] == zone].empty else 0.250
                pitcher_perf = pitcher_zones[pitcher_zones['zone'] == zone]['opponent_avg'].iloc[0] if not pitcher_zones[pitcher_zones['zone'] == zone].empty else 0.250
                
                # Calculate advantage (positive favors batter)
                advantage = batter_perf - pitcher_perf
                advantages[f'Zone_{zone}'] = {
                    'batter_avg': batter_perf,
                    'pitcher_avg': pitcher_perf,
                    'advantage': advantage,
                    'recommendation': 'Favorable' if advantage > 0.050 else 'Neutral' if advantage > -0.050 else 'Avoid'
                }
            
            return advantages
            
        except Exception as e:
            print(f"Error calculating zone advantage: {e}")
            return {}
    
    def get_performance_trends(self, player_data: pd.DataFrame, stat_columns: List[str]) -> Dict:
        """Analyze performance trends for a player"""
        try:
            trends = {}
            
            for stat in stat_columns:
                if stat not in player_data.columns:
                    continue
                
                values = player_data[stat].dropna()
                if len(values) < 5:
                    continue
                
                # Calculate trend using linear regression
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                trend_direction = 'Improving' if slope > 0 else 'Declining' if slope < 0 else 'Stable'
                trend_strength = abs(r_value)
                
                trends[stat] = {
                    'direction': trend_direction,
                    'strength': trend_strength,
                    'slope': slope,
                    'significance': p_value < 0.05,
                    'recent_avg': values.tail(5).mean(),
                    'season_avg': values.mean()
                }
            
            return trends
            
        except Exception as e:
            print(f"Error analyzing performance trends: {e}")
            return {}

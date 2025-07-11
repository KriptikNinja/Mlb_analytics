"""
Advanced Betting Engine for MLB Analytics
Implements sophisticated betting strategies and edge detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedBettingEngine:
    """
    Advanced betting engine that identifies profitable opportunities using machine learning
    and statistical analysis
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.betting_thresholds = {
            'high_confidence': 0.65,
            'medium_confidence': 0.55,
            'min_edge': 0.02,  # 2% minimum edge (was 5%, too high)
            'max_risk': 0.15   # 15% maximum bankroll risk
        }
        
    def analyze_betting_opportunities(self, games: List[Dict], player_predictions: Dict) -> List[Dict]:
        """Identify profitable betting opportunities using hot streaks, ballpark edges, and confidence analysis"""
        opportunities = []
        
        for game in games:
            try:
                # Hot streak opportunities
                hot_streak_opps = self._find_hot_streak_opportunities(game, player_predictions)
                
                # Ballpark advantage opportunities
                ballpark_opps = self._find_ballpark_advantages(game, player_predictions)
                
                # Matchup-specific edges
                matchup_opps = self._find_matchup_edges(game, player_predictions)
                
                # Weather-based opportunities
                weather_opps = self._find_weather_opportunities(game, player_predictions)
                
                # Combine all opportunities
                all_game_opps = hot_streak_opps + ballpark_opps + matchup_opps + weather_opps
                
                # Score each opportunity for confidence
                for opp in all_game_opps:
                    opp['confidence_score'] = self._calculate_confidence_score(opp)
                    opp['betting_edge'] = self._calculate_betting_edge(opp)
                    opp['recommended_units'] = self._calculate_recommended_units(opp)
                
                # Only include opportunities with meaningful edges
                quality_opps = [opp for opp in all_game_opps if opp['betting_edge'] >= 3.0]
                opportunities.extend(quality_opps)
                        
            except Exception as e:
                logger.error(f"Error analyzing betting opportunities for game: {e}")
                continue
        
        # Sort by confidence score and edge combined
        opportunities.sort(key=lambda x: (x['confidence_score'] * x['betting_edge']), reverse=True)
        return opportunities[:25]  # Top 25 opportunities
    
    def _find_hot_streak_opportunities(self, game: Dict, predictions: Dict) -> List[Dict]:
        """Find betting opportunities based on hot streaks and recent performance trends"""
        opportunities = []
        
        try:
            for team_type in ['away_team', 'home_team']:
                team_preds = predictions.get(team_type, {})
                
                # Handle case where team_preds is a string
                if isinstance(team_preds, str):
                    continue
                    
                team_name = game.get(team_type, 'Unknown')
                
                # Hot streak batters
                batters = team_preds.get('batters', []) if isinstance(team_preds, dict) else []
                for batter in batters:
                    if isinstance(batter, dict) and self._is_hot_streak_batter(batter):
                        opportunities.append({
                            'type': 'Hot Streak Player',
                            'category': 'Batter Prop',
                            'game': f"{game['away_team']} @ {game['home_team']}",
                            'player': batter.get('name', 'Unknown'),
                            'bet_type': '1+ Hits',
                            'projection': f"{batter.get('hit_probability', 0.6):.1%} hit probability",
                            'edge_factors': ['Hot streak', 'Above season average', 'Recent form'],
                            'reasoning': self._get_hot_streak_reasoning(batter),
                            'raw_confidence': 0.8,
                            'streak_bonus': 0.2
                        })
                
                # Hot streak pitchers
                pitchers = team_preds.get('pitchers', []) if isinstance(team_preds, dict) else []
                for pitcher in pitchers:
                    if isinstance(pitcher, dict) and self._is_hot_streak_pitcher(pitcher):
                        k_projection = pitcher.get('predicted_strikeouts', 6.5)
                        opportunities.append({
                            'type': 'Hot Streak Player',
                            'category': 'Pitcher Prop',
                            'game': f"{game['away_team']} @ {game['home_team']}",
                            'player': pitcher.get('name', 'Unknown'),
                            'bet_type': f'{int(k_projection)}+ Strikeouts',
                            'projection': f"{k_projection:.1f} strikeouts projected",
                            'edge_factors': ['Dominant recent starts', 'Above strikeout rate', 'Good matchup'],
                            'reasoning': self._get_hot_pitcher_reasoning(pitcher),
                            'raw_confidence': 0.75,
                            'streak_bonus': 0.25
                        })
                        
        except Exception as e:
            logger.error(f"Error finding hot streak opportunities: {e}")
        
        return opportunities
    
    def _find_ballpark_advantages(self, game: Dict, predictions: Dict) -> List[Dict]:
        """Find opportunities based on ballpark factors and venue advantages"""
        opportunities = []
        
        try:
            venue = game.get('venue', '')
            ballpark_factor = self._get_ballpark_run_factor(venue)
            
            # Hitter-friendly parks
            if ballpark_factor > 1.05:  # 5%+ boost to offense
                for team_type in ['away_team', 'home_team']:
                    team_preds = predictions.get(team_type, {})
                    
                    # Handle case where team_preds is a string
                    if isinstance(team_preds, str):
                        continue
                    
                    # Power hitters in hitter-friendly parks
                    batters = team_preds.get('batters', []) if isinstance(team_preds, dict) else []
                    for batter in batters:
                        if not isinstance(batter, dict):
                            continue
                        # Extract HR probability from predictions
                        predictions = batter.get('predictions', {})
                        hr_prob = predictions.get('home_run_probability', 0.05)
                        if hr_prob >= 0.06:  # Lower threshold for realistic opportunities
                            opportunities.append({
                                'type': 'Ballpark Advantage',
                                'category': 'Home Run Prop',
                                'game': f"{game['away_team']} @ {game['home_team']}",
                                'player': batter.get('name', 'Unknown'),
                                'bet_type': 'Home Run',
                                'projection': f"{hr_prob:.1%} base probability + ballpark boost",
                                'edge_factors': ['Hitter-friendly ballpark', 'Power hitter profile', 'Venue history'],
                                'reasoning': f"Playing in {venue} (hitter-friendly). {hr_prob:.1%} HR rate gets {ballpark_factor:.1%} boost.",
                                'raw_confidence': 0.7,
                                'ballpark_bonus': (ballpark_factor - 1.0) * 0.5
                            })
            
            # Pitcher-friendly parks
            elif ballpark_factor < 0.95:  # 5%+ boost to pitching
                for team_type in ['away_team', 'home_team']:
                    team_preds = predictions.get(team_type, {})
                    
                    # Handle case where team_preds is a string
                    if isinstance(team_preds, str):
                        continue
                    
                    pitchers = team_preds.get('pitchers', []) if isinstance(team_preds, dict) else []
                    for pitcher in pitchers:
                        if not isinstance(pitcher, dict):
                            continue
                        # Extract strikeout data from predictions
                        predictions = pitcher.get('predictions', {})
                        k_rate = predictions.get('strikeout_probability', 0.25)
                        if k_rate >= 0.20:  # Lower threshold for more opportunities
                            k_projection = predictions.get('projected_strikeouts', 6.0)
                            opportunities.append({
                                'type': 'Ballpark Advantage',
                                'category': 'Strikeout Prop', 
                                'game': f"{game['away_team']} @ {game['home_team']}",
                                'player': pitcher.get('name', 'Unknown'),
                                'bet_type': f'{int(k_projection)}+ Strikeouts',
                                'projection': f"{k_projection:.1f} strikeouts + pitcher-friendly venue",
                                'edge_factors': ['Pitcher-friendly ballpark', 'High K-rate', 'Venue suppresses offense'],
                                'reasoning': f"Pitching in {venue} (pitcher-friendly). {k_rate:.1%} K-rate benefits from venue.",
                                'raw_confidence': 0.72,
                                'ballpark_bonus': (1.0 - ballpark_factor) * 0.4
                            })
                            
        except Exception as e:
            logger.error(f"Error finding ballpark advantages: {e}")
        
        return opportunities
    
    def _find_matchup_edges(self, game: Dict, predictions: Dict) -> List[Dict]:
        """Find opportunities based on specific batter vs pitcher matchups"""
        opportunities = []
        
        try:
            away_preds = predictions.get('away_team', {})
            home_preds = predictions.get('home_team', {})
            
            # Handle string data
            if isinstance(away_preds, str) or isinstance(home_preds, str):
                return opportunities
            
            # Away batters vs home pitcher
            home_pitchers = home_preds.get('pitchers', []) if isinstance(home_preds, dict) else []
            home_pitcher = home_pitchers[0] if home_pitchers and isinstance(home_pitchers[0], dict) else {}
            
            away_batters = away_preds.get('batters', []) if isinstance(away_preds, dict) else []
            for batter in away_batters:
                if isinstance(batter, dict):
                    matchup_edge = self._analyze_batter_pitcher_matchup(batter, home_pitcher)
                    if matchup_edge:
                        opportunities.append(matchup_edge)
            
            # Home batters vs away pitcher  
            away_pitchers = away_preds.get('pitchers', []) if isinstance(away_preds, dict) else []
            away_pitcher = away_pitchers[0] if away_pitchers and isinstance(away_pitchers[0], dict) else {}
            
            home_batters = home_preds.get('batters', []) if isinstance(home_preds, dict) else []
            for batter in home_batters:
                if isinstance(batter, dict):
                    matchup_edge = self._analyze_batter_pitcher_matchup(batter, away_pitcher)
                    if matchup_edge:
                        opportunities.append(matchup_edge)
                    
        except Exception as e:
            logger.error(f"Error finding matchup edges: {e}")
        
        return opportunities
    
    def _find_weather_opportunities(self, game: Dict, predictions: Dict) -> List[Dict]:
        """Find opportunities based on weather conditions"""
        opportunities = []
        
        try:
            weather = game.get('weather', {})
            wind_speed = weather.get('wind_speed', 0)
            wind_direction = weather.get('wind_direction', '')
            temperature = weather.get('temperature', 70)
            
            # Strong wind helping home runs
            if wind_speed >= 15 and 'out' in wind_direction.lower():
                for team_type in ['away_team', 'home_team']:
                    team_preds = predictions.get(team_type, {})
                    
                    # Handle string data
                    if isinstance(team_preds, str):
                        continue
                    
                    batters = team_preds.get('batters', []) if isinstance(team_preds, dict) else []
                    for batter in batters:
                        if not isinstance(batter, dict):
                            continue
                        # Extract HR probability from predictions
                        predictions = batter.get('predictions', {})
                        hr_prob = predictions.get('home_run_probability', 0.05)
                        if hr_prob >= 0.04:  # Lower threshold for more opportunities
                            opportunities.append({
                                'type': 'Weather Advantage',
                                'category': 'Home Run Prop',
                                'game': f"{game['away_team']} @ {game['home_team']}",
                                'player': batter.get('name', 'Unknown'),
                                'bet_type': 'Home Run',
                                'projection': f"{hr_prob:.1%} + wind assistance",
                                'edge_factors': ['Strong tailwind', 'Power hitter', 'Weather boost'],
                                'reasoning': f"{wind_speed}mph wind blowing out. Great conditions for home runs.",
                                'raw_confidence': 0.65,
                                'weather_bonus': min(wind_speed / 100, 0.15)
                            })
            
            # Hot weather boosting offense
            if temperature >= 85:
                total_runs_projection = self._calculate_weather_total_runs(game, predictions, temperature)
                if total_runs_projection >= 9.5:
                    opportunities.append({
                        'type': 'Weather Advantage', 
                        'category': 'Game Total',
                        'game': f"{game['away_team']} @ {game['home_team']}",
                        'player': 'Game Total',
                        'bet_type': 'Over Total Runs',
                        'projection': f"{total_runs_projection:.1f} runs projected",
                        'edge_factors': ['Hot weather', 'Offense-friendly conditions', 'Both lineups benefit'],
                        'reasoning': f"{temperature}°F temperature. Hot weather typically increases scoring.",
                        'raw_confidence': 0.6,
                        'weather_bonus': min((temperature - 75) / 200, 0.1)
                    })
                    
        except Exception as e:
            logger.error(f"Error finding weather opportunities: {e}")
        
        return opportunities
    
    def _analyze_game_betting_value(self, game: Dict, predictions: Dict) -> List[Dict]:
        """Analyze game-level betting opportunities (totals, run lines, etc.)"""
        opportunities = []
        
        try:
            # Get team predictions
            away_preds = predictions.get('away_team', {})
            home_preds = predictions.get('home_team', {})
            
            # Calculate projected total runs
            away_runs = self._calculate_team_run_projection(away_preds)
            home_runs = self._calculate_team_run_projection(home_preds)
            projected_total = away_runs + home_runs
            
            # Weather and ballpark adjustments
            ballpark_factor = self._get_ballpark_run_factor(game.get('venue', ''))
            weather_factor = self._get_weather_run_factor(game)
            adjusted_total = projected_total * ballpark_factor * weather_factor
            
            # Analyze over/under opportunities
            if 'over_under' in game:
                book_total = float(game['over_under'])
                total_edge = abs(adjusted_total - book_total) / book_total
                
                if total_edge >= self.betting_thresholds['min_edge']:
                    bet_type = "OVER" if adjusted_total > book_total else "UNDER"
                    opportunities.append({
                        'type': 'Total Runs',
                        'game': f"{game['away_team']} @ {game['home_team']}",
                        'bet': f"{bet_type} {book_total}",
                        'projection': round(adjusted_total, 1),
                        'edge': round(total_edge * 100, 1),
                        'confidence': self._calculate_confidence(total_edge, 'total'),
                        'reasoning': f"Projected {adjusted_total:.1f} vs book {book_total}",
                        'recommended_units': self._calculate_bet_size(total_edge, 'total')
                    })
            
            # Analyze run line opportunities
            run_line_analysis = self._analyze_run_line(game, away_runs, home_runs, ballpark_factor)
            opportunities.extend(run_line_analysis)
            
        except Exception as e:
            logger.error(f"Error in game betting analysis: {e}")
        
        return opportunities
    
    def _analyze_player_props(self, game: Dict, predictions: Dict) -> List[Dict]:
        """Analyze player proposition betting opportunities"""
        opportunities = []
        
        for team_type in ['away_team', 'home_team']:
            team_preds = predictions.get(team_type, {})
            team_name = game.get(team_type, '')
            
            # Batter props
            for batter in team_preds.get('batters', [])[:9]:  # Starting lineup
                batter_props = self._analyze_batter_props(batter, team_name, game)
                opportunities.extend(batter_props)
            
            # Pitcher props
            pitchers = team_preds.get('pitchers', [])
            if pitchers:
                starter = pitchers[0]  # Starting pitcher
                pitcher_props = self._analyze_pitcher_props(starter, team_name, game)
                opportunities.extend(pitcher_props)
        
        return opportunities
    
    def _analyze_batter_props(self, batter: Dict, team: str, game: Dict) -> List[Dict]:
        """Analyze betting opportunities for batter props"""
        opportunities = []
        
        try:
            if not isinstance(batter, dict):
                return opportunities
                
            name = batter.get('name', 'Unknown')
            
            # Hits prop analysis - check multiple possible field names
            hit_prob = batter.get('hit_probability', 
                        batter.get('predicted_batting_avg', 
                        batter.get('batting_avg', 0.0)))
            
            # Convert batting average to hit probability if needed
            if hit_prob <= 1.0 and hit_prob > 0:
                hit_prob = min(hit_prob * 3.5, 0.85)  # Convert avg to probability
            
            if hit_prob >= 0.55:  # Lowered threshold for more opportunities
                edge = max(0, (hit_prob - 0.52) * 100)  # More conservative baseline
                if edge >= 3:  # Minimum 3% edge
                    opportunities.append({
                        'type': 'Player Prop',
                        'game': f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                        'bet': f"{name} 1+ Hits",
                        'projection': f"{hit_prob:.1%} probability",
                        'edge': round(edge, 1),
                        'confidence': 'High' if hit_prob >= 0.70 else 'Medium',
                        'reasoning': self._get_batter_reasoning(batter),
                        'recommended_units': self._calculate_bet_size(edge / 100, 'prop')
                    })
            
            # Home run prop analysis
            hr_prob = batter.get('home_run_probability', 
                        batter.get('predicted_home_runs', 0.0))
            
            if hr_prob >= 0.08:  # Lowered threshold
                edge = max(0, (hr_prob - 0.06) * 100)  # More conservative baseline
                if edge >= 2:  # Minimum 2% edge for HR props
                    opportunities.append({
                        'type': 'Player Prop',
                        'game': f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                        'bet': f"{name} Home Run",
                        'projection': f"{hr_prob:.1%} probability",
                        'edge': round(edge, 1),
                        'confidence': 'High' if hr_prob >= 0.18 else 'Medium',
                        'reasoning': f"HR probability: {hr_prob:.1%}, {self._get_batter_reasoning(batter)}",
                        'recommended_units': self._calculate_bet_size(edge / 100, 'prop')
                    })
            
            # RBI opportunities
            rbi_projection = batter.get('predicted_rbis', 
                            batter.get('rbi_projection', 0.0))
            if rbi_projection >= 1.0:  # Lowered threshold
                edge = max(0, (rbi_projection - 0.8) * 25)
                if edge >= 3:
                    opportunities.append({
                        'type': 'Player Prop',
                        'game': f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                        'bet': f"{name} 1+ RBI",
                        'projection': f"{rbi_projection:.1f} projected RBIs",
                        'edge': round(edge, 1),
                        'confidence': 'Medium',
                        'reasoning': f"Projects {rbi_projection:.1f} RBIs",
                        'recommended_units': self._calculate_bet_size(edge / 100, 'prop')
                    })
                
        except Exception as e:
            logger.error(f"Error analyzing batter props: {e}")
        
        return opportunities
    
    def _analyze_pitcher_props(self, pitcher: Dict, team: str, game: Dict) -> List[Dict]:
        """Analyze betting opportunities for pitcher props"""
        opportunities = []
        
        try:
            if not isinstance(pitcher, dict):
                return opportunities
                
            name = pitcher.get('name', 'Unknown')
            
            # Strikeout props - check multiple field names
            k_projection = pitcher.get('predicted_strikeouts', 
                            pitcher.get('strikeout_projection', 
                            pitcher.get('strikeouts', 0.0)))
            
            if k_projection >= 5.5:  # Lowered threshold
                edge = max(0, (k_projection - 5.0) * 12)
                if edge >= 3:
                    opportunities.append({
                        'type': 'Player Prop',
                        'game': f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                        'bet': f"{name} {int(k_projection-0.5)}+ Strikeouts",
                        'projection': f"{k_projection:.1f} projected K's",
                        'edge': round(edge, 1),
                        'confidence': 'High' if k_projection >= 7.5 else 'Medium',
                        'reasoning': self._get_pitcher_reasoning(pitcher),
                        'recommended_units': self._calculate_bet_size(edge / 100, 'prop')
                    })
            
            # ERA-based opportunities
            era = pitcher.get('season_era', pitcher.get('era', 4.50))
            recent_era = pitcher.get('recent_era', era)
            
            # Generate opportunities for good pitchers
            if era <= 4.00 and recent_era <= 3.75:
                edge = max(0, (4.00 - era) * 15)
                if edge >= 3:
                    opportunities.append({
                        'type': 'Player Prop',
                        'game': f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                        'bet': f"{name} Quality Start",
                        'projection': f"ERA: {era:.2f}, Recent: {recent_era:.2f}",
                        'edge': round(edge, 1),
                        'confidence': 'High' if era <= 3.00 else 'Medium',
                        'reasoning': f"Strong form: {recent_era:.2f} recent ERA vs {era:.2f} season",
                        'recommended_units': self._calculate_bet_size(edge / 100, 'prop')
                    })
                
        except Exception as e:
            logger.error(f"Error analyzing pitcher props: {e}")
        
        return opportunities
    
    def _calculate_team_run_projection(self, team_preds: Dict) -> float:
        """Calculate projected runs for a team"""
        try:
            # Handle different data structures
            if isinstance(team_preds, str):
                return 4.5  # Fallback for string data
                
            batters = team_preds.get('batters', []) if isinstance(team_preds, dict) else []
            if not batters:
                return 4.5  # League average
            
            # Sum expected offensive production
            total_hits = 0
            total_hrs = 0
            
            for b in batters[:9]:  # Starting lineup
                if isinstance(b, dict):
                    avg = b.get('predicted_batting_avg', b.get('batting_avg', 0.250))
                    hrs = b.get('predicted_home_runs', b.get('home_run_probability', 0.05))
                    total_hits += avg * 4
                    total_hrs += hrs
            
            # Convert to run projection using linear weights
            projected_runs = (total_hits * 0.5) + (total_hrs * 1.4) + 1.0  # Base runs
            
            return max(2.0, min(12.0, projected_runs))  # Reasonable bounds
            
        except Exception as e:
            logger.error(f"Error calculating team run projection: {e}")
            return 4.5
    
    def _get_ballpark_run_factor(self, venue: str) -> float:
        """Get ballpark run scoring factor"""
        ballpark_factors = {
            'Coors Field': 1.15,  # High altitude, offense friendly
            'Yankee Stadium': 1.08,  # Short porch
            'Fenway Park': 1.05,  # Green Monster
            'Minute Maid Park': 1.03,
            'Citizens Bank Park': 1.02,
            'Great American Ball Park': 1.02,
            'Globe Life Field': 1.01,
            'Tropicana Field': 0.95,  # Pitcher friendly
            'Marlins Park': 0.96,
            'Petco Park': 0.94,
            'Oakland Coliseum': 0.93,
            'T-Mobile Park': 0.95
        }
        
        return ballpark_factors.get(venue, 1.0)  # Neutral if unknown
    
    def _get_weather_run_factor(self, game: Dict) -> float:
        """Get weather impact on run scoring"""
        # Simplified weather factor - in production would use real weather API
        return 1.0  # Neutral for now
    
    def _analyze_run_line(self, game: Dict, away_runs: float, home_runs: float, ballpark_factor: float) -> List[Dict]:
        """Analyze run line betting opportunities"""
        opportunities = []
        
        try:
            run_diff = abs(home_runs - away_runs) * ballpark_factor
            
            if run_diff >= 1.8:  # Significant projected difference
                favorite = game['home_team'] if home_runs > away_runs else game['away_team']
                line = -1.5 if run_diff >= 2.0 else -1.0
                
                opportunities.append({
                    'type': 'Run Line',
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'bet': f"{favorite} {line}",
                    'projection': f"{run_diff:.1f} run difference",
                    'edge': round((run_diff - 1.5) * 30, 1),
                    'confidence': 'High' if run_diff >= 2.5 else 'Medium',
                    'reasoning': f"Projected {favorite} to win by {run_diff:.1f}",
                    'recommended_units': self._calculate_bet_size((run_diff - 1.5) / 3, 'spread')
                })
                
        except Exception as e:
            logger.error(f"Error analyzing run line: {e}")
        
        return opportunities
    
    def _get_batter_reasoning(self, batter: Dict) -> str:
        """Generate reasoning for batter prop bet"""
        reasons = []
        
        avg = batter.get('predicted_batting_avg', 0.0)
        if avg >= 0.300:
            reasons.append(f"Hot bat (.{int(avg*1000)})")
        
        vs_pitcher = batter.get('vs_pitcher_avg', 0.0)
        if vs_pitcher >= 0.350:
            reasons.append(f"Good matchup (.{int(vs_pitcher*1000)} career)")
        
        recent = batter.get('recent_avg', 0.0)
        if recent >= 0.320:
            reasons.append(f"Recent form (.{int(recent*1000)})")
        
        return ", ".join(reasons) if reasons else "Solid projection"
    
    def _get_pitcher_reasoning(self, pitcher: Dict) -> str:
        """Generate reasoning for pitcher prop bet"""
        reasons = []
        
        k_rate = pitcher.get('strikeout_rate', 0.0)
        if k_rate >= 0.28:
            reasons.append(f"High K rate ({k_rate:.1%})")
        
        recent_era = pitcher.get('recent_era', 5.0)
        if recent_era <= 3.0:
            reasons.append(f"Strong recent form ({recent_era:.2f} ERA)")
        
        vs_team = pitcher.get('vs_opp_era', 5.0)
        if vs_team <= 3.5:
            reasons.append(f"Good vs opponent ({vs_team:.2f} ERA)")
        
        return ", ".join(reasons) if reasons else "Solid projection"
    
    def _calculate_confidence(self, edge: float, bet_type: str) -> str:
        """Calculate confidence level for bet"""
        if edge >= 0.15:
            return "Very High"
        elif edge >= 0.10:
            return "High"
        elif edge >= 0.05:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_bet_size(self, edge: float, bet_type: str) -> float:
        """Calculate recommended bet size using Kelly Criterion-like approach"""
        try:
            # Simplified Kelly formula: (edge * confidence) / variance
            max_units = 5.0  # Maximum bet size
            
            if edge <= 0:
                return 0.0
            
            # Conservative sizing
            if bet_type == 'prop':
                base_size = min(edge * 8, max_units)  # More aggressive on props
            else:
                base_size = min(edge * 6, max_units)  # Conservative on totals/spreads
            
            return round(max(0.5, base_size), 1)
            
        except Exception as e:
            logger.error(f"Error calculating bet size: {e}")
            return 1.0
    
    def calculate_bankroll_allocation(self, opportunities: List[Dict], total_bankroll: float) -> Dict:
        """Calculate optimal bankroll allocation across opportunities"""
        try:
            if not opportunities or total_bankroll <= 0:
                return {'total_risk': 0, 'recommendations': []}
            
            # Sort by betting_edge and confidence_score
            sorted_opps = sorted(opportunities, 
                               key=lambda x: (x.get('betting_edge', 0), x.get('confidence_score', 0)), 
                               reverse=True)
            
            total_units = sum(opp.get('recommended_units', 1.0) for opp in sorted_opps[:10])
            max_risk = total_bankroll * self.betting_thresholds['max_risk']
            
            # Scale down if total risk exceeds threshold
            if total_units * 100 > max_risk:  # Assuming $100 per unit
                scale_factor = max_risk / (total_units * 100)
                for opp in sorted_opps:
                    opp['recommended_units'] *= scale_factor
            
            return {
                'total_risk': min(total_units * 100, max_risk),
                'max_single_bet': max_risk * 0.3,  # No more than 30% of max risk on one bet
                'number_of_bets': len(sorted_opps[:10]),
                'expected_roi': sum(opp.get('betting_edge', 0) * opp.get('recommended_units', 1) for opp in sorted_opps[:10]),
                'recommendations': sorted_opps[:10]
            }
            
        except Exception as e:
            logger.error(f"Error calculating bankroll allocation: {e}")
            return {'total_risk': 0, 'recommendations': []}
    
    def generate_betting_report(self, opportunities: List[Dict], bankroll: float = 10000) -> Dict:
        """Generate comprehensive betting report"""
        try:
            allocation = self.calculate_bankroll_allocation(opportunities, bankroll)
            
            # Categorize opportunities using correct field names
            high_value = [o for o in opportunities if o.get('betting_edge', 0) >= 10]
            medium_value = [o for o in opportunities if 5 <= o.get('betting_edge', 0) < 10]
            low_value = [o for o in opportunities if o.get('betting_edge', 0) < 5]
            
            # Calculate metrics using correct field names
            total_expected_value = sum(o.get('betting_edge', 0) * o.get('recommended_units', 1) for o in opportunities[:10])
            
            report = {
                'summary': {
                    'total_opportunities': len(opportunities),
                    'high_value_bets': len(high_value),
                    'medium_value_bets': len(medium_value),
                    'recommended_bets': len(allocation['recommendations']),
                    'total_risk': allocation['total_risk'],
                    'expected_profit': total_expected_value * 100,  # Assuming $100 units
                    'roi_estimate': f"{(total_expected_value / allocation['total_risk'] * 100):.1f}%" if allocation['total_risk'] > 0 else "0%"
                },
                'top_opportunities': opportunities[:5],
                'allocation': allocation,
                'risk_management': {
                    'max_single_bet': allocation.get('max_single_bet', 0),
                    'diversification': len(set(o.get('type', 'Unknown') for o in allocation.get('recommendations', []))),
                    'confidence_distribution': self._analyze_confidence_distribution(opportunities)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating betting report: {e}")
            return {'summary': {}, 'top_opportunities': [], 'allocation': {}}
    
    def _analyze_confidence_distribution(self, opportunities: List[Dict]) -> Dict:
        """Analyze confidence distribution of opportunities"""
        confidence_counts = {}
        for opp in opportunities:
            conf = opp.get('confidence', 'Low')
            confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        
        return confidence_counts
    
    def _is_hot_streak_batter(self, batter: Dict) -> bool:
        """Determine if batter is on a hot streak"""
        try:
            # Work with actual data structure from real_prediction_engine
            predictions = batter.get('predictions', {})
            season_stats = batter.get('season_stats', {})
            
            # Check if already marked as hot streak
            if predictions.get('hot_streak'):
                return True
            
            # Check high hit probability
            hit_prob = predictions.get('hit_probability', 0.5)
            if hit_prob > 0.65:
                return True
                
            # Check season batting average
            batting_avg = season_stats.get('avg', 0.0)
            if isinstance(batting_avg, str):
                try:
                    batting_avg = float(batting_avg)
                except:
                    batting_avg = 0.0
                    
            # Hot if hitting above .300 or high confidence
            if batting_avg > 0.300 or predictions.get('confidence', 0) > 0.75:
                return True
                
            return False
        except:
            return False
    
    def _is_hot_streak_pitcher(self, pitcher: Dict) -> bool:
        """Determine if pitcher is on a hot streak"""
        try:
            # Work with actual data structure from real_prediction_engine
            predictions = pitcher.get('predictions', {})
            season_stats = pitcher.get('season_stats', {})
            
            # Check if already marked as hot streak
            if predictions.get('hot_streak'):
                return True
            
            # Check high strikeout probability
            k_prob = predictions.get('strikeout_probability', 0.20)
            if k_prob > 0.30:
                return True
                
            # Check season ERA
            era = season_stats.get('era', 5.00)
            if isinstance(era, str):
                try:
                    era = float(era)
                except:
                    era = 5.00
                    
            # Hot if ERA under 3.25 or high confidence
            if era < 3.25 or predictions.get('confidence', 0) > 0.75:
                return True
                
            return False
        except:
            return False
    
    def _get_hot_streak_reasoning(self, batter: Dict) -> str:
        """Generate reasoning for hot streak batter"""
        predictions = batter.get('predictions', {})
        season_stats = batter.get('season_stats', {})
        
        batting_avg = season_stats.get('avg', 0.250)
        if isinstance(batting_avg, str):
            try:
                batting_avg = float(batting_avg)
            except:
                batting_avg = 0.250
                
        hit_prob = predictions.get('hit_probability', 0.6)
        
        if predictions.get('hot_streak'):
            return f"Marked as hot streak player. {hit_prob:.1%} hit probability today."
        elif batting_avg > 0.300:
            return f"Strong {batting_avg:.3f} season average. Consistent contact hitter."
        else:
            return f"High confidence prediction with {hit_prob:.1%} hit probability."
    
    def _get_hot_pitcher_reasoning(self, pitcher: Dict) -> str:
        """Generate reasoning for hot streak pitcher"""
        predictions = pitcher.get('predictions', {})
        season_stats = pitcher.get('season_stats', {})
        
        era = season_stats.get('era', 4.50)
        if isinstance(era, str):
            try:
                era = float(era)
            except:
                era = 4.50
                
        k_rate = predictions.get('strikeout_probability', 0.20)
        k_projection = predictions.get('projected_strikeouts', 6.5)
        
        if predictions.get('hot_streak'):
            return f"Marked as hot streak pitcher. {k_projection:.1f} strikeouts projected."
        elif era < 3.25:
            return f"Excellent {era:.2f} season ERA. Dominant form."
        elif k_rate > 0.30:
            return f"Elite {k_rate:.1%} strikeout rate. Missing bats consistently."
        else:
            return f"High confidence prediction with {k_projection:.1f} K projection."
    
    def _analyze_batter_pitcher_matchup(self, batter: Dict, pitcher: Dict) -> Dict:
        """Analyze specific batter vs pitcher matchup for edges"""
        try:
            if not batter or not pitcher:
                return None
                
            batter_hand = batter.get('bats', 'R')  # R/L/S
            pitcher_hand = pitcher.get('throws', 'R')  # R/L
            
            # Platoon advantages
            platoon_advantage = False
            if (batter_hand == 'L' and pitcher_hand == 'R') or (batter_hand == 'R' and pitcher_hand == 'L'):
                platoon_advantage = True
            
            # Contact vs power pitcher mismatch
            batter_preds = batter.get('predictions', {})
            pitcher_preds = pitcher.get('predictions', {})
            
            batter_contact = batter_preds.get('contact_rate', 0.75)
            pitcher_k_rate = pitcher_preds.get('strikeout_probability', 0.25)
            
            # Good contact hitter vs high strikeout pitcher = edge for contact
            if batter_contact > 0.80 and pitcher_k_rate > 0.28:
                return {
                    'type': 'Matchup Edge',
                    'category': 'Contact vs Power',
                    'game': f"Matchup Analysis",
                    'player': batter.get('name', 'Unknown'),
                    'bet_type': '1+ Hits',
                    'projection': f"High contact vs high K-rate pitcher",
                    'edge_factors': ['Contact hitter', 'Power pitcher mismatch', 'Historical advantage'],
                    'reasoning': f"Contact specialist ({batter_contact:.1%}) vs strikeout pitcher ({pitcher_k_rate:.1%}). Good spots for contact.",
                    'raw_confidence': 0.68,
                    'matchup_bonus': 0.12
                }
            
            # Platoon advantage for power hitters
            batter_hr_prob = batter_preds.get('home_run_probability', 0.05)
            if platoon_advantage and batter_hr_prob > 0.05:
                return {
                    'type': 'Matchup Edge',
                    'category': 'Platoon Advantage',
                    'game': f"Matchup Analysis", 
                    'player': batter.get('name', 'Unknown'),
                    'bet_type': 'Home Run',
                    'projection': f"Platoon advantage + power",
                    'edge_factors': ['Favorable handedness', 'Power profile', 'Platoon split'],
                    'reasoning': f"{batter_hand}H batter vs {pitcher_hand}P pitcher. Classic platoon advantage for power.",
                    'raw_confidence': 0.70,
                    'matchup_bonus': 0.15
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing matchup: {e}")
            return None
    
    def _calculate_weather_total_runs(self, game: Dict, predictions: Dict, temperature: int) -> float:
        """Calculate projected total runs with weather adjustment"""
        try:
            away_runs = self._calculate_team_run_projection(predictions.get('away_team', {}))
            home_runs = self._calculate_team_run_projection(predictions.get('home_team', {}))
            base_total = away_runs + home_runs
            
            # Hot weather boost (ball carries better, players more active)
            if temperature >= 85:
                weather_boost = min((temperature - 75) * 0.01, 0.15)  # Max 15% boost
                return base_total * (1 + weather_boost)
            
            return base_total
        except:
            return 9.0  # Default projection
    
    def _calculate_confidence_score(self, opportunity: Dict) -> float:
        """Calculate overall confidence score for opportunity"""
        try:
            base_confidence = opportunity.get('raw_confidence', 0.5)
            
            # Add bonuses from different factors
            streak_bonus = opportunity.get('streak_bonus', 0.0)
            ballpark_bonus = opportunity.get('ballpark_bonus', 0.0)
            weather_bonus = opportunity.get('weather_bonus', 0.0)
            matchup_bonus = opportunity.get('matchup_bonus', 0.0)
            
            total_confidence = base_confidence + streak_bonus + ballpark_bonus + weather_bonus + matchup_bonus
            return min(total_confidence, 0.95)  # Cap at 95%
        except:
            return 0.5
    
    def _calculate_betting_edge(self, opportunity: Dict) -> float:
        """Calculate betting edge percentage"""
        try:
            confidence = opportunity.get('confidence_score', 0.5)
            edge_factors = len(opportunity.get('edge_factors', []))
            
            # Base edge from confidence above 50%
            base_edge = max(0, (confidence - 0.50) * 100)
            
            # Bonus for multiple edge factors
            factor_bonus = min(edge_factors * 0.5, 2.0)
            
            return base_edge + factor_bonus
        except:
            return 0.0
    
    def _calculate_recommended_units(self, opportunity: Dict) -> float:
        """Calculate recommended bet size in units based on Kelly-like formula"""
        try:
            edge = opportunity.get('betting_edge', 0.0) / 100  # Convert to decimal
            confidence = opportunity.get('confidence_score', 0.5)
            
            # Conservative Kelly-like sizing
            # Standard Kelly: f = (bp - q) / b, but we use conservative approach
            if edge <= 0:
                return 0.0
            
            # Base sizing with confidence modifier
            base_units = edge * confidence * 10  # Conservative multiplier
            
            # Cap bet sizes for safety
            max_units = 3.0  # Maximum 3 units per bet
            
            return round(min(max(base_units, 0.5), max_units), 1)
            
        except:
            return 1.0  # Default 1 unit
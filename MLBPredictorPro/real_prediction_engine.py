"""
Real MLB prediction engine using authentic data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from real_data_fetcher import RealMLBDataFetcher
import warnings
warnings.filterwarnings('ignore')

class RealMLBPredictionEngine:
    """
    Generates real MLB predictions using authentic data sources
    """
    
    def __init__(self):
        self.data_fetcher = RealMLBDataFetcher()
        
    def generate_game_predictions(self, game_info: Dict) -> Dict:
        """Generate comprehensive predictions for a game"""
        predictions = {
            'game_id': game_info.get('game_id'),
            'home_team': game_info.get('home_team'),
            'away_team': game_info.get('away_team'),
            'home_players': [],
            'away_players': [],
            'team_predictions': {},
            'key_matchups': [],
            'data_source': 'MLB Stats API + Baseball Savant'
        }
        
        try:
            # Get team rosters and predictions
            home_team_id = game_info.get('home_team_id')
            away_team_id = game_info.get('away_team_id')
            
            if home_team_id and away_team_id:
                predictions['home_players'] = self._get_team_predictions(home_team_id, 'home', game_info)
                predictions['away_players'] = self._get_team_predictions(away_team_id, 'away', game_info)
                predictions['team_predictions'] = self.data_fetcher.get_team_win_probability(home_team_id, away_team_id)
                
                # Generate key matchups
                predictions['key_matchups'] = self._generate_key_matchups(
                    predictions['home_players'], 
                    predictions['away_players'],
                    game_info
                )
        
        except Exception as e:
            print(f"Error generating predictions for game {game_info.get('game_id')}: {e}")
            predictions['error'] = str(e)
        
        return predictions
    
    def _get_team_predictions(self, team_id: int, home_away: str, game_info: Dict) -> List[Dict]:
        """Get predictions for all players on a team"""
        try:
            roster = self.data_fetcher.get_team_roster(team_id)
            player_predictions = []
            
            for player in roster:
                player_id = player.get('id')
                position_type = player.get('position_type')
                
                if not player_id:
                    continue
                
                # Determine if batter or pitcher
                is_pitcher = position_type == 'Pitcher'
                
                # FILTER: For pitchers, only include the starting pitcher for today's game
                if is_pitcher:
                    probable_pitcher_id = None
                    if home_away == 'home' and game_info.get('home_pitcher', {}).get('id'):
                        probable_pitcher_id = game_info['home_pitcher']['id']
                    elif home_away == 'away' and game_info.get('away_pitcher', {}).get('id'):
                        probable_pitcher_id = game_info['away_pitcher']['id']
                    
                    # Skip this pitcher if they're not the probable starter
                    if probable_pitcher_id and player_id != probable_pitcher_id:
                        continue
                    # If no probable pitcher identified, skip all pitchers to avoid confusion
                    elif not probable_pitcher_id:
                        continue
                
                stats_type = 'pitching' if is_pitcher else 'hitting'
                
                # Get season stats
                season_stats = self.data_fetcher.get_player_season_stats(player_id, stats_type)
                
                # Get advanced stats from Baseball Savant
                advanced_stats = self.data_fetcher.get_baseball_savant_data(
                    player_id, 
                    'pitching' if is_pitcher else 'hitting'
                )
                
                # Calculate predictions with enhanced rolling averages and matchup history
                player_type = 'pitcher' if is_pitcher else 'batter'
                predictions = self.data_fetcher.calculate_predictions(
                    season_stats, 
                    advanced_stats, 
                    player_type,
                    player_id  # Pass player_id for rolling averages
                )
                
                # Add batter vs pitcher history if this is a batter
                if player_type == 'batter':
                    # Find opposing starting pitcher
                    opposing_pitcher_id = None
                    if home_away == 'home':
                        # Home batter vs away pitcher
                        if game_info.get('away_pitcher', {}).get('id'):
                            opposing_pitcher_id = game_info['away_pitcher']['id']
                    else:
                        # Away batter vs home pitcher
                        if game_info.get('home_pitcher', {}).get('id'):
                            opposing_pitcher_id = game_info['home_pitcher']['id']
                    
                    if opposing_pitcher_id:
                        matchup_history = self.data_fetcher.get_batter_vs_pitcher_history(player_id, opposing_pitcher_id)
                        if matchup_history:
                            predictions['matchup_history'] = matchup_history
                
                # Compile player prediction data
                player_pred = {
                    'name': player.get('name', 'Unknown'),
                    'position': player.get('position'),
                    'jersey_number': player.get('jersey_number'),
                    'player_type': player_type,
                    'season_stats': season_stats,
                    'predictions': predictions,
                    'data_quality': 'real' if season_stats else 'limited'
                }
                
                player_predictions.append(player_pred)
            
            # Sort batters by likely batting order, pitchers by role
            player_predictions.sort(key=lambda x: (
                x['player_type'] == 'pitcher',  # Batters first
                x['position'] in ['SP', 'RP'],  # Starting pitchers first among pitchers
                -(x['predictions'].get('confidence', 0))  # Higher confidence first
            ))
            
            return player_predictions
            
        except Exception as e:
            print(f"Error getting team predictions for team {team_id}: {e}")
            return []
    
    def _generate_key_matchups(self, home_players: List[Dict], away_players: List[Dict], game_info: Dict) -> List[str]:
        """Generate key matchup insights"""
        insights = []
        
        try:
            # Find top batters and pitchers
            home_batters = [p for p in home_players if p['player_type'] == 'batter'][:5]
            away_batters = [p for p in away_players if p['player_type'] == 'batter'][:5]
            home_pitchers = [p for p in home_players if p['player_type'] == 'pitcher'][:3]
            away_pitchers = [p for p in away_players if p['player_type'] == 'pitcher'][:3]
            
            # Probable pitcher matchup
            home_pitcher_info = game_info.get('home_pitcher')
            away_pitcher_info = game_info.get('away_pitcher')
            
            if home_pitcher_info and away_pitcher_info:
                insights.append(f"🥎 Pitching Matchup: {away_pitcher_info['name']} vs {home_pitcher_info['name']}")
            
            # Hot batters
            for player in home_batters + away_batters:
                if player['predictions'].get('hot_streak'):
                    hit_prob = player['predictions'].get('hit_probability', 0)
                    insights.append(f"🔥 {player['name']} is hot - {hit_prob:.1%} hit probability")
            
            # High strikeout pitchers
            for player in home_pitchers + away_pitchers:
                k_prob = player['predictions'].get('strikeout_probability', 0)
                if k_prob > 0.25:
                    insights.append(f"⚡ {player['name']} - {k_prob:.1%} strikeout rate")
            
            # Home run threats
            for player in home_batters + away_batters:
                hr_prob = player['predictions'].get('home_run_probability', 0)
                if hr_prob > 0.05:
                    insights.append(f"💥 {player['name']} - {hr_prob:.1%} home run probability")
            
            # Team win probability insight
            team_predictions = game_info.get('team_predictions', {})
            home_win_prob = team_predictions.get('home_win_probability', 0.5)
            if home_win_prob > 0.6:
                insights.append(f"🏠 {game_info['home_team']} favored with {home_win_prob:.1%} win probability")
            elif home_win_prob < 0.4:
                insights.append(f"✈️ {game_info['away_team']} favored with {1-home_win_prob:.1%} win probability")
            else:
                insights.append("⚖️ Evenly matched teams - coin flip game")
        
        except Exception as e:
            print(f"Error generating key matchups: {e}")
            insights.append("Unable to generate matchup insights - check data connection")
        
        return insights[:6]  # Return top 6 insights
    
    def test_data_sources(self) -> Dict:
        """Test connectivity to real data sources"""
        return self.data_fetcher.test_api_connectivity()
    
    def get_data_source_status(self) -> str:
        """Get current data source status"""
        test_results = self.test_data_sources()
        
        if test_results['mlb_stats_api'] and test_results['baseball_savant']:
            return "✅ Connected to MLB Stats API and Baseball Savant"
        elif test_results['mlb_stats_api']:
            return "⚠️ Connected to MLB Stats API only (Baseball Savant unavailable)"
        elif test_results['baseball_savant']:
            return "⚠️ Connected to Baseball Savant only (MLB Stats API unavailable)"
        else:
            error_msg = "; ".join(test_results.get('error_messages', []))
            return f"❌ No data sources available: {error_msg}"
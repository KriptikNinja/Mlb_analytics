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
        
    def generate_game_predictions(self, game_info: Dict, simplified: bool = False) -> Dict:
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
                if simplified:
                    # Simplified mode - get basic roster without detailed stats to avoid slow API calls
                    predictions['home_players'] = self._get_simplified_team_predictions(home_team_id, 'home', game_info)
                    predictions['away_players'] = self._get_simplified_team_predictions(away_team_id, 'away', game_info)
                else:
                    predictions['home_players'] = self._get_team_predictions(home_team_id, 'home', game_info)
                    predictions['away_players'] = self._get_team_predictions(away_team_id, 'away', game_info)
                    
                    # Calculate realistic win probabilities based on team performance
                    home_win_prob = self._calculate_team_win_probability(predictions['home_players'], predictions['away_players'], True)
                    away_win_prob = 1.0 - home_win_prob
                    
                    predictions['home_win_probability'] = home_win_prob
                    predictions['away_win_probability'] = away_win_prob
                    predictions['team_predictions'] = {
                        'home_win_probability': home_win_prob,
                        'away_win_probability': away_win_prob
                    }
                    
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
    
    def _get_simplified_team_predictions(self, team_id: int, home_away: str, game_info: Dict) -> List[Dict]:
        """Get simplified team predictions without slow API calls for betting engine"""
        try:
            roster = self.data_fetcher.get_team_roster(team_id)
            simplified_predictions = []
            
            for player in roster[:12]:  # Get more players for betting
                player_id = player.get('id')
                position_type = player.get('position_type')
                
                if not player_id:
                    print(f"⚠️ Skipping player {player.get('name', 'Unknown')} - no ID")
                    continue
                
                # Get actual stats from data fetcher - skip if no authentic data
                stats = self.data_fetcher.get_player_season_stats(player_id)
                if not stats or not stats.get('avg') or stats.get('data_source') in ['fallback', 'error', 'none']:
                    print(f"❌ EXCLUDING {player.get('name', 'Unknown')} (ID: {player_id}) - no authentic MLB data")
                    continue
                
                # Skip recent stats for now to avoid missing method error
                recent_stats = {}
                
                # Debug actual API data
                print(f"✅ AUTHENTIC DATA for {player.get('name', 'Unknown')} (ID: {player_id}): {stats.get('data_source', 'no_source')} - AVG: {stats.get('avg', 'missing')}")
                # Create proper structure for betting engine
                prediction = {
                    'id': player_id,
                    'player_id': player_id,
                    'name': player.get('name', 'Unknown Player'),
                    'position': player.get('position', 'Unknown'),
                    'player_type': 'batter' if position_type != 'Pitcher' else 'pitcher',
                    'team_id': team_id,
                    'batting_order': len(simplified_predictions) + 1,
                    
                    # Add season_stats structure for betting engine
                    'season_stats': {
                        'avg': stats.get('avg'),
                        'obp': stats.get('obp'),
                        'slg': stats.get('slg'),
                        'homeRuns': stats.get('homeRuns'),
                        'rbi': stats.get('rbi'),
                        'data_source': 'mlb_api_authentic'
                    },
                    
                    # ONLY authentic MLB API stats - skip player if no real data
                    'avg': float(stats.get('avg', 0)) if stats.get('avg') else 0,
                    'obp': float(stats.get('obp', 0)) if stats.get('obp') else 0,
                    'slg': float(stats.get('slg', 0)) if stats.get('slg') else 0,
                    'ops': (float(stats.get('obp', 0)) + float(stats.get('slg', 0))) if stats.get('obp') and stats.get('slg') else 0,
                    'era': float(stats.get('era', 0)) if position_type == 'Pitcher' and stats.get('era') else 0,
                    'whip': float(stats.get('whip', 0)) if position_type == 'Pitcher' and stats.get('whip') else 0,
                    'k_9': float(stats.get('strikeoutsPerNine', 0)) if position_type == 'Pitcher' and stats.get('strikeoutsPerNine') else 0,
                    
                    # Predictions structure with authentic data only - add proper pitcher stats
                    'predictions': {
                        'obp': float(stats.get('obp', 0)) if stats.get('obp') else 0,
                        'slg': float(stats.get('slg', 0)) if stats.get('slg') else 0,
                        'hr_rate': float(stats.get('homeRuns', 0)) / max(float(stats.get('atBats', 1)), 1) if stats.get('homeRuns') and stats.get('atBats') else 0,
                        'recent_performance': float(recent_stats.get('avg', 0)) / float(stats.get('avg', 1)) if recent_stats.get('avg') and stats.get('avg', 0) > 0 else 1.0,
                        # Add pitcher predictions
                        'strikeouts': float(stats.get('strikeoutsPerNine', 0)) * 5.5 / 9 if position_type == 'Pitcher' and stats.get('strikeoutsPerNine') else None,
                        'era': float(stats.get('era', 0)) if position_type == 'Pitcher' and stats.get('era') else None,
                        'walks': float(stats.get('walksPer9Inn', 0)) * 5.5 / 9 if position_type == 'Pitcher' and stats.get('walksPer9Inn') else None,
                        'strikeout_probability': float(stats.get('strikeoutsPerNine', 0)) / 27 if position_type == 'Pitcher' and stats.get('strikeoutsPerNine') else None
                    },
                    'data_source': 'MLB_API_authentic'
                }
                
                simplified_predictions.append(prediction)
            
            return simplified_predictions
            
        except Exception as e:
            print(f"Error in simplified team predictions: {e}")
            return []
    
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
                
                # Get player handedness
                handedness = self.data_fetcher.get_player_handedness(player_id)
                
                # Calculate predictions with enhanced rolling averages and matchup history
                player_type = 'pitcher' if is_pitcher else 'batter'
                predictions = self.data_fetcher.calculate_predictions(
                    season_stats, 
                    advanced_stats, 
                    player_type,
                    player_id  # Pass player_id for rolling averages
                )
                
                # Add handedness information
                predictions['handedness'] = handedness
                
                # For pitchers, get last 5 starts and handedness splits
                if is_pitcher:
                    predictions['last_5_starts'] = self.data_fetcher.get_pitcher_last_5_starts(player_id)
                    predictions['vs_handedness'] = self.data_fetcher.get_pitcher_vs_handedness_splits(player_id)
                
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
    
    def _calculate_team_win_probability(self, home_players: List[Dict], away_players: List[Dict], is_home: bool) -> float:
        """Calculate realistic win probability based on team performance"""
        try:
            # Get offensive performance for each team
            home_ops = []
            away_ops = []
            
            for player in home_players:
                if player.get('player_type') == 'batter' and player.get('ops', 0) > 0:
                    home_ops.append(player['ops'])
            
            for player in away_players:
                if player.get('player_type') == 'batter' and player.get('ops', 0) > 0:
                    away_ops.append(player['ops'])
            
            # Calculate team offensive strength
            home_offense = np.mean(home_ops[:9]) if home_ops else 0.700  # League average
            away_offense = np.mean(away_ops[:9]) if away_ops else 0.700
            
            # Get starting pitcher ERAs
            home_pitchers = [p for p in home_players if p.get('player_type') == 'pitcher' and p.get('era', 0) > 0]
            away_pitchers = [p for p in away_players if p.get('player_type') == 'pitcher' and p.get('era', 0) > 0]
            
            home_pitcher_era = home_pitchers[0]['era'] if home_pitchers else 4.20
            away_pitcher_era = away_pitchers[0]['era'] if away_pitchers else 4.20
            
            # Calculate performance differential
            offensive_diff = (home_offense - away_offense) * 0.3  # Scale factor
            pitching_diff = (away_pitcher_era - home_pitcher_era) * 0.05  # Lower ERA is better
            
            # Base probability with home field advantage
            home_field_advantage = 0.54 if is_home else 0.46
            
            # Apply performance adjustments
            win_prob = home_field_advantage + offensive_diff + pitching_diff
            
            # Keep within realistic bounds
            return max(0.25, min(0.75, win_prob))
            
        except Exception as e:
            print(f"Error calculating win probability: {e}")
            return 0.54 if is_home else 0.46  # Default with home field advantage
    
    def test_data_sources(self) -> Dict:
        """Test connectivity to real data sources"""
        return self.data_fetcher.test_api_connectivity()
    
    def get_data_source_status(self) -> str:
        """Get current data source status"""
        test_results = self.test_data_sources()
        
        if test_results['mlb_stats_api'] and test_results['baseball_savant']:
            return "✅ Connected to MLB Stats API and Baseball Savant"
        elif test_results['mlb_stats_api']:
            return "⚠️ Connected to MLB Stats API only (Baseball Savant unavailable) - Using enhanced fallbacks for missing stats"
        elif test_results['baseball_savant']:
            return "⚠️ Connected to Baseball Savant only (MLB Stats API unavailable)"
        else:
            error_msg = "; ".join(test_results.get('error_messages', []))
            return f"❌ No data sources available: {error_msg}"
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional, Tuple
import time

from database_manager import DatabaseManager
from real_data_fetcher import RealMLBDataFetcher

class MLBDataFetcher:
    """
    Fetches MLB data from various sources including MLB Stats API
    """
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.db_manager = DatabaseManager()
        self.real_data_fetcher = RealMLBDataFetcher()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with caching"""
        cache_key = f"{endpoint}_{str(params)}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.cache[cache_key] = (data, time.time())
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    def get_teams(self) -> List[str]:
        """Get list of real MLB teams"""
        real_teams = self.real_data_fetcher.get_real_teams()
        if real_teams:
            return [team['name'] for team in real_teams]
        return self.db_manager.get_teams()
    
    def search_player(self, player_name: str) -> List[Dict]:
        """Search for players by name"""
        try:
            params = {
                'names': player_name,
                'sportId': 1  # MLB
            }
            data = self._make_request("people/search", params)
            return data.get('people', [])
        except Exception:
            return []
    
    def get_player_stats(self, player_name: str, player_type: str = "batter") -> pd.DataFrame:
        """Get player statistics from database"""
        return self.db_manager.get_player_stats(player_name, days=30)
    
    def _process_batting_stats(self, data: Dict) -> pd.DataFrame:
        """Process batting statistics"""
        stats_data = []
        
        for stat_group in data.get('stats', []):
            for split in stat_group.get('splits', []):
                stat = split.get('stat', {})
                
                stats_data.append({
                    'date': datetime.now().date(),
                    'batting_avg': float(stat.get('avg', 0)),
                    'home_runs': int(stat.get('homeRuns', 0)),
                    'rbi': int(stat.get('rbi', 0)),
                    'ops': float(stat.get('ops', 0)),
                    'hits': int(stat.get('hits', 0)),
                    'at_bats': int(stat.get('atBats', 0)),
                    'stolen_bases': int(stat.get('stolenBases', 0)),
                    'runs': int(stat.get('runs', 0))
                })
        
        return pd.DataFrame(stats_data) if stats_data else pd.DataFrame()
    
    def _process_pitching_stats(self, data: Dict) -> pd.DataFrame:
        """Process pitching statistics"""
        stats_data = []
        
        for stat_group in data.get('stats', []):
            for split in stat_group.get('splits', []):
                stat = split.get('stat', {})
                
                stats_data.append({
                    'date': datetime.now().date(),
                    'era': float(stat.get('era', 0)),
                    'whip': float(stat.get('whip', 0)),
                    'strikeouts': int(stat.get('strikeOuts', 0)),
                    'wins': int(stat.get('wins', 0)),
                    'losses': int(stat.get('losses', 0)),
                    'saves': int(stat.get('saves', 0)),
                    'innings_pitched': float(stat.get('inningsPitched', 0)),
                    'hits_allowed': int(stat.get('hits', 0))
                })
        
        return pd.DataFrame(stats_data) if stats_data else pd.DataFrame()
    
    def _generate_sample_player_data(self, player_name: str, player_type: str) -> pd.DataFrame:
        """Generate sample data structure when API is unavailable"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        if player_type == "batter":
            # Generate batting stats with realistic trends
            base_avg = np.random.uniform(0.250, 0.320)
            data = {
                'date': dates,
                'batting_avg': np.random.normal(base_avg, 0.020, len(dates)),
                'home_runs': np.random.poisson(0.5, len(dates)),
                'rbi': np.random.poisson(1.2, len(dates)),
                'ops': np.random.normal(0.800, 0.050, len(dates)),
                'hits': np.random.poisson(1.5, len(dates)),
                'at_bats': np.random.poisson(4.0, len(dates)),
                'stolen_bases': np.random.poisson(0.1, len(dates)),
                'runs': np.random.poisson(0.8, len(dates))
            }
        else:
            # Generate pitching stats with realistic trends
            base_era = np.random.uniform(3.50, 4.50)
            data = {
                'date': dates,
                'era': np.random.normal(base_era, 0.200, len(dates)),
                'whip': np.random.normal(1.200, 0.050, len(dates)),
                'strikeouts': np.random.poisson(8.0, len(dates)),
                'wins': np.random.poisson(0.3, len(dates)),
                'losses': np.random.poisson(0.2, len(dates)),
                'saves': np.random.poisson(0.1, len(dates)),
                'innings_pitched': np.random.normal(6.0, 1.0, len(dates)),
                'hits_allowed': np.random.poisson(6.0, len(dates))
            }
        
        return pd.DataFrame(data)
    
    def get_team_recent_performance(self, team_name: str) -> pd.DataFrame:
        """Get recent team performance data from database"""
        return self.db_manager.get_team_recent_performance(team_name, days=15)
    
    def _generate_sample_team_data(self, team_name: str) -> pd.DataFrame:
        """Generate sample team performance data"""
        players = [
            f"Player_{i}" for i in range(1, 26)  # 25 players
        ]
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=15), end=datetime.now(), freq='D')
        
        data = []
        for player in players:
            for date in dates:
                # Generate both batting and pitching stats
                is_pitcher = np.random.choice([True, False], p=[0.4, 0.6])
                
                if is_pitcher:
                    data.append({
                        'player_name': player,
                        'date': date,
                        'player_type': 'pitcher',
                        'era': np.random.uniform(2.50, 6.00),
                        'whip': np.random.uniform(0.90, 1.80),
                        'strikeouts': np.random.poisson(6),
                        'innings_pitched': np.random.uniform(1.0, 7.0)
                    })
                else:
                    data.append({
                        'player_name': player,
                        'date': date,
                        'player_type': 'batter',
                        'batting_avg': np.random.uniform(0.150, 0.450),
                        'home_runs': np.random.poisson(0.3),
                        'rbi': np.random.poisson(1.0),
                        'ops': np.random.uniform(0.500, 1.200)
                    })
        
        return pd.DataFrame(data)
    
    def get_matchup_history(self, batter_name: str, pitcher_name: str) -> pd.DataFrame:
        """Get historical matchup data between batter and pitcher"""
        try:
            # This would fetch real historical data
            # For now, generate sample matchup data
            return self._generate_sample_matchup_data(batter_name, pitcher_name)
        except Exception:
            return pd.DataFrame()
    
    def _generate_sample_matchup_data(self, batter_name: str, pitcher_name: str) -> pd.DataFrame:
        """Generate sample historical matchup data"""
        num_matchups = np.random.randint(5, 25)
        
        data = []
        for i in range(num_matchups):
            data.append({
                'date': datetime.now().date() - timedelta(days=np.random.randint(1, 365)),
                'batter': batter_name,
                'pitcher': pitcher_name,
                'hit': np.random.choice([0, 1], p=[0.7, 0.3]),
                'home_run': np.random.choice([0, 1], p=[0.95, 0.05]),
                'rbi': np.random.poisson(0.3),
                'strikeout': np.random.choice([0, 1], p=[0.75, 0.25]),
                'walk': np.random.choice([0, 1], p=[0.9, 0.1])
            })
        
        return pd.DataFrame(data)
    
    def get_games_for_date(self, game_date: datetime.date) -> List[Dict]:
        """Get real games for specified date from MLB Stats API"""
        try:
            # Convert date to datetime for API call
            date_obj = datetime.combine(game_date, datetime.min.time())
            
            # Get real data for any date
            real_games = self.real_data_fetcher.get_games_for_date(date_obj)
            if real_games:
                return real_games
        except Exception as e:
            print(f"Error fetching games from API for {game_date}: {e}")
        
        # Fallback to database for other dates
        return self.db_manager.get_games_for_date(game_date)
    
    def _generate_sample_games(self, game_date: datetime.date) -> List[Dict]:
        """Generate sample games for development"""
        teams = self.get_teams()
        num_games = np.random.randint(8, 15)
        
        games = []
        used_teams = set()
        
        for i in range(num_games):
            available_teams = [t for t in teams if t not in used_teams]
            if len(available_teams) < 2:
                break
            
            home_team = np.random.choice(available_teams)
            available_teams.remove(home_team)
            away_team = np.random.choice(available_teams)
            
            used_teams.add(home_team)
            used_teams.add(away_team)
            
            games.append({
                'game_id': f"game_{i}",
                'home_team': home_team,
                'away_team': away_team,
                'game_time': f"{game_date}T19:00:00Z"
            })
        
        return games
    
    def get_strike_zone_data(self, player_name: str, player_type: str) -> pd.DataFrame:
        """Get strike zone data for a player"""
        try:
            # This would fetch real strike zone data
            # For now, generate sample data
            return self._generate_sample_strike_zone_data(player_name, player_type)
        except Exception:
            return pd.DataFrame()
    
    def _generate_sample_strike_zone_data(self, player_name: str, player_type: str) -> pd.DataFrame:
        """Generate sample strike zone data"""
        zones = list(range(1, 10))  # 9 zones
        
        data = []
        for zone in zones:
            if player_type == "batter":
                # Batting performance by zone
                data.append({
                    'zone': zone,
                    'batting_avg': np.random.uniform(0.150, 0.400),
                    'slugging_pct': np.random.uniform(0.250, 0.650),
                    'at_bats': np.random.randint(20, 100),
                    'hits': np.random.randint(5, 35),
                    'home_runs': np.random.randint(0, 8)
                })
            else:
                # Pitching performance by zone
                data.append({
                    'zone': zone,
                    'opponent_avg': np.random.uniform(0.180, 0.350),
                    'whiff_rate': np.random.uniform(0.15, 0.45),
                    'pitches': np.random.randint(50, 200),
                    'strikes': np.random.randint(25, 120),
                    'swings_and_misses': np.random.randint(8, 40)
                })
        
        return pd.DataFrame(data)

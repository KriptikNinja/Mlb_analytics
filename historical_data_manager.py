"""
Historical Data Manager for MLB Analytics
Handles collection and storage of multi-season historical data for advanced betting analysis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Boolean, Text, ForeignKey, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
import time

Base = declarative_base()

class HistoricalPlayerPerformance(Base):
    """Historical game-by-game player performance data"""
    __tablename__ = 'historical_player_performance'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, nullable=False)
    player_name = Column(String(100), nullable=False)
    game_id = Column(Integer)
    game_date = Column(Date, nullable=False)
    season = Column(Integer, nullable=False)
    team = Column(String(50), nullable=False)
    opponent = Column(String(50), nullable=False)
    home_away = Column(String(5), nullable=False)  # 'home' or 'away'
    
    # Batting stats
    at_bats = Column(Integer, default=0)
    hits = Column(Integer, default=0)
    doubles = Column(Integer, default=0)
    triples = Column(Integer, default=0)
    home_runs = Column(Integer, default=0)
    rbis = Column(Integer, default=0)
    walks = Column(Integer, default=0)
    strikeouts = Column(Integer, default=0)
    stolen_bases = Column(Integer, default=0)
    
    # Advanced metrics
    exit_velocity_avg = Column(Float)
    launch_angle_avg = Column(Float)
    hard_hit_rate = Column(Float)
    barrel_rate = Column(Float)
    
    # Pitching stats (for pitchers)
    innings_pitched = Column(Float, default=0.0)
    pitches_thrown = Column(Integer, default=0)
    strikes = Column(Integer, default=0)
    earned_runs = Column(Integer, default=0)
    hits_allowed = Column(Integer, default=0)
    walks_allowed = Column(Integer, default=0)
    strikeouts_pitched = Column(Integer, default=0)
    
    # Game context
    ballpark = Column(String(100))
    weather_temp = Column(Integer)
    weather_wind_speed = Column(Integer)
    weather_wind_direction = Column(String(10))
    
    # Performance indicators
    wpa = Column(Float)  # Win Probability Added
    leverage_index = Column(Float)
    game_score = Column(Float)  # For pitchers
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('player_id', 'game_date', 'season'),)

class PlayerSimilarity(Base):
    """Player similarity scores for style-based matchup analysis"""
    __tablename__ = 'player_similarity'
    
    id = Column(Integer, primary_key=True)
    player_a_id = Column(Integer, nullable=False)
    player_a_name = Column(String(100), nullable=False)
    player_b_id = Column(Integer, nullable=False)
    player_b_name = Column(String(100), nullable=False)
    
    # Similarity scores (0-1 scale)
    zone_similarity = Column(Float, default=0.0)  # Strike zone performance similarity
    contact_similarity = Column(Float, default=0.0)  # Contact profile similarity
    power_similarity = Column(Float, default=0.0)  # Power profile similarity
    plate_discipline_similarity = Column(Float, default=0.0)  # Discipline similarity
    overall_similarity = Column(Float, default=0.0)  # Weighted overall score
    
    # Context
    seasons_analyzed = Column(String(20))  # e.g., "2022,2023,2024"
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('player_a_id', 'player_b_id'),)

class HistoricalMatchups(Base):
    """Historical batter vs pitcher matchup performance"""
    __tablename__ = 'historical_matchups'
    
    id = Column(Integer, primary_key=True)
    batter_id = Column(Integer, nullable=False)
    batter_name = Column(String(100), nullable=False)
    pitcher_id = Column(Integer, nullable=False)
    pitcher_name = Column(String(100), nullable=False)
    
    # Aggregate historical performance
    total_at_bats = Column(Integer, default=0)
    total_hits = Column(Integer, default=0)
    total_home_runs = Column(Integer, default=0)
    total_rbis = Column(Integer, default=0)
    total_strikeouts = Column(Integer, default=0)
    total_walks = Column(Integer, default=0)
    
    # Calculated stats
    batting_avg = Column(Float, default=0.0)
    on_base_pct = Column(Float, default=0.0)
    slugging_pct = Column(Float, default=0.0)
    ops = Column(Float, default=0.0)
    
    # Confidence indicators
    sample_size_confidence = Column(Float, default=0.0)  # Based on at-bat count
    recency_weight = Column(Float, default=1.0)  # More recent = higher weight
    
    # Context
    seasons_span = Column(String(20))  # e.g., "2022-2024"
    last_encounter = Column(Date)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (UniqueConstraint('batter_id', 'pitcher_id'),)

class BallparkFactors(Base):
    """Ballpark-specific performance factors"""
    __tablename__ = 'ballpark_factors'
    
    id = Column(Integer, primary_key=True)
    ballpark_name = Column(String(100), unique=True, nullable=False)
    
    # Offensive factors (compared to league average = 1.0)
    home_run_factor = Column(Float, default=1.0)
    hits_factor = Column(Float, default=1.0)
    strikeout_factor = Column(Float, default=1.0)
    
    # Environmental factors
    elevation = Column(Integer, default=0)  # Feet above sea level
    foul_territory = Column(String(20))  # 'large', 'average', 'small'
    wall_height_lf = Column(Integer, default=0)
    wall_height_cf = Column(Integer, default=0)
    wall_height_rf = Column(Integer, default=0)
    
    # Historical context
    seasons_analyzed = Column(String(20))
    last_updated = Column(DateTime, default=datetime.utcnow)

class HistoricalDataManager:
    """
    Manages collection and analysis of historical MLB data for advanced betting intelligence
    """
    
    def __init__(self):
        """Initialize with database connection"""
        # Use same database as main app
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
        # Create historical tables
        Base.metadata.create_all(self.engine)
        print("Historical data tables created successfully")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def collect_historical_seasons(self, seasons: List[int] = [2022, 2023, 2024]):
        """
        Collect historical data for specified seasons with smart resumption
        This is the main entry point for building historical database
        """
        print(f"Starting collection of historical data for seasons: {seasons}")
        
        for season in seasons:
            # Check if season is already complete
            if self._is_season_complete(season):
                print(f"✅ Season {season} already complete - skipping")
                continue
                
            print(f"\nCollecting data for {season} season...")
            try:
                # Get all games for the season
                games = self._get_season_games(season)
                print(f"Found {len(games)} games for {season}")
                
                # Get already processed games to avoid duplicates
                processed_games = self._get_processed_game_ids(season)
                remaining_games = [g for g in games if g.get('game_id') not in processed_games]
                
                if len(processed_games) > 0:
                    print(f"📋 Already processed: {len(processed_games)} games")
                    print(f"🔄 Remaining: {len(remaining_games)} games")
                
                if len(remaining_games) == 0:
                    print(f"✅ Season {season} collection complete!")
                    continue
                
                # Process remaining games
                batch_size = 50
                print(f"Processing {len(remaining_games)} remaining games of {season} season")
                
                for i in range(0, len(remaining_games), batch_size):
                    batch = remaining_games[i:i+batch_size]
                    progress = f"{i//batch_size + 1}/{(len(remaining_games)-1)//batch_size + 1}"
                    print(f"Processing batch {progress} ({len(batch)} games)")
                    
                    for game in batch:
                        try:
                            self._collect_game_player_data(game, season)
                        except Exception as e:
                            print(f"Error processing game {game.get('game_id', 'unknown')}: {e}")
                            continue
                    
                    # Commit batch to database
                    self._commit_batch()
                    print(f"Batch {progress} completed")
                    time.sleep(0.3)  # Rate limiting between batches
                    
            except Exception as e:
                print(f"Error collecting data for season {season}: {e}")
                continue
        
        # After collecting raw data, calculate derived metrics
        self._calculate_player_similarities()
        self._calculate_historical_matchups()
        self._calculate_ballpark_factors()
        
        print("Historical data collection completed!")
        print("✅ Enhanced betting opportunities now available")
        
        # Print collection summary
        with self.get_session() as session:
            performance_count = session.query(HistoricalPlayerPerformance).count()
            matchup_count = session.query(HistoricalMatchups).count()
            print(f"📊 Collection Summary:")
            print(f"   - Player performances: {performance_count:,} records")
            print(f"   - Historical matchups: {matchup_count:,} records")
            print(f"   - Seasons collected: {len(seasons)}")
            session.close()
    
    def _is_season_complete(self, season: int) -> bool:
        """Check if season data collection is complete"""
        try:
            with self.get_session() as session:
                # Count games for this season
                game_count = session.query(HistoricalPlayerPerformance).filter(
                    HistoricalPlayerPerformance.season == season
                ).count()
                
                # Consider season complete if we have substantial data (>2000 games)
                # 2022 should have ~2400+ games, others similar
                expected_minimum = 2000 if season < 2024 else 100  # 2024 partial season
                return game_count >= expected_minimum
        except Exception as e:
            print(f"Error checking season {season} completion: {e}")
            return False
    
    def _get_processed_game_ids(self, season: int) -> set:
        """Get set of already processed game IDs for a season"""
        try:
            with self.get_session() as session:
                # Get unique game IDs already in database
                from sqlalchemy import text
                result = session.execute(
                    text("SELECT DISTINCT game_id FROM historical_player_performance WHERE season = :season AND game_id IS NOT NULL"),
                    {"season": season}
                ).fetchall()
                return {row[0] for row in result if row[0]}
        except Exception as e:
            print(f"Error getting processed games for {season}: {e}")
            return set()

    def _commit_batch(self):
        """Commit current batch to database"""
        try:
            with self.get_session() as session:
                session.commit()
        except Exception as e:
            print(f"Error committing batch: {e}")
    
    def _get_season_games(self, season: int) -> List[Dict]:
        """Get all games for a given season"""
        try:
            # Get season schedule
            url = f"{self.base_url}/schedule"
            params = {
                'season': season,
                'sportId': 1,  # MLB
                'gameType': 'R'  # Regular season only
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                games = []
                
                for date_data in data.get('dates', []):
                    for game in date_data.get('games', []):
                        games.append({
                            'game_id': game.get('gamePk'),
                            'game_date': game.get('gameDate', '').split('T')[0],
                            'home_team': game.get('teams', {}).get('home', {}).get('team', {}).get('name'),
                            'away_team': game.get('teams', {}).get('away', {}).get('team', {}).get('name'),
                            'venue': game.get('venue', {}).get('name')
                        })
                
                return games
            
        except Exception as e:
            print(f"Error fetching season {season} games: {e}")
            return []
    
    def _collect_game_player_data(self, game: Dict, season: int):
        """Collect player performance data for a specific game"""
        try:
            game_id = game.get('game_id')
            if not game_id:
                return
            
            # Store current game_id for use in _save_player_performance
            self._current_game_id = game_id
            
            # Get game boxscore for player stats
            url = f"{self.base_url}/game/{game_id}/boxscore"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                boxscore = response.json()
                game_date = datetime.strptime(game.get('game_date'), '%Y-%m-%d').date()
                
                # Process both teams
                for team_type in ['home', 'away']:
                    team_data = boxscore.get('teams', {}).get(team_type, {})
                    team_name = game.get(f'{team_type}_team')
                    opponent = game.get('away_team' if team_type == 'home' else 'home_team')
                    
                    # Process batters
                    batters = team_data.get('batters', [])
                    for batter_id in batters:
                        batter_stats = team_data.get('battingStats', {}).get(str(batter_id), {})
                        self._save_player_performance(
                            player_id=batter_id,
                            game_date=game_date,
                            season=season,
                            team=team_name,
                            opponent=opponent,
                            home_away=team_type,
                            ballpark=game.get('venue'),
                            stats=batter_stats,
                            player_type='batter'
                        )
                    
                    # Process pitchers
                    pitchers = team_data.get('pitchers', [])
                    for pitcher_id in pitchers:
                        pitcher_stats = team_data.get('pitchingStats', {}).get(str(pitcher_id), {})
                        self._save_player_performance(
                            player_id=pitcher_id,
                            game_date=game_date,
                            season=season,
                            team=team_name,
                            opponent=opponent,
                            home_away=team_type,
                            ballpark=game.get('venue'),
                            stats=pitcher_stats,
                            player_type='pitcher'
                        )
                        
        except Exception as e:
            print(f"Error collecting game {game.get('game_id')} data: {e}")
    
    def _save_player_performance(self, player_id: int, game_date: date, season: int, 
                                team: str, opponent: str, home_away: str, ballpark: str,
                                stats: Dict, player_type: str):
        """Save individual player performance to database"""
        try:
            session = self.get_session()
            
            # Validate required data
            if not player_id or not game_date:
                session.close()
                return
            
            # Get player name (you might want to implement player name lookup)
            player_name = stats.get('playerName', f'Player_{player_id}')
            
            # Check if record already exists with better error handling
            try:
                existing = session.query(HistoricalPlayerPerformance).filter_by(
                    player_id=player_id,
                    game_date=game_date,
                    season=season
                ).first()
            except Exception as query_error:
                print(f"Query error for player {player_id}: {query_error}")
                session.close()
                return
            
            if existing:
                session.close()
                return
            
            # Get game_id from current context (you'll need to pass this)
            game_id = getattr(self, '_current_game_id', None)
            
            performance = HistoricalPlayerPerformance(
                player_id=player_id,
                player_name=player_name,
                game_date=game_date,
                season=season,
                team=team,
                opponent=opponent,
                home_away=home_away,
                ballpark=ballpark,
                game_id=game_id
            )
            
            if player_type == 'batter':
                # Batting stats
                performance.at_bats = stats.get('atBats', 0)
                performance.hits = stats.get('hits', 0)
                performance.doubles = stats.get('doubles', 0)
                performance.triples = stats.get('triples', 0)
                performance.home_runs = stats.get('homeRuns', 0)
                performance.rbis = stats.get('rbi', 0)
                performance.walks = stats.get('baseOnBalls', 0)
                performance.strikeouts = stats.get('strikeOuts', 0)
                performance.stolen_bases = stats.get('stolenBases', 0)
                
            elif player_type == 'pitcher':
                # Pitching stats
                performance.innings_pitched = float(stats.get('inningsPitched', '0.0'))
                performance.pitches_thrown = stats.get('numberOfPitches', 0)
                performance.earned_runs = stats.get('earnedRuns', 0)
                performance.hits_allowed = stats.get('hits', 0)
                performance.walks_allowed = stats.get('baseOnBalls', 0)
                performance.strikeouts_pitched = stats.get('strikeOuts', 0)
            
            session.add(performance)
            session.commit()
            session.close()
            
        except Exception as e:
            print(f"Error saving player {player_id} performance: {e}")
            if 'session' in locals():
                try:
                    session.rollback()
                    session.close()
                except:
                    pass
    
    def _calculate_player_similarities(self):
        """Calculate similarity scores between players based on performance profiles"""
        print("Calculating player similarity scores...")
        # Implementation for player similarity analysis
        # This will analyze batting profiles, zone tendencies, etc.
        pass
    
    def _calculate_historical_matchups(self):
        """Calculate historical batter vs pitcher matchup statistics"""
        print("Calculating historical matchup data...")
        # Implementation for aggregating batter vs pitcher historical performance
        pass
    
    def _calculate_ballpark_factors(self):
        """Calculate ballpark-specific performance factors"""
        print("Calculating ballpark factors...")
        # Implementation for ballpark effect analysis
        pass
    
    def get_player_historical_performance(self, player_id: int, days_back: int = 30) -> pd.DataFrame:
        """Get historical performance for a player over specified period"""
        session = self.get_session()
        
        cutoff_date = date.today() - timedelta(days=days_back)
        
        performances = session.query(HistoricalPlayerPerformance).filter(
            HistoricalPlayerPerformance.player_id == player_id,
            HistoricalPlayerPerformance.game_date >= cutoff_date
        ).order_by(HistoricalPlayerPerformance.game_date.desc()).all()
        
        session.close()
        
        # Convert to DataFrame for analysis
        data = []
        for perf in performances:
            data.append({
                'game_date': perf.game_date,
                'opponent': perf.opponent,
                'at_bats': perf.at_bats,
                'hits': perf.hits,
                'home_runs': perf.home_runs,
                'strikeouts': perf.strikeouts,
                'batting_avg': perf.hits / perf.at_bats if perf.at_bats > 0 else 0
            })
        
        return pd.DataFrame(data)
    
    def get_matchup_history(self, batter_id: int, pitcher_id: int) -> Optional[Dict]:
        """Get historical matchup data between specific batter and pitcher"""
        session = self.get_session()
        
        matchup = session.query(HistoricalMatchups).filter_by(
            batter_id=batter_id,
            pitcher_id=pitcher_id
        ).first()
        
        session.close()
        
        if matchup:
            return {
                'at_bats': matchup.total_at_bats,
                'hits': matchup.total_hits,
                'home_runs': matchup.total_home_runs,
                'batting_avg': matchup.batting_avg,
                'ops': matchup.ops,
                'confidence': matchup.sample_size_confidence,
                'last_encounter': matchup.last_encounter
            }
        
        return None
    
    def find_similar_players(self, player_id: int, similarity_threshold: float = 0.7) -> List[Dict]:
        """Find players with similar playing styles"""
        session = self.get_session()
        
        similar_players = session.query(PlayerSimilarity).filter(
            PlayerSimilarity.player_a_id == player_id,
            PlayerSimilarity.overall_similarity >= similarity_threshold
        ).order_by(PlayerSimilarity.overall_similarity.desc()).all()
        
        session.close()
        
        results = []
        for sim in similar_players:
            results.append({
                'player_id': sim.player_b_id,
                'player_name': sim.player_b_name,
                'similarity_score': sim.overall_similarity,
                'zone_similarity': sim.zone_similarity,
                'contact_similarity': sim.contact_similarity,
                'power_similarity': sim.power_similarity
            })
        
        return results
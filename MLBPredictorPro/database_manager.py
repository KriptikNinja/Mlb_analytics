"""
Database manager for MLB analytics application.
Handles all database operations including data storage, retrieval, and caching.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, 
    Boolean, Text, Date, ForeignKey, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import insert
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Team(Base):
    """MLB Teams table"""
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    abbreviation = Column(String(5), nullable=False)
    division = Column(String(50))
    league = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)

class Player(Base):
    """MLB Players table"""
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'))
    position = Column(String(20))
    player_type = Column(String(20))  # 'batter' or 'pitcher'
    jersey_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    team = relationship("Team", back_populates="players")

class Game(Base):
    """MLB Games table"""
    __tablename__ = 'games'
    
    id = Column(Integer, primary_key=True)
    game_date = Column(Date, nullable=False)
    home_team_id = Column(Integer, ForeignKey('teams.id'))
    away_team_id = Column(Integer, ForeignKey('teams.id'))
    home_score = Column(Integer)
    away_score = Column(Integer)
    status = Column(String(20), default='scheduled')  # scheduled, in_progress, completed
    game_time = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

class PlayerStats(Base):
    """Player statistics table"""
    __tablename__ = 'player_stats'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    game_id = Column(Integer, ForeignKey('games.id'))
    stat_date = Column(Date, nullable=False)
    
    # Batting stats
    at_bats = Column(Integer, default=0)
    hits = Column(Integer, default=0)
    home_runs = Column(Integer, default=0)
    rbis = Column(Integer, default=0)
    batting_average = Column(Float, default=0.0)
    on_base_percentage = Column(Float, default=0.0)
    slugging_percentage = Column(Float, default=0.0)
    
    # Pitching stats
    innings_pitched = Column(Float, default=0.0)
    strikeouts = Column(Integer, default=0)
    walks = Column(Integer, default=0)
    earned_runs = Column(Integer, default=0)
    era = Column(Float, default=0.0)
    whip = Column(Float, default=0.0)
    
    # Performance indicators
    hot_streak = Column(Boolean, default=False)
    cold_streak = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    player = relationship("Player")
    game = relationship("Game")

class Prediction(Base):
    """ML predictions table"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    game_id = Column(Integer, ForeignKey('games.id'))
    prediction_date = Column(Date, nullable=False)
    
    # Prediction values
    predicted_batting_avg = Column(Float)
    predicted_home_runs = Column(Float)
    predicted_strikeouts = Column(Float)
    predicted_era = Column(Float)
    
    # Probabilities
    hit_probability = Column(Float)
    home_run_probability = Column(Float)
    strikeout_probability = Column(Float)
    quality_start_probability = Column(Float)
    
    # Confidence scores
    confidence_score = Column(Float)
    model_version = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    player = relationship("Player")
    game = relationship("Game")

class StrikeZoneData(Base):
    """Strike zone analysis data"""
    __tablename__ = 'strike_zone_data'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    zone_number = Column(Integer, nullable=False)  # 1-9 zones
    performance_value = Column(Float, nullable=False)
    data_type = Column(String(20), nullable=False)  # 'batter' or 'pitcher'
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    player = relationship("Player")
    
    __table_args__ = (UniqueConstraint('player_id', 'zone_number', 'data_type'),)

# Add relationship back-references
Team.players = relationship("Player", back_populates="team")

class DatabaseManager:
    """
    Manages all database operations for the MLB analytics application
    """
    
    def __init__(self):
        """Initialize database connection and create tables"""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        # Create engine with SSL settings for better connection handling
        try:
            self.engine = create_engine(
                self.database_url,
                pool_size=3,
                max_overflow=5,
                pool_pre_ping=True,
                pool_recycle=300,
                connect_args={
                    "sslmode": "prefer",
                    "connect_timeout": 10
                }
            )
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create all tables
            self.create_tables()
            
            # Initialize with sample data if empty
            self._initialize_sample_data()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Create a minimal fallback for development
            self.engine = None
            self.SessionLocal = None
    
    def create_tables(self):
        """Create all database tables"""
        try:
            if self.engine:
                Base.metadata.create_all(bind=self.engine)
                logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            # Don't raise error to allow app to continue with fallback data
    
    def get_session(self):
        """Get database session"""
        if self.SessionLocal:
            return self.SessionLocal()
        return None
    
    def _initialize_sample_data(self):
        """Initialize database with sample MLB data if empty"""
        session = self.get_session()
        if not session:
            logger.warning("No database session available - skipping data initialization")
            return
            
        try:
            # Check if teams exist
            team_count = session.query(Team).count()
            if team_count == 0:
                self._populate_teams(session)
                session.commit()
                
                self._populate_players(session) 
                session.commit()
                
                self._populate_games(session)
                session.commit()
                
                self._populate_sample_stats(session)
                session.commit()
                
                logger.info("Sample data initialized")
        except Exception as e:
            session.rollback()
            logger.error(f"Error initializing sample data: {e}")
        finally:
            session.close()
    
    def _populate_teams(self, session):
        """Populate teams table with MLB teams"""
        mlb_teams = [
            ("New York Yankees", "NYY", "AL East", "American League"),
            ("Boston Red Sox", "BOS", "AL East", "American League"),
            ("Tampa Bay Rays", "TB", "AL East", "American League"),
            ("Toronto Blue Jays", "TOR", "AL East", "American League"),
            ("Baltimore Orioles", "BAL", "AL East", "American League"),
            ("Houston Astros", "HOU", "AL West", "American League"),
            ("Seattle Mariners", "SEA", "AL West", "American League"),
            ("Los Angeles Angels", "LAA", "AL West", "American League"),
            ("Oakland Athletics", "OAK", "AL West", "American League"),
            ("Texas Rangers", "TEX", "AL West", "American League"),
            ("Chicago White Sox", "CWS", "AL Central", "American League"),
            ("Cleveland Guardians", "CLE", "AL Central", "American League"),
            ("Detroit Tigers", "DET", "AL Central", "American League"),
            ("Kansas City Royals", "KC", "AL Central", "American League"),
            ("Minnesota Twins", "MIN", "AL Central", "American League"),
            ("Los Angeles Dodgers", "LAD", "NL West", "National League"),
            ("San Diego Padres", "SD", "NL West", "National League"),
            ("San Francisco Giants", "SF", "NL West", "National League"),
            ("Arizona Diamondbacks", "ARI", "NL West", "National League"),
            ("Colorado Rockies", "COL", "NL West", "National League"),
            ("Atlanta Braves", "ATL", "NL East", "National League"),
            ("New York Mets", "NYM", "NL East", "National League"),
            ("Philadelphia Phillies", "PHI", "NL East", "National League"),
            ("Miami Marlins", "MIA", "NL East", "National League"),
            ("Washington Nationals", "WSH", "NL East", "National League"),
            ("Milwaukee Brewers", "MIL", "NL Central", "National League"),
            ("Chicago Cubs", "CHC", "NL Central", "National League"),
            ("St. Louis Cardinals", "STL", "NL Central", "National League"),
            ("Cincinnati Reds", "CIN", "NL Central", "National League"),
            ("Pittsburgh Pirates", "PIT", "NL Central", "National League")
        ]
        
        for name, abbr, division, league in mlb_teams:
            team = Team(name=name, abbreviation=abbr, division=division, league=league)
            session.add(team)
    
    def _populate_players(self, session):
        """Populate players table with sample players"""
        # Get teams from database
        teams = session.query(Team).all()
        
        # Sample player names for each team
        batter_names = [
            "Mike Trout", "Aaron Judge", "Mookie Betts", "Ronald Acuna Jr.", "Juan Soto",
            "Vladimir Guerrero Jr.", "Freddie Freeman", "Jose Altuve", "Fernando Tatis Jr.",
            "Bryce Harper", "Manny Machado", "Pete Alonso", "Austin Riley", "Bo Bichette",
            "Gleyber Torres", "Xander Bogaerts", "Matt Olson", "Kyle Tucker", "Rafael Devers"
        ]
        
        pitcher_names = [
            "Jacob deGrom", "Gerrit Cole", "Shane Bieber", "Walker Buehler", "Tyler Glasnow",
            "Corbin Burnes", "Max Scherzer", "Sandy Alcantara", "Kevin Gausman", "Dylan Cease",
            "Framber Valdez", "Logan Webb", "Jose Berrios", "Pablo Lopez", "Chris Bassitt"
        ]
        
        for team in teams:
            # Add batters
            for i, name in enumerate(batter_names[:9]):  # 9 batters per team
                player = Player(
                    name=f"{name}_{team.abbreviation}",
                    team_id=team.id,
                    position="OF" if i < 3 else "IF",
                    player_type="batter",
                    jersey_number=i + 1
                )
                session.add(player)
            
            # Add pitchers
            for i, name in enumerate(pitcher_names[:5]):  # 5 pitchers per team
                player = Player(
                    name=f"{name}_{team.abbreviation}",
                    team_id=team.id,
                    position="SP" if i < 2 else "RP",
                    player_type="pitcher",
                    jersey_number=i + 10
                )
                session.add(player)
    
    def _populate_games(self, session):
        """Populate games table with today's games"""
        teams = session.query(Team).all()
        today = datetime.now().date()
        
        # Create sample games for today
        game_matchups = [
            (teams[0], teams[1]),   # Yankees vs Red Sox
            (teams[2], teams[3]),   # Rays vs Blue Jays
            (teams[4], teams[5]),   # Orioles vs Astros
            (teams[6], teams[7]),   # Mariners vs Angels
            (teams[15], teams[16]), # Dodgers vs Padres
            (teams[20], teams[21]), # Braves vs Mets
        ]
        
        for i, (home_team, away_team) in enumerate(game_matchups):
            game = Game(
                game_date=today,
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                status='scheduled',
                game_time=f"{13 + i}:00"  # Staggered start times
            )
            session.add(game)
    
    def _populate_sample_stats(self, session):
        """Populate sample player statistics"""
        players = session.query(Player).all()
        today = datetime.now().date()
        
        for player in players:
            # Generate last 30 days of stats
            for days_back in range(30):
                stat_date = today - timedelta(days=days_back)
                
                if player.player_type == "batter":
                    stats = PlayerStats(
                        player_id=player.id,
                        stat_date=stat_date,
                        at_bats=np.random.randint(3, 6),
                        hits=np.random.randint(0, 4),
                        home_runs=np.random.randint(0, 2),
                        rbis=np.random.randint(0, 4),
                        batting_average=np.random.uniform(0.200, 0.350),
                        on_base_percentage=np.random.uniform(0.250, 0.450),
                        slugging_percentage=np.random.uniform(0.300, 0.600)
                    )
                else:  # pitcher
                    stats = PlayerStats(
                        player_id=player.id,
                        stat_date=stat_date,
                        innings_pitched=np.random.uniform(1.0, 7.0),
                        strikeouts=np.random.randint(2, 12),
                        walks=np.random.randint(0, 4),
                        earned_runs=np.random.randint(0, 5),
                        era=np.random.uniform(2.00, 5.50),
                        whip=np.random.uniform(0.80, 1.60)
                    )
                
                session.add(stats)
    
    def get_teams(self) -> List[str]:
        """Get list of all MLB teams"""
        session = self.get_session()
        try:
            teams = session.query(Team).all()
            return [team.name for team in teams]
        finally:
            session.close()
    
    def get_team_players(self, team_name: str, player_type: str = None) -> pd.DataFrame:
        """Get players for a specific team"""
        session = self.get_session()
        try:
            query = session.query(Player).join(Team).filter(Team.name == team_name)
            if player_type:
                query = query.filter(Player.player_type == player_type)
            
            players = query.all()
            
            data = []
            for player in players:
                data.append({
                    'player_id': player.id,
                    'player_name': player.name,
                    'position': player.position,
                    'player_type': player.player_type,
                    'jersey_number': player.jersey_number
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_player_stats(self, player_name: str, days: int = 30) -> pd.DataFrame:
        """Get recent player statistics"""
        session = self.get_session()
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            stats = session.query(PlayerStats).join(Player).filter(
                Player.name.like(f"%{player_name}%"),
                PlayerStats.stat_date >= cutoff_date
            ).order_by(PlayerStats.stat_date.desc()).all()
            
            data = []
            for stat in stats:
                data.append({
                    'date': stat.stat_date,
                    'player_name': stat.player.name,
                    'player_type': stat.player.player_type,
                    'at_bats': stat.at_bats,
                    'hits': stat.hits,
                    'home_runs': stat.home_runs,
                    'rbis': stat.rbis,
                    'batting_average': stat.batting_average,
                    'innings_pitched': stat.innings_pitched,
                    'strikeouts': stat.strikeouts,
                    'era': stat.era,
                    'whip': stat.whip
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_games_for_date(self, game_date: datetime.date) -> List[Dict]:
        """Get games for a specific date"""
        session = self.get_session()
        try:
            games = session.query(Game).filter(Game.game_date == game_date).all()
            
            result = []
            for game in games:
                result.append({
                    'game_id': game.id,
                    'home_team': game.home_team.name,
                    'away_team': game.away_team.name,
                    'game_time': game.game_time,
                    'status': game.status
                })
            
            return result
        finally:
            session.close()
    
    def save_predictions(self, predictions: List[Dict]):
        """Save ML predictions to database"""
        session = self.get_session()
        try:
            for pred in predictions:
                prediction = Prediction(**pred)
                session.add(prediction)
            session.commit()
            logger.info(f"Saved {len(predictions)} predictions")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving predictions: {e}")
        finally:
            session.close()
    
    def get_team_recent_performance(self, team_name: str, days: int = 15) -> pd.DataFrame:
        """Get recent team performance data"""
        session = self.get_session()
        try:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            
            # Get all players from the team with their recent stats
            stats = session.query(PlayerStats, Player, Team).join(
                Player, PlayerStats.player_id == Player.id
            ).join(
                Team, Player.team_id == Team.id
            ).filter(
                Team.name == team_name,
                PlayerStats.stat_date >= cutoff_date
            ).all()
            
            data = []
            for stat, player, team in stats:
                data.append({
                    'player_name': player.name,
                    'player_type': player.player_type,
                    'date': stat.stat_date,
                    'batting_average': stat.batting_average,
                    'home_runs': stat.home_runs,
                    'rbis': stat.rbis,
                    'strikeouts': stat.strikeouts,
                    'era': stat.era,
                    'whip': stat.whip
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def update_strike_zone_data(self, player_name: str, zone_data: np.ndarray, data_type: str):
        """Update strike zone data for a player"""
        session = self.get_session()
        try:
            # Find player
            player = session.query(Player).filter(Player.name.like(f"%{player_name}%")).first()
            if not player:
                logger.warning(f"Player {player_name} not found")
                return
            
            # Update zone data
            for zone_num in range(1, 10):  # 9 zones
                zone_value = float(zone_data.flatten()[zone_num - 1])
                
                # Use upsert (insert or update)
                stmt = insert(StrikeZoneData).values(
                    player_id=player.id,
                    zone_number=zone_num,
                    performance_value=zone_value,
                    data_type=data_type,
                    last_updated=datetime.utcnow()
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=['player_id', 'zone_number', 'data_type'],
                    set_={
                        'performance_value': stmt.excluded.performance_value,
                        'last_updated': stmt.excluded.last_updated
                    }
                )
                session.execute(stmt)
            
            session.commit()
            logger.info(f"Updated strike zone data for {player_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating strike zone data: {e}")
        finally:
            session.close()
    
    def get_strike_zone_data(self, player_name: str, data_type: str) -> Optional[np.ndarray]:
        """Get strike zone data for a player"""
        session = self.get_session()
        try:
            player = session.query(Player).filter(Player.name.like(f"%{player_name}%")).first()
            if not player:
                return None
            
            zone_data = session.query(StrikeZoneData).filter(
                StrikeZoneData.player_id == player.id,
                StrikeZoneData.data_type == data_type
            ).order_by(StrikeZoneData.zone_number).all()
            
            if len(zone_data) != 9:
                return None
            
            # Convert to 3x3 array
            values = [z.performance_value for z in zone_data]
            return np.array(values).reshape(3, 3)
        finally:
            session.close()
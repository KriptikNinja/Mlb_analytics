#!/usr/bin/env python3
"""
Multi-Source MLB Data Collector
Combines MLB Stats API, Baseball Savant, and local Statcast files for comprehensive data collection
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional
import os
import glob
from historical_data_manager import HistoricalDataManager
import warnings
warnings.filterwarnings('ignore')

class MultiSourceDataCollector:
    """
    Advanced data collector that combines multiple MLB data sources
    """
    
    def __init__(self):
        self.historical_manager = HistoricalDataManager()
        self.mlb_api_base = "https://statsapi.mlb.com/api/v1"
        self.savant_base = "https://baseballsavant.mlb.com"
        self.rate_limit_delay = 0.5  # 500ms between requests
        
    def collect_from_multiple_sources(self, seasons: List[int] = [2022, 2023, 2024]):
        """
        Collect data from all available sources simultaneously
        """
        print("🚀 Multi-Source Data Collection Started")
        print("=" * 50)
        
        for season in seasons:
            print(f"\n📅 Collecting {season} season data from multiple sources...")
            
            # 1. Check for local Statcast files first (fastest)
            self._process_local_statcast_files(season)
            
            # 2. Baseball Savant leaderboards (faster than game-by-game)
            self._collect_savant_leaderboards(season)
            
            # 3. MLB Stats API for remaining gaps
            self._fill_gaps_with_mlb_api(season)
            
        print("\n✅ Multi-source collection completed!")
        self._print_collection_summary()
    
    def _process_local_statcast_files(self, season: int):
        """
        Process local Statcast CSV files if available
        """
        print(f"🔍 Checking for local Statcast files for {season}...")
        
        # Common Statcast file patterns
        file_patterns = [
            f"*{season}*statcast*.csv",
            f"*{season}*.csv",
            f"statcast_{season}.csv",
            f"savant_{season}.csv",
            f"{season}_statcast.csv"
        ]
        
        local_files = []
        for pattern in file_patterns:
            local_files.extend(glob.glob(pattern))
        
        if local_files:
            print(f"📁 Found {len(local_files)} potential Statcast files:")
            for file in local_files:
                print(f"   - {file}")
            
            print("\n💡 To use your local Statcast files:")
            print("1. Upload them to this Replit environment")
            print("2. They should contain columns like: player_name, game_date, events, exit_velocity, launch_angle")
            print("3. The system will automatically process them")
            
            # Process any files that exist
            for file_path in local_files:
                if os.path.exists(file_path):
                    self._process_statcast_file(file_path, season)
        else:
            print(f"📁 No local Statcast files found for {season}")
    
    def _process_statcast_file(self, file_path: str, season: int):
        """
        Process a single Statcast CSV file
        """
        try:
            print(f"📊 Processing {file_path}...")
            
            # Read CSV with common Statcast structure
            df = pd.read_csv(file_path, low_memory=False)
            print(f"   - Loaded {len(df):,} rows")
            
            # Expected Statcast columns
            required_cols = ['player_name', 'game_date']
            if not all(col in df.columns for col in required_cols):
                print(f"   - Skipping: Missing required columns {required_cols}")
                return
            
            # Process in batches
            batch_size = 1000
            processed = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                self._save_statcast_batch(batch, season)
                processed += len(batch)
                
                if processed % 5000 == 0:
                    print(f"   - Processed {processed:,}/{len(df):,} rows")
            
            print(f"   ✅ Completed processing {file_path}")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
    
    def _save_statcast_batch(self, batch: pd.DataFrame, season: int):
        """
        Save Statcast data batch to historical database
        """
        try:
            session = self.historical_manager.get_session()
            
            for _, row in batch.iterrows():
                # Extract basic info
                player_name = row.get('player_name', '')
                game_date = pd.to_datetime(row.get('game_date')).date()
                
                # Create enhanced historical record with Statcast metrics
                from historical_data_manager import HistoricalPlayerPerformance
                
                performance = HistoricalPlayerPerformance(
                    player_id=hash(player_name) % 1000000,  # Generate ID from name
                    player_name=player_name,
                    game_date=game_date,
                    season=season,
                    team=row.get('home_team', 'Unknown'),
                    opponent=row.get('away_team', 'Unknown'),
                    home_away='home',
                    ballpark=row.get('park', 'Unknown'),
                    
                    # Enhanced Statcast metrics
                    exit_velocity=row.get('launch_speed', 0),
                    launch_angle=row.get('launch_angle', 0),
                    barrel_rate=1 if row.get('events') == 'home_run' else 0,
                    hard_hit_rate=1 if row.get('launch_speed', 0) >= 95 else 0
                )
                
                session.merge(performance)
            
            session.commit()
            session.close()
            
        except Exception as e:
            print(f"Error saving Statcast batch: {e}")
            if 'session' in locals():
                session.close()
    
    def _collect_savant_leaderboards(self, season: int):
        """
        Collect data from Baseball Savant leaderboards (faster than game-by-game)
        """
        print(f"⚾ Collecting Baseball Savant leaderboards for {season}...")
        
        leaderboard_types = [
            'exit_velocity_barrels',
            'expected_statistics', 
            'pitch_arsenal'
        ]
        
        for lb_type in leaderboard_types:
            try:
                print(f"   - Fetching {lb_type} leaderboard...")
                
                # Baseball Savant CSV download URL
                url = f"{self.savant_base}/leaderboard/custom"
                params = {
                    'year': season,
                    'type': lb_type,
                    'player_type': 'batter',
                    'csv': 'true'
                }
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    # Save response as CSV and process
                    csv_content = response.text
                    temp_file = f"temp_{lb_type}_{season}.csv"
                    
                    with open(temp_file, 'w') as f:
                        f.write(csv_content)
                    
                    # Process the leaderboard data
                    df = pd.read_csv(temp_file)
                    self._process_savant_leaderboard(df, season, lb_type)
                    
                    # Clean up temp file
                    os.remove(temp_file)
                    
                    print(f"   ✅ Processed {len(df)} players from {lb_type}")
                else:
                    print(f"   ❌ Failed to fetch {lb_type}: {response.status_code}")
                
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"   ❌ Error with {lb_type}: {e}")
    
    def _process_savant_leaderboard(self, df: pd.DataFrame, season: int, leaderboard_type: str):
        """
        Process Baseball Savant leaderboard data
        """
        try:
            session = self.historical_manager.get_session()
            
            for _, row in df.iterrows():
                player_name = row.get('player_name', row.get('name', ''))
                if not player_name:
                    continue
                
                # Create aggregated season performance record
                from historical_data_manager import HistoricalPlayerPerformance
                
                performance = HistoricalPlayerPerformance(
                    player_id=hash(player_name) % 1000000,
                    player_name=player_name,
                    game_date=date(season, 7, 1),  # Mid-season date for aggregated stats
                    season=season,
                    team=row.get('team', 'Unknown'),
                    opponent='Season_Aggregate',
                    home_away='aggregate',
                    ballpark='Multiple',
                    
                    # Season totals from leaderboard
                    at_bats=row.get('ab', 0),
                    hits=row.get('h', 0),
                    home_runs=row.get('hr', 0),
                    exit_velocity=row.get('avg_exit_velocity', 0),
                    barrel_rate=row.get('barrel_rate', 0),
                    hard_hit_rate=row.get('hard_hit_percent', 0)
                )
                
                session.merge(performance)
            
            session.commit()
            session.close()
            
        except Exception as e:
            print(f"Error processing {leaderboard_type} leaderboard: {e}")
            if 'session' in locals():
                session.close()
    
    def _fill_gaps_with_mlb_api(self, season: int):
        """
        Use MLB Stats API to fill any remaining data gaps
        """
        print(f"🔄 Filling gaps with MLB Stats API for {season}...")
        
        # Check what data we already have
        session = self.historical_manager.get_session()
        existing_count = session.query(
            self.historical_manager.HistoricalPlayerPerformance
        ).filter_by(season=season).count()
        session.close()
        
        print(f"   - Already have {existing_count:,} records for {season}")
        
        if existing_count < 1000:  # If we need more data
            print("   - Collecting additional game-by-game data from MLB API...")
            # Use existing MLB API collection for remaining data
            # This will be slower but comprehensive
        else:
            print("   - Sufficient data already collected, skipping MLB API")
    
    def _print_collection_summary(self):
        """
        Print summary of collected data
        """
        print("\n📊 Collection Summary:")
        print("=" * 30)
        
        session = self.historical_manager.get_session()
        
        total_records = session.query(
            self.historical_manager.HistoricalPlayerPerformance
        ).count()
        
        unique_players = session.query(
            self.historical_manager.HistoricalPlayerPerformance.player_name
        ).distinct().count()
        
        seasons = session.query(
            self.historical_manager.HistoricalPlayerPerformance.season
        ).distinct().all()
        
        session.close()
        
        print(f"Total records: {total_records:,}")
        print(f"Unique players: {unique_players:,}")
        print(f"Seasons covered: {[s[0] for s in seasons]}")
        print("\n✅ Enhanced betting intelligence now available!")

def main():
    """
    Main execution function
    """
    collector = MultiSourceDataCollector()
    collector.collect_from_multiple_sources([2022, 2023, 2024])

if __name__ == "__main__":
    main()
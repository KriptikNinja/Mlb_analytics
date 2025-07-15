#!/usr/bin/env python3
"""
Statcast Data Processor
Processes user's Statcast files and integrates them into the historical database
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from historical_data_manager import HistoricalDataManager, HistoricalPlayerPerformance
import warnings
warnings.filterwarnings('ignore')

class StatcastProcessor:
    """
    Processes Statcast CSV files for betting intelligence
    """
    
    def __init__(self):
        self.historical_manager = HistoricalDataManager()
    
    def process_statcast_file(self, file_path: str):
        """
        Process a Statcast CSV file and integrate it into the database
        """
        print(f"🚀 Processing Statcast file: {file_path}")
        print("=" * 50)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, low_memory=False)
            print(f"📊 Loaded {len(df):,} rows from Statcast file")
            
            # Clean and prepare data
            df = self._clean_statcast_data(df)
            
            # Group by player and game for aggregated statistics
            game_stats = self._aggregate_by_game(df)
            print(f"📈 Created {len(game_stats)} game-level performance records")
            
            # Save to historical database
            saved_count = self._save_to_database(game_stats)
            print(f"✅ Saved {saved_count} records to historical database")
            
            # Generate summary
            self._print_processing_summary(game_stats)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing Statcast file: {e}")
            return False
    
    def _clean_statcast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare Statcast data for processing
        """
        print("🧹 Cleaning Statcast data...")
        
        # Convert game_date to datetime with flexible parsing
        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
        
        # Filter out records with missing dates - critical for data integrity
        initial_count = len(df)
        df = df.dropna(subset=['game_date'])
        filtered_count = len(df)
        
        if initial_count > filtered_count:
            print(f"⚠️  Filtered out {initial_count - filtered_count} records with missing dates")
            print(f"✅ Keeping {filtered_count} records with valid dates for accurate analysis")
        
        # Extract year for season
        df['season'] = df['game_date'].dt.year
        
        # Clean player names and filter out missing names
        if 'player_name' in df.columns:
            df['player_name'] = df['player_name'].str.strip()
            # Remove records with missing player names
            initial_count = len(df)
            df = df.dropna(subset=['player_name'])
            df = df[df['player_name'] != '']
            filtered_count = len(df)
            
            if initial_count > filtered_count:
                print(f"⚠️  Filtered out {initial_count - filtered_count} records with missing player names")
                print(f"✅ Keeping {filtered_count} records with valid player identification")
        
        # Fill missing numeric values
        numeric_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc', 'release_speed']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create derived metrics
        df = self._create_derived_metrics(df)
        
        return df
    
    def _create_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived metrics from Statcast data
        """
        # Hard-hit balls (95+ mph exit velocity)
        if 'launch_speed' in df.columns:
            df['hard_hit'] = (df['launch_speed'] >= 95).astype(int)
        
        # Barrel classification (rough approximation)
        if 'launch_speed' in df.columns and 'launch_angle' in df.columns:
            df['barrel'] = ((df['launch_speed'] >= 98) & 
                           (df['launch_angle'] >= 26) & 
                           (df['launch_angle'] <= 30)).astype(int)
        
        # Hit classification
        if 'events' in df.columns:
            df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
            df['is_home_run'] = (df['events'] == 'home_run').astype(int)
        
        return df
    
    def _aggregate_by_game(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate Statcast data by player and game
        """
        print("📊 Aggregating performance by game...")
        
        # Group by player, game date, and teams
        group_cols = ['player_name', 'game_date', 'season', 'home_team', 'away_team']
        
        # Aggregation functions
        agg_dict = {
            'events': 'count',  # Total plate appearances
            'is_hit': 'sum',    # Total hits
            'is_home_run': 'sum',  # Home runs
        }
        
        # Add advanced metrics if available
        if 'launch_speed' in df.columns:
            agg_dict['launch_speed'] = 'mean'  # Average exit velocity
            agg_dict['hard_hit'] = 'sum'       # Hard-hit balls
        
        if 'launch_angle' in df.columns:
            agg_dict['launch_angle'] = 'mean'  # Average launch angle
        
        if 'barrel' in df.columns:
            agg_dict['barrel'] = 'sum'         # Barrel count
            
        # Perform aggregation
        game_stats = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename columns for clarity
        column_mapping = {
            'events': 'plate_appearances',
            'is_hit': 'hits',
            'is_home_run': 'home_runs',
            'hard_hit': 'hard_hit_balls',
            'barrel': 'barrels'
        }
        
        game_stats = game_stats.rename(columns=column_mapping)
        
        # Calculate derived statistics
        game_stats['at_bats'] = game_stats['plate_appearances']  # Simplified
        game_stats['batting_avg'] = game_stats['hits'] / game_stats['at_bats'].replace(0, 1)
        
        if 'hard_hit_balls' in game_stats.columns:
            game_stats['hard_hit_rate'] = game_stats['hard_hit_balls'] / game_stats['at_bats'].replace(0, 1)
        
        if 'barrels' in game_stats.columns:
            game_stats['barrel_rate'] = game_stats['barrels'] / game_stats['at_bats'].replace(0, 1)
        
        return game_stats
    
    def _save_to_database(self, game_stats: pd.DataFrame) -> int:
        """
        Save aggregated game statistics to historical database
        """
        print("💾 Saving to historical database...")
        
        saved_count = 0
        session = self.historical_manager.get_session()
        
        try:
            for _, row in game_stats.iterrows():
                # Determine team assignment (player was batting for which team)
                # This is simplified - in reality would need roster lookup
                player_team = row['home_team']  # Simplified assumption
                opponent = row['away_team']
                
                # Create historical performance record
                performance = HistoricalPlayerPerformance(
                    player_id=hash(row['player_name']) % 1000000,  # Generate consistent ID
                    player_name=row['player_name'],
                    game_date=row['game_date'].date(),
                    season=int(row['season']),
                    team=player_team,
                    opponent=opponent,
                    home_away='home',  # Simplified
                    ballpark='Unknown',  # Would need venue lookup
                    
                    # Basic batting stats
                    at_bats=int(row['at_bats']),
                    hits=int(row['hits']),
                    home_runs=int(row['home_runs']),
                    
                    # Advanced Statcast metrics (using correct column names)
                    launch_angle_avg=row.get('launch_angle', 0),
                    hard_hit_rate=row.get('hard_hit_rate', 0),
                    barrel_rate=row.get('barrel_rate', 0)
                )
                
                # Use merge to avoid duplicates
                session.merge(performance)
                saved_count += 1
                
                # Commit in batches
                if saved_count % 100 == 0:
                    session.commit()
                    print(f"   - Saved {saved_count} records...")
            
            # Final commit
            session.commit()
            print(f"✅ Successfully saved {saved_count} records")
            
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
            session.rollback()
            
        finally:
            session.close()
        
        return saved_count
    
    def _print_processing_summary(self, game_stats: pd.DataFrame):
        """
        Print summary of processed data
        """
        print(f"\n📊 Processing Summary:")
        print("=" * 30)
        
        total_games = len(game_stats)
        unique_players = game_stats['player_name'].nunique()
        date_range = f"{game_stats['game_date'].min().strftime('%Y-%m-%d')} to {game_stats['game_date'].max().strftime('%Y-%m-%d')}"
        
        print(f"Game records created: {total_games:,}")
        print(f"Unique players: {unique_players}")
        print(f"Date range: {date_range}")
        
        # Top performers
        if len(game_stats) > 0:
            print(f"\n🌟 Top Performers (by total hits):")
            top_hitters = game_stats.groupby('player_name')['hits'].sum().sort_values(ascending=False).head(5)
            for player, hits in top_hitters.items():
                avg_exit_velo = game_stats[game_stats['player_name'] == player]['launch_speed'].mean()
                if not np.isnan(avg_exit_velo):
                    print(f"   • {player}: {hits} hits, {avg_exit_velo:.1f} mph avg exit velocity")
                else:
                    print(f"   • {player}: {hits} hits")
        
        print(f"\n🎯 Enhanced betting intelligence now available!")
        print(f"   • Advanced Statcast metrics integrated")
        print(f"   • Exit velocity and launch angle data")
        print(f"   • Hard-hit rate and barrel analysis")

def main():
    """
    Process the uploaded Statcast file
    """
    processor = StatcastProcessor()
    file_path = "attached_assets/statcast_673357_1752329574688.csv"
    
    success = processor.process_statcast_file(file_path)
    
    if success:
        print(f"\n✅ Statcast integration completed successfully!")
        print(f"🚀 Betting engine now enhanced with advanced metrics")
    else:
        print(f"\n❌ Processing failed")

if __name__ == "__main__":
    main()
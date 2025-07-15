#!/usr/bin/env python3
"""
Optimized Batch Statcast Processor
High-performance processing for large-scale Statcast datasets
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from historical_data_manager import HistoricalDataManager, HistoricalPlayerPerformance
from sqlalchemy import text
import warnings
warnings.filterwarnings('ignore')

class OptimizedStatcastProcessor:
    """
    High-performance processor for large Statcast datasets
    """
    
    def __init__(self):
        self.historical_manager = HistoricalDataManager()
        self.processed_count = 0
        self.total_records = 0
        
    def process_all_files_optimized(self, batch_size=10):
        """
        Process all Statcast files with optimized performance
        """
        print("🚀 Optimized Statcast Batch Processor")
        print("=" * 60)
        
        csv_files = glob.glob("statcast_data/*.csv")
        print(f"📊 Found {len(csv_files)} Statcast files to process")
        print(f"⚡ Processing in batches of {batch_size} files for optimal performance")
        
        # Process files in batches
        total_processed = 0
        total_failed = 0
        
        for i in range(0, len(csv_files), batch_size):
            batch = csv_files[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(csv_files) + batch_size - 1) // batch_size
            
            print(f"\n🔄 Processing Batch {batch_num}/{total_batches} ({len(batch)} files)")
            print("=" * 40)
            
            batch_success, batch_failed = self._process_file_batch(batch)
            total_processed += batch_success
            total_failed += batch_failed
            
            print(f"📊 Batch {batch_num} Results: {batch_success} success, {batch_failed} failed")
        
        print(f"\n✅ PROCESSING COMPLETE")
        print("=" * 40)
        print(f"Successfully processed: {total_processed} files")
        print(f"Failed: {total_failed} files")
        print(f"Total Statcast records integrated: {self.total_records:,}")
        
        # Show final database stats
        self._show_final_stats()
    
    def _process_file_batch(self, file_batch):
        """
        Process a batch of files efficiently
        """
        successful = 0
        failed = 0
        
        session = self.historical_manager.get_session()
        
        try:
            for file_path in file_batch:
                filename = os.path.basename(file_path)
                print(f"   📄 Processing: {filename}")
                
                try:
                    # Quick file processing
                    record_count = self._process_single_file_fast(file_path, session)
                    self.total_records += record_count
                    successful += 1
                    print(f"      ✅ {record_count:,} records")
                    
                except Exception as e:
                    failed += 1
                    print(f"      ❌ Error: {str(e)[:50]}...")
            
            # Commit batch
            session.commit()
            print(f"   💾 Batch committed to database")
            
        except Exception as e:
            print(f"   ❌ Batch error: {e}")
            session.rollback()
            
        finally:
            session.close()
        
        return successful, failed
    
    def _process_single_file_fast(self, file_path, session):
        """
        Fast processing of a single Statcast file
        """
        # Read CSV with optimized settings
        df = pd.read_csv(file_path, low_memory=False, usecols=[
            'player_name', 'game_date', 'events', 'home_team', 'away_team',
            'launch_speed', 'launch_angle', 'hit_distance_sc', 'bb_type'
        ])
        
        # Quick data cleaning
        df['game_date'] = pd.to_datetime(df['game_date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['game_date', 'player_name'])
        df = df[df['player_name'] != '']
        df['season'] = df['game_date'].dt.year
        
        if len(df) == 0:
            return 0
        
        # Create derived metrics
        df['is_hit'] = df['events'].isin(['single', 'double', 'triple', 'home_run']).astype(int)
        df['is_home_run'] = (df['events'] == 'home_run').astype(int)
        df['hard_hit'] = (df['launch_speed'].fillna(0) >= 95).astype(int)
        
        # Aggregate by game quickly
        game_stats = df.groupby(['player_name', 'game_date', 'season', 'home_team', 'away_team']).agg({
            'events': 'count',
            'is_hit': 'sum',
            'is_home_run': 'sum',
            'launch_speed': 'mean',
            'launch_angle': 'mean',
            'hard_hit': 'sum'
        }).reset_index()
        
        game_stats.columns = ['player_name', 'game_date', 'season', 'home_team', 'away_team',
                              'at_bats', 'hits', 'home_runs', 'avg_exit_velo', 'avg_launch_angle', 'hard_hit_balls']
        
        # Calculate rates
        game_stats['hard_hit_rate'] = game_stats['hard_hit_balls'] / game_stats['at_bats'].replace(0, 1)
        game_stats['barrel_rate'] = 0.0  # Simplified for speed
        
        # Bulk insert using merge
        records_created = 0
        for _, row in game_stats.iterrows():
            performance = HistoricalPlayerPerformance(
                player_id=hash(row['player_name']) % 1000000,
                player_name=row['player_name'],
                game_date=row['game_date'].date(),
                season=int(row['season']),
                team=row['home_team'],  # Simplified
                opponent=row['away_team'],
                home_away='home',
                ballpark='Unknown',
                at_bats=int(row['at_bats']),
                hits=int(row['hits']),
                home_runs=int(row['home_runs']),
                launch_angle_avg=row['avg_launch_angle'] if not pd.isna(row['avg_launch_angle']) else 0,
                hard_hit_rate=row['hard_hit_rate'],
                barrel_rate=row['barrel_rate']
            )
            
            session.merge(performance)
            records_created += 1
        
        return records_created
    
    def _show_final_stats(self):
        """
        Show final database statistics
        """
        session = self.historical_manager.get_session()
        try:
            result = session.execute(text("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT player_name) as unique_players,
                    COUNT(CASE WHEN launch_angle_avg > 0 THEN 1 END) as with_statcast,
                    MIN(game_date) as earliest_date,
                    MAX(game_date) as latest_date
                FROM historical_player_performance
            """)).fetchone()
            
            print(f"\n📈 Final Database Statistics:")
            print("=" * 40)
            print(f"Total records: {result[0]:,}")
            print(f"Unique players: {result[1]:,}")
            print(f"Records with Statcast metrics: {result[2]:,}")
            print(f"Date range: {result[3]} to {result[4]}")
            
        except Exception as e:
            print(f"Could not retrieve final stats: {e}")
        finally:
            session.close()

def main():
    processor = OptimizedStatcastProcessor()
    processor.process_all_files_optimized(batch_size=20)

if __name__ == "__main__":
    main()
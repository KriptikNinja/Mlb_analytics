#!/usr/bin/env python3
"""
Statcast File Analyzer
Analyzes and processes user's Statcast CSV files for enhanced betting intelligence
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def analyze_statcast_file(file_path: str):
    """
    Analyze the structure and content of a Statcast CSV file
    """
    print(f"📊 Analyzing Statcast file: {file_path}")
    print("=" * 50)
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        
        print(f"📈 File Statistics:")
        print(f"   - Total rows: {len(df):,}")
        print(f"   - Total columns: {len(df.columns)}")
        print(f"   - File size: ~{len(df) * len(df.columns)} data points")
        
        # Key columns analysis
        key_columns = {
            'player_name': 'Player identification',
            'game_date': 'Game date information', 
            'events': 'Plate appearance outcomes',
            'launch_speed': 'Exit velocity (mph)',
            'launch_angle': 'Launch angle (degrees)',
            'hit_distance_sc': 'Hit distance (feet)',
            'bb_type': 'Batted ball type',
            'home_team': 'Home team',
            'away_team': 'Away team',
            'pitcher': 'Pitcher ID',
            'batter': 'Batter ID',
            'release_speed': 'Pitch velocity',
            'pitch_type': 'Pitch type',
            'zone': 'Strike zone location'
        }
        
        print(f"\n🎯 Key Columns Available:")
        available_key_cols = []
        for col, description in key_columns.items():
            if col in df.columns:
                available_key_cols.append(col)
                non_null_count = df[col].notna().sum()
                print(f"   ✓ {col}: {description} ({non_null_count:,} non-null values)")
            else:
                print(f"   ✗ {col}: {description} (not found)")
        
        # Date range analysis
        if 'game_date' in df.columns:
            df['game_date'] = pd.to_datetime(df['game_date'])
            date_range = f"{df['game_date'].min().strftime('%Y-%m-%d')} to {df['game_date'].max().strftime('%Y-%m-%d')}"
            unique_dates = df['game_date'].dt.date.nunique()
            print(f"\n📅 Date Coverage:")
            print(f"   - Date range: {date_range}")
            print(f"   - Unique game dates: {unique_dates}")
        
        # Player analysis
        if 'player_name' in df.columns:
            unique_players = df['player_name'].nunique()
            top_players = df['player_name'].value_counts().head(5)
            print(f"\n👥 Player Coverage:")
            print(f"   - Unique players: {unique_players}")
            print(f"   - Top 5 players by plate appearances:")
            for player, count in top_players.items():
                print(f"     • {player}: {count} PA")
        
        # Events analysis (outcomes)
        if 'events' in df.columns:
            events = df['events'].value_counts().head(10)
            print(f"\n⚾ Plate Appearance Outcomes:")
            for event, count in events.items():
                if pd.notna(event):
                    print(f"   • {event}: {count}")
        
        # Advanced metrics availability
        advanced_metrics = {
            'launch_speed': 'Exit velocity data',
            'launch_angle': 'Launch angle data', 
            'hit_distance_sc': 'Hit distance data',
            'bb_type': 'Batted ball type data',
            'barrel': 'Barrel classification',
            'release_speed': 'Pitch velocity data',
            'spin_rate_deprecated': 'Spin rate data'
        }
        
        print(f"\n🚀 Advanced Metrics Available:")
        for metric, description in advanced_metrics.items():
            if metric in df.columns:
                non_null = df[metric].notna().sum()
                if non_null > 0:
                    print(f"   ✓ {metric}: {description} ({non_null:,} values)")
                    
                    # Show sample stats for numeric columns
                    if df[metric].dtype in ['float64', 'int64'] and non_null > 0:
                        mean_val = df[metric].mean()
                        if not np.isnan(mean_val):
                            print(f"     - Average: {mean_val:.1f}")
        
        # Quality assessment
        print(f"\n✅ Data Quality Assessment:")
        
        # Check for essential betting analytics columns
        essential_cols = ['player_name', 'game_date', 'events']
        has_essentials = all(col in available_key_cols for col in essential_cols)
        print(f"   - Essential columns: {'✓ Complete' if has_essentials else '✗ Missing some'}")
        
        # Check for advanced analytics
        advanced_cols = ['launch_speed', 'launch_angle', 'hit_distance_sc']
        has_advanced = any(col in available_key_cols for col in advanced_cols)
        print(f"   - Advanced metrics: {'✓ Available' if has_advanced else '✗ Limited'}")
        
        # Check data density
        total_cells = len(df) * len(df.columns)
        non_null_cells = df.notna().sum().sum()
        density = (non_null_cells / total_cells) * 100
        print(f"   - Data density: {density:.1f}% complete")
        
        print(f"\n🎯 Integration Readiness:")
        if has_essentials and len(df) > 1000:
            print("   ✅ EXCELLENT - Ready for immediate integration")
            print("   ✅ Will significantly enhance betting intelligence")
            print("   ✅ Contains comprehensive plate appearance data")
        elif has_essentials:
            print("   ✅ GOOD - Ready for integration")
            print("   ⚠️  Smaller dataset but still valuable")
        else:
            print("   ⚠️  LIMITED - Missing some essential columns")
            
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'available_key_columns': available_key_cols,
            'has_essentials': has_essentials,
            'has_advanced': has_advanced,
            'data_density': density,
            'date_range': date_range if 'game_date' in df.columns else None,
            'unique_players': unique_players if 'player_name' in df.columns else 0
        }
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None

def create_processing_plan(analysis_result: Dict):
    """
    Create a processing plan based on file analysis
    """
    if not analysis_result:
        return
        
    print(f"\n📋 Processing Plan:")
    print("=" * 30)
    
    print("1. 🔄 Data Integration Steps:")
    print("   • Parse game dates and player names")
    print("   • Extract plate appearance outcomes")
    print("   • Calculate performance metrics per game")
    
    if analysis_result['has_advanced']:
        print("   • Process advanced Statcast metrics:")
        print("     - Exit velocity analysis")
        print("     - Launch angle distribution") 
        print("     - Hard-hit rate calculation")
        print("     - Barrel rate determination")
    
    print("\n2. 🎯 Betting Enhancement Features:")
    print("   • Player hot streak detection from recent games")
    print("   • Advanced performance trends")
    print("   • Platoon advantage analysis")
    
    if analysis_result['has_advanced']:
        print("   • Statcast-based edge detection:")
        print("     - High exit velocity trends")
        print("     - Barrel rate advantages")
        print("     - Launch angle optimization")
    
    estimated_time = max(1, analysis_result['total_rows'] // 1000)
    print(f"\n3. ⏱️  Processing Estimate:")
    print(f"   • Estimated processing time: {estimated_time} minutes")
    print(f"   • Will create ~{analysis_result['total_rows']//10} performance records")
    print(f"   • Enhanced data for {analysis_result['unique_players']} players")

if __name__ == "__main__":
    # Analyze the uploaded file
    file_path = "attached_assets/statcast_673357_1752329574688.csv"
    analysis = analyze_statcast_file(file_path)
    
    if analysis:
        create_processing_plan(analysis)
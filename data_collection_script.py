#!/usr/bin/env python3
"""
Data Collection Script for Historical MLB Data
Collects and processes historical MLB data for betting analysis
"""

import sys
import os
from datetime import datetime
from historical_data_manager import HistoricalDataManager

def main():
    """Main data collection function"""
    print("MLB Historical Data Collection Script")
    print("=" * 50)
    
    try:
        # Initialize historical data manager
        historical_manager = HistoricalDataManager()
        print("✅ Historical data manager initialized")
        
        # Define seasons to collect (comprehensive multi-season collection)
        current_year = datetime.now().year
        seasons = [2022, 2023, 2024]  # Full 3-season collection for robust analysis
        print(f"📅 Collecting comprehensive data for seasons: {seasons}")
        print("🔄 Full historical collection - building complete database for advanced analytics")
        
        # Start collection process
        print("🚀 Starting historical data collection...")
        historical_manager.collect_historical_seasons(seasons)
        
        print("✅ Historical data collection completed successfully!")
        print("\nNext steps:")
        print("1. Historical matchup analysis ready")
        print("2. Player similarity scoring available") 
        print("3. Enhanced betting opportunities enabled")
        
    except Exception as e:
        print(f"❌ Error during data collection: {e}")
        print("Please check your database connection and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
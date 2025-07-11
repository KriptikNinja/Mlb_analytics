#!/usr/bin/env python3
"""
Simple database test script to verify functionality
"""

import os
import sys
import traceback
from datetime import datetime

def test_database():
    """Test basic database functionality"""
    try:
        print("Testing database connection...")
        
        # Check environment variables
        if not os.getenv('DATABASE_URL'):
            print("❌ DATABASE_URL not found")
            return False
        
        print("✓ DATABASE_URL found")
        
        # Import and test database manager
        from database_manager import DatabaseManager
        
        print("✓ DatabaseManager imported")
        
        # Initialize database
        db = DatabaseManager()
        print("✓ Database initialized")
        
        # Test teams
        teams = db.get_teams()
        print(f"✓ Found {len(teams)} teams")
        
        if len(teams) > 0:
            print(f"  Sample teams: {teams[:3]}")
            
            # Test games
            today = datetime.now().date()
            games = db.get_games_for_date(today)
            print(f"✓ Found {len(games)} games for {today}")
            
            if games:
                print(f"  Sample game: {games[0]}")
                
            print("✓ Database test completed successfully")
            return True
        else:
            print("⚠ No teams found - database may need initialization")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_database()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify historical data integration with betting engine
"""

from advanced_betting_engine import AdvancedBettingEngine
from datetime import datetime

def test_betting_engine_integration():
    """Test that betting engine can integrate with historical data"""
    print("Testing Historical Data Integration")
    print("=" * 40)
    
    try:
        # Initialize betting engine (should auto-load historical manager)
        betting_engine = AdvancedBettingEngine()
        
        # Check if historical manager loaded
        if betting_engine.historical_manager:
            print("✅ Historical data manager loaded successfully")
            
            # Test basic functionality
            print("🧪 Testing basic historical data functionality...")
            
            # Create mock game data for testing
            test_game = {
                'away_team': 'Test Away',
                'home_team': 'Test Home',
                'game_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            test_batter = {
                'id': 12345,
                'name': 'Test Batter',
                'player_type': 'batter',
                'predictions': {
                    'hit_probability': 0.30,
                    'predicted_avg': 0.275
                }
            }
            
            # Test historical matchup boost
            matchup_boost = betting_engine._get_historical_matchup_boost(test_batter, test_game)
            print(f"📊 Matchup boost test: {matchup_boost}")
            
            print("✅ Historical integration test completed")
            
        else:
            print("⚠️  Historical data manager not loaded - will use basic analysis only")
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        print("The system will fall back to basic betting analysis")

if __name__ == "__main__":
    test_betting_engine_integration()
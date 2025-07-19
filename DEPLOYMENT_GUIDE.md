# MLB Analytics Self-Hosting Deployment Guide

## Data Architecture Overview

### Current Data Storage (Replit Environment)
The system uses **PostgreSQL database** hosted on Replit's infrastructure:
- **Database Type**: PostgreSQL (managed by Replit)
- **Connection**: Via DATABASE_URL environment variable
- **Tables**: 3 main tables storing your authentic MLB data
- **Size**: 76,000+ historical records + growing 2025 season data

### What Data Gets Saved

#### 1. Historical Player Performance Table
```sql
historical_player_performance
- player_id, player_name, game_date, season
- team, opponent, home_away, ballpark
- Batting stats: at_bats, hits, doubles, triples, home_runs, rbis, walks, strikeouts
- Pitching stats: innings_pitched, earned_runs, strikeouts_pitched
- Advanced metrics: exit_velocity_avg, launch_angle_avg, hard_hit_rate, barrel_rate
- Weather data: temperature, wind_speed, wind_direction
```

#### 2. Player Similarity Scores Table
```sql
player_similarity
- Similarity calculations between players for matchup analysis
- Used for finding comparable performance patterns
```

#### 3. Historical Matchups Table
```sql
historical_matchups
- Batter vs pitcher historical performance data
- Career head-to-head statistics
- Sample size confidence scoring
```

### Your Statcast Files (Static Assets)
- **Location**: `statcast_data/` folder
- **Count**: 792 CSV files (2.37 million pitch records)
- **Transfer**: These files copy directly in the ZIP download
- **Usage**: Baseline Statcast metrics for zone analysis

## Self-Hosting Deployment Options

### Option 1: Complete Database Export (Recommended)
```bash
# Export your current database to a SQL file
pg_dump $DATABASE_URL > mlb_analytics_backup.sql

# On your Linux machine:
# 1. Install PostgreSQL
sudo apt update && sudo apt install postgresql postgresql-contrib

# 2. Create database and user
sudo -u postgres createdb mlb_analytics
sudo -u postgres createuser mlb_user -P

# 3. Import your data
psql -U mlb_user -d mlb_analytics -f mlb_analytics_backup.sql

# 4. Update environment variables
export DATABASE_URL="postgresql://mlb_user:password@localhost/mlb_analytics"
```

### Option 2: Fresh Setup with Historical Collection
If you prefer to start fresh on your Linux machine:

```bash
# The historical data collection will rebuild everything
# from your 792 Statcast files + live MLB API data
python data_collection_script.py
```

### Environment Variables for Self-Hosting
```bash
# Required for database connection
export DATABASE_URL="postgresql://username:password@localhost/database_name"
export PGHOST="localhost"
export PGPORT="5432"
export PGUSER="your_username"
export PGPASSWORD="your_password" 
export PGDATABASE="mlb_analytics"

# Optional: API keys if you want enhanced features
export OPENAI_API_KEY="your_key_here"  # For AI analysis
```

### Daily Updates on Self-Hosted System
The system will continue updating automatically:

1. **Live MLB Data**: Fetched daily from MLB Stats API (no API key required)
2. **Historical Collection**: Continues adding 2025 season games
3. **Statcast Integration**: Your 792 files provide baseline metrics
4. **Database Growth**: Approximately 1,000-2,000 new records per day during season

### File Structure for Self-Hosting
```
mlb_analytics/
├── statcast_data/           # Your 792 CSV files (transfers)
├── app.py                   # Main Streamlit application
├── real_data_fetcher.py     # MLB API integration
├── historical_data_manager.py # Database management
├── simple_betting_engine.py  # Betting analysis
├── requirements.txt         # Python dependencies
└── mlb_analytics_backup.sql # Your database export
```

### Key Benefits of Self-Hosting
- **Full Data Ownership**: Complete control over your 76,000+ records
- **No API Limits**: Direct MLB Stats API access (public endpoints)
- **Continuous Updates**: System keeps collecting 2025 season data
- **Your Statcast Dataset**: All 792 files transfer with the project
- **Scalability**: Add more seasons or enhanced features

### Performance Considerations
- **Database Size**: Currently ~100MB, grows ~50MB per season
- **Memory Usage**: 1-2GB RAM recommended for Streamlit + PostgreSQL
- **Storage**: 2-5GB total including Statcast files and database
- **Update Frequency**: Daily collection adds minimal load

The beauty of this system is that **all your authentic data transfers completely** - both the database records and your comprehensive Statcast file collection.
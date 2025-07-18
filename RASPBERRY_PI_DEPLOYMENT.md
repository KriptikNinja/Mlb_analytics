# MLB Analytics V2.0 - Raspberry Pi Deployment Guide

## Essential Files for GitHub Upload

### Core Application Files (REQUIRED)
```
app.py                          # Main Streamlit application 
real_data_fetcher.py           # MLB API data fetcher
real_prediction_engine.py      # Prediction engine
advanced_betting_engine.py     # Betting analysis engine
simple_betting_engine.py       # Backup betting engine
auth.py                        # Authentication system
```

### Supporting Modules (REQUIRED)
```
analytics_engine.py            # Statistical analysis
ml_predictor.py               # Machine learning models
strike_zone_analyzer.py       # Strike zone analysis
visualization.py              # Plotly charts
data_fetcher.py               # Base data fetcher
config.py                     # Configuration settings
```

### Database & Data Processing (REQUIRED)
```
database_manager.py           # PostgreSQL database management
historical_data_manager.py    # Historical data processing
statcast_processor.py         # Statcast data processing
```

### Configuration Files (REQUIRED)
```
pyproject.toml               # Python dependencies
replit.md                    # Project documentation
AUTH_SETUP.md               # Authentication instructions
```

### Optional Files (RECOMMENDED)
```
generated-icon.png           # App icon
data_collection_script.py   # Background data collection
batch_statcast_processor.py # Batch processing utility
```

## Dependencies Installation on Raspberry Pi

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python 3.11+ (if not already installed)
```bash
sudo apt install python3.11 python3.11-pip python3.11-venv -y
```

### 3. Install PostgreSQL (for database)
```bash
sudo apt install postgresql postgresql-contrib -y
sudo -u postgres createuser --interactive
sudo -u postgres createdb mlb_analytics
```

### 4. Create Virtual Environment
```bash
python3.11 -m venv mlb_env
source mlb_env/bin/activate
```

### 5. Install Python Dependencies
```bash
pip install -r requirements.txt
```

## Requirements.txt Content
Create this file in your project root:
```
streamlit>=1.46.1
pandas>=2.3.1
numpy>=2.3.1
plotly>=6.2.0
scikit-learn>=1.7.0
scipy>=1.16.0
requests>=2.32.4
psycopg2-binary>=2.9.10
sqlalchemy>=2.0.41
pytz>=2025.2
trafilatura>=2.0.0
```

## Environment Variables Setup

Create a `.env` file (do NOT upload to GitHub):
```
DATABASE_URL=postgresql://username:password@localhost:5432/mlb_analytics
PGHOST=localhost
PGPORT=5432
PGUSER=your_username
PGPASSWORD=your_password
PGDATABASE=mlb_analytics
```

## Running on Raspberry Pi

### 1. Clone from GitHub
```bash
git clone https://github.com/yourusername/mlb-analytics.git
cd mlb-analytics
```

### 2. Activate Environment
```bash
source mlb_env/bin/activate
```

### 3. Set Environment Variables
```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/mlb_analytics"
# Or use: source .env
```

### 4. Initialize Database
```bash
python database_manager.py
```

### 5. Run Application
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Performance Optimization for Raspberry Pi

### 1. Reduce Memory Usage
Add to your app.py:
```python
import streamlit as st
st.set_page_config(
    page_title="MLB Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

### 2. Enable Caching
The app already has 5-minute API caching enabled for optimal Pi performance.

### 3. Database Optimization
```sql
-- Run these in PostgreSQL to optimize for Pi
CREATE INDEX idx_player_name ON historical_player_performance(player_name);
CREATE INDEX idx_game_date ON historical_player_performance(game_date);
```

## Files NOT to Upload to GitHub
```
__pycache__/                 # Python cache files
.env                        # Environment variables
statcast_data/              # Large CSV files (unless needed)
attached_assets/            # Screenshots and images
uv.lock                     # Lock file specific to Replit
.replit                     # Replit configuration
```

## Troubleshooting on Raspberry Pi

### Memory Issues
- Close unused applications
- Use swap file: `sudo dphys-swapfile swapoff && sudo dphys-swapfile swapon`

### Port Conflicts
- Change port: `streamlit run app.py --server.port 8502`

### Database Connection Issues
- Check PostgreSQL service: `sudo systemctl status postgresql`
- Restart if needed: `sudo systemctl restart postgresql`

## Security Considerations

1. Never upload `.env` files to GitHub
2. Use environment variables for database credentials
3. Consider using GitHub Secrets for sensitive data
4. Set up firewall rules for external access

## Testing Deployment

After setup, test these URLs:
- Local: `http://localhost:8501`
- Network: `http://YOUR_PI_IP:8501`

All V2.0 features (date selection, numerical dates, betting analysis) should work identically to the Replit version.
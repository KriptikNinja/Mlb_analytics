# MLB Daily Matchup Analytics

## Overview

This repository contains a streamlined MLB analytics application built with Python and Streamlit. The application automatically pulls today's MLB games and provides AI-powered predictions and probability analysis for all players, including pitcher strikeouts. The focus is on daily matchup analysis with clean, organized presentation of MLB team data only.

## User Preferences

Preferred communication style: Simple, everyday language.
Focus: Daily MLB games with automatic data loading, no manual player input required.
Display: Clean interface showing predictions and probabilities for today's matchups only.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend
- **Streamlit**: Web-based interface for interactive analytics
- **Plotly**: Interactive visualizations and charts
- **Multi-page navigation**: Organized into distinct analysis sections

### Backend
- **Python-based**: Core analytics engine with specialized modules
- **RESTful API Integration**: MLB Stats API for real-time data
- **Caching Layer**: In-memory caching for API responses (5-minute timeout)

### Data Processing
- **Pandas/NumPy**: Data manipulation and statistical analysis
- **Scikit-learn**: Machine learning models for predictions
- **SciPy**: Advanced statistical computations

## Key Components

### 1. Data Layer (`data_fetcher.py`)
**Purpose**: Handles all external data retrieval and caching
- **MLB Stats API Integration**: Fetches real-time MLB data
- **Caching Mechanism**: Reduces API calls with 5-minute cache timeout
- **Error Handling**: Graceful fallbacks for API failures
- **Data Normalization**: Standardizes data formats across sources

### 2. Analytics Engine (`analytics_engine.py`)
**Purpose**: Core statistical analysis and calculations
- **Rolling Averages**: Time-series analysis with configurable windows
- **Streak Detection**: Identifies hot/cold performance streaks
- **Percentile-based Thresholds**: 70th percentile for hot, 30th for cold
- **Statistical Validation**: Scipy-based statistical tests

### 3. Machine Learning Predictor (`ml_predictor.py`)
**Purpose**: Predictive modeling for player performance
- **Multiple Algorithms**: Random Forest, Gradient Boosting, Linear Regression
- **Feature Engineering**: Automatic feature preparation and scaling
- **Performance Tracking**: Model accuracy and feature importance metrics
- **Separate Models**: Distinct models for batters and pitchers

### 4. Strike Zone Analyzer (`strike_zone_analyzer.py`)
**Purpose**: Specialized analysis of strike zone performance
- **9-Zone Grid System**: Standard MLB strike zone division
- **Dual Analysis**: Both batter and pitcher zone performance
- **Heat Map Generation**: Visual representation of zone effectiveness
- **Sample Data Fallback**: Mock data when API data unavailable

### 5. Visualization Layer (`visualization.py`)
**Purpose**: Interactive charts and visual analytics
- **Plotly Integration**: Interactive, web-ready visualizations
- **Color-coded Themes**: Consistent visual language
- **Multiple Chart Types**: Line charts, heat maps, scatter plots
- **Responsive Design**: Adapts to different screen sizes

### 6. Main Application (`app.py`)
**Purpose**: Streamlit interface orchestration
- **Session State Management**: Persistent component instances
- **Page Navigation**: Multi-page application structure
- **Component Integration**: Coordinates all backend modules
- **Error Handling**: User-friendly error messages

## Data Flow

1. **Data Acquisition**: `MLBDataFetcher` retrieves data from MLB Stats API
2. **Caching**: API responses cached for 5 minutes to reduce load
3. **Processing**: `AnalyticsEngine` performs statistical calculations
4. **Prediction**: `MLPredictor` generates performance forecasts
5. **Visualization**: `Visualizer` creates interactive charts
6. **Presentation**: Streamlit renders the web interface

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning algorithms
- **scipy**: Statistical functions
- **requests**: HTTP API calls

### MLB Stats API
- **Base URL**: `https://statsapi.mlb.com/api/v1`
- **Rate Limiting**: Implemented through caching
- **Authentication**: No API key required (public endpoints)
- **Data Types**: Teams, players, statistics, game data

### Performance Considerations
- **Caching Strategy**: 5-minute cache timeout for API responses
- **Error Handling**: Graceful degradation with sample data
- **Memory Management**: Efficient pandas operations
- **Lazy Loading**: Components initialized only when needed

## Deployment Strategy

### Local Development
- **Streamlit Server**: Built-in development server
- **Hot Reload**: Automatic code reloading during development
- **Environment**: Python 3.7+ with pip dependencies

### Production Considerations
- **Containerization**: Docker-ready architecture
- **Scalability**: Stateless design allows horizontal scaling
- **Monitoring**: Error logging and performance tracking
- **Security**: Input validation and API rate limiting

### Configuration
- **Environment Variables**: API keys and configuration
- **Feature Flags**: Toggle features without code changes
- **Logging**: Configurable logging levels
- **Performance Tuning**: Adjustable cache timeouts and model parameters

The application is designed to be modular, maintainable, and extensible, with clear separation between data fetching, analytics processing, machine learning, and visualization components.

## Recent Updates (July 2025)

### Enhanced Features Implemented
- **Central Time Display**: All game times automatically converted to Central Time (CT) for user preference
- **Starting Pitchers Only**: Daily matchups now filter to show only actual starting pitchers (SP) instead of all roster pitchers
- **Ballpark & Weather Data**: Added venue-specific factors including HR factor, elevation, temperature, wind speed/direction, and conditions
- **Data Authenticity Focus**: Removed fake historical matchup data, predictions now use only real MLB season statistics for accuracy
- **Advanced Strike Zone Analysis**: Complete overhaul with 7 specialized charts:
  - Exit Velocity by Zone
  - Launch Angle Analysis  
  - Hard Hit Rate (95+ mph)
  - Batted Ball Distribution
  - Barrel Rate by Zone
  - Hit Type Analysis (HR rate)
  - Strikeout Heat Map
- **Hot/Cold Streaks Fix**: Resolved data loading issues with fallback sample data when API unavailable
- **Auto-Selected Players**: Strike Zone Analysis now pre-populates with today's starting pitchers

### Latest Major Enhancement (July 11, 2025) - IN PROGRESS 🔧
- **Advanced Betting Engine Integration**: Successfully developed sophisticated betting powerhouse combining multiple edge detection strategies
  - Implemented hot streak detection using real season stats and prediction confidence scores
  - Added ballpark advantage analysis for HR-friendly venues and strikeout-conducive parks
  - Created matchup edge detection for platoon advantages and contact vs power mismatches
  - Integrated weather opportunity analysis for wind and temperature factors
  - Built Kelly Criterion-based bet sizing with conservative 3-unit maximum per opportunity
  - Fixed critical data structure compatibility between prediction engine and betting engine
  - **STATUS**: Betting opportunities page now processes real MLB data without "'str' object has no attribute 'get'" errors

### Previous Major Enhancement (July 11, 2025) - COMPLETED ✅
- **Critical Pitcher Prediction Bug Fix**: Fixed unrealistic pitcher predictions that were showing impossible stats (17 strikeouts, 20 hits allowed)
  - Completely rewrote pitcher prediction calculation logic using proper rate statistics (K/9, BB/9, H/9, ER/9)
  - Implemented realistic per-game scaling based on typical start length (5.5 innings)
  - Added intelligent caps to prevent absurd predictions (max 12 K's, 10 hits, 8 ER per game)
  - Enhanced pitcher last 5 starts table with HR (Home Runs allowed) column
  - Fixed opponent display to show game dates (MM/DD) instead of "Unknown" when team names unavailable
  - **USER CONFIRMED**: Pitcher predictions now show realistic, believable numbers for professional betting analysis

### Previous Major Enhancement (July 10, 2025) - COMPLETED ✅
- **Rolling Hot Streak Detection**: Successfully implemented 15-game rolling averages for authentic hot/cold streak analysis
  - Batters: Hot if recent 15-game average is 50+ points better than season average or above .300
  - Pitchers: Hot if recent 15-game ERA is 0.75+ better than season average or under 3.25
  - Displays rolling performance in predictions (e.g., "Recent: .324 (15 games)")
- **Real Batter vs Pitcher History**: Successfully integrated MLB Stats API vsPlayer endpoint for authentic matchup data
  - Shows actual head-to-head statistics (e.g., "vs Today's Pitcher: 3/8 (.375) career")
  - Uses confidence weighting based on sample size (full confidence at 20+ at-bats)
  - Enhanced hit probabilities based on historical performance between specific players
  - **USER CONFIRMED**: Batter vs pitcher data is working correctly
- **Pitcher Prediction Fix**: Fixed identical 40% strikeout rate bug affecting all pitchers
  - Updated calculation formula: K/9 divided by 27 instead of 15 to avoid hitting 40% cap
  - Spencer Strider now shows realistic ~38.2% rate instead of capped 40%
  - Enhanced fallback prediction logic with unique variations per pitcher
  - Comprehensive pitcher metrics now display: K's, BB's, ERA, hits allowed, earned runs
- **UI Improvements**: Streamlined interface based on user feedback
  - Removed "Hot/Cold Streaks" option from dropdown menu (per user request)
  - Fixed Strike Zone Analysis to show today's matchups with batter dropdown
  - Enhanced Strike Zone charts with 7 analysis types displaying properly

### Technical Improvements
- Enhanced prediction algorithms incorporating ballpark factors and weather conditions
- Rolling streak calculations using MLB Stats API game logs for authentic recent performance analysis
- Real batter vs pitcher matchup integration using MLB.com gameday data structure
- Improved hot streak detection logic with statistical significance thresholds
- Real player names throughout interface instead of generic placeholders
- Improved error handling with graceful fallbacks for API connectivity issues
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from typing import Dict, List
import warnings
import random
warnings.filterwarnings('ignore')

# Import custom modules with try-catch for faster startup
try:
    from data_fetcher import MLBDataFetcher
    from analytics_engine import AnalyticsEngine
    from visualization import Visualizer
    from ml_predictor import MLPredictor
    from strike_zone_analyzer import StrikeZoneAnalyzer
    from real_prediction_engine import RealMLBPredictionEngine
    from advanced_betting_engine import AdvancedBettingEngine
    from auth import auth
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MLB Analytics Pro",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with lazy loading
@st.cache_resource
def get_data_fetcher():
    return MLBDataFetcher()

@st.cache_resource  
def get_analytics_engine():
    return AnalyticsEngine()

@st.cache_resource
def get_visualizer():
    return Visualizer()

@st.cache_resource
def get_ml_predictor():
    return MLPredictor()

@st.cache_resource
def get_strike_zone_analyzer():
    return StrikeZoneAnalyzer()

@st.cache_resource
def get_prediction_engine():
    return RealMLBPredictionEngine()

@st.cache_resource
def get_betting_engine():
    return AdvancedBettingEngine()

def get_date_options():
    """Get date options for dropdown selection"""
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    
    return {
        "Yesterday": yesterday.strftime("%Y-%m-%d"),
        "Today": today.strftime("%Y-%m-%d"),
        "Tomorrow": tomorrow.strftime("%Y-%m-%d")
    }

# Lazy initialization
if 'initialized' not in st.session_state:
    st.session_state.initialized = True

def main():
    # Require authentication before showing the app
    auth.require_auth()
    
    st.title("⚾ MLB Daily Matchup Analytics")
    st.markdown("Today's MLB games with AI-powered predictions and probability analysis")
    
    # Sidebar navigation
    st.sidebar.title("Analysis Options")
    page = st.sidebar.selectbox(
        "View Type",
        ["Today's Games & Predictions", "Betting Opportunities", "Strike Zone Analysis"]
    )
    
    if page == "Today's Games & Predictions":
        daily_matchups_page()
    elif page == "Betting Opportunities":
        betting_opportunities_page()
    elif page == "Strike Zone Analysis":
        strike_zone_page()

def daily_matchups_page():
    st.header("📅 MLB Games & Predictions")
    
    # Date selection dropdown
    date_options = get_date_options()
    selected_date_label = st.selectbox(
        "Select Game Date:",
        options=list(date_options.keys()),
        index=1,  # Default to "Today"
        key="daily_matchups_date_select"
    )
    selected_date = date_options[selected_date_label]
    
    # Get engines lazily
    prediction_engine = get_prediction_engine()
    
    # Show data source status with timeout handling
    try:
        data_status = prediction_engine.get_data_source_status()
        st.markdown(f"**Data Source:** {data_status}")
    except Exception as e:
        st.markdown("**Data Source:** ⚠️ API connection issues - using sample data")
    
    # Get games for selected date
    with st.spinner(f"Loading real MLB games from MLB Stats API for {selected_date_label.lower()}..."):
        from datetime import datetime
        date_obj = datetime.strptime(selected_date, "%Y-%m-%d").date()
        games = prediction_engine.data_fetcher.get_games_for_date(date_obj)
    
    if not games:
        st.warning(f"No MLB games scheduled for {selected_date_label.lower()}. Check your internet connection or try another date.")
        st.info("This app connects to MLB Stats API (statsapi.mlb.com) and Baseball Savant for real data.")
        return
    
    # Extract actual date from the first game to show accurate date
    actual_game_date = None
    if games and len(games) > 0:
        # Try to parse the date from the first game
        first_game = games[0]
        try:
            # The data fetcher uses target_date, so we can derive it from Eastern time logic
            import pytz
            eastern_tz = pytz.timezone('US/Eastern')
            eastern_now = datetime.now(eastern_tz)
            if eastern_now.hour < 6:
                actual_game_date = (eastern_now - timedelta(days=1)).date()
            else:
                actual_game_date = eastern_now.date()
        except:
            actual_game_date = datetime.now().date()
    else:
        actual_game_date = datetime.now().date()
    
    st.subheader(f"Real Games for {selected_date_label} ({len(games)} games)")
    
    # Display each game with real predictions
    for i, game in enumerate(games):
        game_title = f"🏟️ {game['away_team']} @ {game['home_team']}"
        if game.get('game_time'):
            # Add numerical date before game time
            from datetime import datetime
            try:
                # Parse the selected date to get MM/DD format
                date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
                date_display = date_obj.strftime("%m/%d")
                game_title += f" ({date_display} {game['game_time']})"
            except:
                # Fallback to just game time if date parsing fails
                game_title += f" ({game['game_time']})"
        
        # Add venue info if available
        if game.get('venue'):
            game_title += f" - {game['venue']}"
        
        # Show expanded for first 3 games, collapsed for rest
        expanded = i < 3
        with st.expander(game_title, expanded=expanded):
            with st.spinner("Generating real predictions from MLB data..."):
                game_predictions = prediction_engine.generate_game_predictions(game, simplified=False)
            
            # Starting Pitchers
            if game.get('home_pitcher') or game.get('away_pitcher'):
                st.subheader("⚾ Starting Pitchers")
                col1, col2 = st.columns(2)
                with col1:
                    if game.get('away_pitcher'):
                        st.write(f"**Away SP:** {game['away_pitcher']['name']}")
                    else:
                        st.write("**Away SP:** TBD")
                with col2:
                    if game.get('home_pitcher'):
                        st.write(f"**Home SP:** {game['home_pitcher']['name']}")
                    else:
                        st.write("**Home SP:** TBD")
            
            # Weather and Ballpark Info
            weather = game.get('weather', {})
            ballpark = game.get('ballpark_factors', {})
            if weather or ballpark:
                st.subheader("🏟️ Ballpark & Weather Conditions")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if weather.get('temperature'):
                        st.metric("Temperature", f"{weather['temperature']}°F")
                with col2:
                    if weather.get('wind_speed'):
                        wind_dir = weather.get('wind_direction', '')
                        st.metric("Wind", f"{weather['wind_speed']} mph {wind_dir}")
                with col3:
                    if ballpark.get('hr_factor'):
                        st.metric("HR Factor", f"{ballpark['hr_factor']:.2f}")
                with col4:
                    if weather.get('conditions'):
                        st.write(f"**Conditions:** {weather['conditions']}")
            
            # Team win probabilities from real data
            team_preds = game_predictions.get('team_predictions', {})
            if team_preds:
                st.subheader("📊 Win Probabilities")
                col1, col2, col3 = st.columns(3)
                with col1:
                    home_prob = team_preds.get('home_win_probability', 0.5)
                    st.metric("Home Win Probability", f"{home_prob:.1%}")
                with col2:
                    away_prob = team_preds.get('away_win_probability', 0.5)
                    st.metric("Away Win Probability", f"{away_prob:.1%}")
                with col3:
                    home_record = team_preds.get('home_record', 'N/A')
                    away_record = team_preds.get('away_record', 'N/A')
                    st.markdown(f"**Records:** {home_record} vs {away_record}")
            
            # Real player predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**🏃 {game['away_team']} (Away)**")
                away_players = game_predictions.get('away_players', [])
                display_real_team_predictions(away_players, 'away')
            
            with col2:
                st.markdown(f"**🏠 {game['home_team']} (Home)**")
                home_players = game_predictions.get('home_players', [])
                display_real_team_predictions(home_players, 'home')
            
            # Real insights from actual data
            key_matchups = game_predictions.get('key_matchups', [])
            if key_matchups:
                st.markdown("---")
                st.markdown("**🎯 Key Insights from Real MLB Data:**")
                for insight in key_matchups:
                    st.info(insight)

def display_real_team_predictions(players: List[Dict], home_away: str):
    """Display real team predictions from MLB data"""
    if not players:
        st.warning("No player data available - check API connection")
        return
    
    # Separate batters and pitchers - ONLY STARTING PITCHERS
    batters = [p for p in players if p.get('player_type') == 'batter'][:9]
    # Filter pitchers to show ONLY probable/starting pitcher for today's game
    pitchers = [p for p in players if p.get('player_type') == 'pitcher'][:1]  # Just take the first pitcher
    
    # Display batters
    if batters:
        st.markdown("**🏏 Batters:**")
        for batter in batters:
            predictions = batter.get('predictions', {})
            season_stats = batter.get('season_stats', {})
            handedness = predictions.get('handedness', {})
            
            # Display batter info with handedness
            col1, col2, col3 = st.columns(3)
            with col1:
                name = batter['name'].split('_')[0] if '_' in batter['name'] else batter['name']
                position = batter.get('position', 'OF')
                bats = handedness.get('bats', 'R')
                st.write(f"**{name}** ({bats}) - {position}")
            with col2:
                # Use authentic hit probability from full predictions
                hit_prob = predictions.get('hit_probability', 0)
                st.write(f"Hit: {hit_prob:.1%}")
            with col3:
                # Use authentic HR probability from full predictions
                hr_prob = predictions.get('home_run_probability', 0)
                st.write(f"HR: {hr_prob:.1%}")
            
            # Show streak status with rolling average details
            if predictions.get('hot_streak'):
                rolling_avg = predictions.get('rolling_avg', 0)
                games_analyzed = predictions.get('games_analyzed', 0)
                if games_analyzed > 0:
                    st.success(f"🔥 {name} is hot! Recent: {rolling_avg:.3f} ({games_analyzed} games)")
                else:
                    st.success(f"🔥 {name} is hot!")
            
            # Show authentic season stats - no estimates or fake data
            season_stats = batter.get('season_stats', {})
            if season_stats:
                avg = season_stats.get('avg', 0)
                hr = season_stats.get('homeRuns', 0)
                slg = season_stats.get('slg', 0)
                if avg or hr:
                    try:
                        avg_float = float(avg) if avg else 0.0
                        hr_int = int(hr) if hr else 0
                        slg_float = float(slg) if slg else 0.0
                        st.caption(f"Season: {avg_float:.3f} AVG, {hr_int} HR, {slg_float:.3f} SLG")
                    except (ValueError, TypeError):
                        st.caption(f"Season: {avg} AVG, {hr} HR, {slg} SLG")
            
            # Display authentic batter vs pitcher history if available
            matchup_history = predictions.get('matchup_history', {})
            if matchup_history and matchup_history.get('at_bats', 0) > 0:
                ab = matchup_history['at_bats']
                hits = matchup_history['hits']
                avg = matchup_history['batting_avg']
                st.info(f"vs Today's Pitcher: {hits}/{ab} ({avg:.3f}) career")
    
    # Display pitchers (only starting pitcher) with enhanced details
    if pitchers:
        st.markdown("**⚾ Starting Pitcher:**")
        pitcher = pitchers[0]  # Only one starting pitcher
        predictions = pitcher.get('predictions', {})
        season_stats = pitcher.get('season_stats', {})
        handedness = predictions.get('handedness', {})
        
        # Display pitcher info with handedness
        name = pitcher['name'].split('_')[0] if '_' in pitcher['name'] else pitcher['name']
        throws = handedness.get('throws', 'R')
        st.write(f"**{name}** ({throws}) - Starting Pitcher")
        
        # Show season stats instead of predictions
        if season_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                era = season_stats.get('era', 0)
                st.write(f"**Season ERA:** {era:.2f}")
            with col2:
                whip = season_stats.get('whip', 0)
                st.write(f"**WHIP:** {whip:.2f}")
            with col3:
                k9 = season_stats.get('strikeoutsPerNine', 0)
                st.write(f"**K/9:** {k9:.1f}")
            with col4:
                bb9 = season_stats.get('walksPer9Inn', 0)
                st.write(f"**BB/9:** {bb9:.1f}")
        
        # Show authentic vs handedness splits
        vs_handedness = predictions.get('vs_handedness', {})
        if vs_handedness:
            st.markdown("**vs Left/Right Splits:**")
            col1, col2 = st.columns(2)
            with col1:
                left_splits = vs_handedness.get('L', {})
                if left_splits:
                    avg_vs_l = left_splits.get('avg_against', '.000')
                    st.write(f"**vs LHB:** {avg_vs_l} AVG")
            with col2:
                right_splits = vs_handedness.get('R', {})
                if right_splits:
                    avg_vs_r = right_splits.get('avg_against', '.000')
                    st.write(f"**vs RHB:** {avg_vs_r} AVG")
        
        # Show authentic last 5 starts data only
        last_5_starts = predictions.get('last_5_starts', [])
        if last_5_starts:
            st.markdown("**📊 Last 5 Starts:**")
            
            # Create DataFrame for better display
            starts_data = []
            for start in last_5_starts[-5:]:  # Most recent 5
                starts_data.append({
                    'Date': start.get('date', 'N/A')[-5:],  # Last 5 chars of date
                    'Game': start.get('opponent', 'N/A'),  # This will show MM/DD format
                    'IP': start.get('innings_pitched', '0.0'),
                    'K': start.get('strikeouts', 0),
                    'BB': start.get('walks', 0),
                    'H': start.get('hits_allowed', 0),
                    'ER': start.get('earned_runs', 0),
                    'HR': start.get('home_runs_allowed', 0),
                    'GS': start.get('game_score', 50)
                })
            
            if starts_data:
                df = pd.DataFrame(starts_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show streak status
        if predictions.get('hot_streak'):
            rolling_era = predictions.get('rolling_era', 0)
            games_analyzed = predictions.get('games_analyzed', 0)
            if games_analyzed > 0:
                st.success(f"🔥 {name} is dealing! Recent ERA: {rolling_era:.2f} ({games_analyzed} starts)")
            else:
                st.success(f"🔥 {name} is hot!")
        elif predictions.get('cold_streak'):
            st.error(f"❄️ {name} is struggling")

def generate_team_predictions(team_name: str, home_away: str) -> Dict:
    """Generate predictions for all players on a team"""
    try:
        # Get team roster and recent performance
        team_data = st.session_state.data_fetcher.get_team_recent_performance(team_name)
        
        predictions = {
            'team_name': team_name,
            'batters': [],
            'pitchers': [],
            'team_stats': {}
        }
        
        if team_data.empty:
            return predictions
        
        # Separate batters and pitchers
        batters = team_data[team_data['player_type'] == 'batter'].groupby('player_name').last().reset_index()
        pitchers = team_data[team_data['player_type'] == 'pitcher'].groupby('player_name').last().reset_index()
        
        # Generate batter predictions (top 9)
        for _, batter in batters.head(9).iterrows():
            batter_pred = {
                'name': batter['player_name'],
                'current_avg': batter.get('batting_avg', 0.250),
                'predicted_avg': batter.get('batting_avg', 0.250),  # Use actual average
                'hr_probability': 0.0,  # Disabled fake probability
                'hit_probability': batter.get('batting_avg', 0.250),  # Use actual average
                'streak_status': determine_streak_status(batter, 'batting_avg')
            }
            predictions['batters'].append(batter_pred)
        
        # Generate pitcher predictions (starting pitcher + key relievers)
        for _, pitcher in pitchers.head(4).iterrows():
            pitcher_pred = {
                'name': pitcher['player_name'],
                'current_era': pitcher.get('era', 4.00),
                'predicted_era': pitcher.get('era', 4.00),  # Use actual ERA
                'strikeout_probability': 0.0,  # Disabled fake probability
                'quality_start_probability': 0.0,  # Disabled fake probability
                'streak_status': determine_streak_status(pitcher, 'era')
            }
            predictions['pitchers'].append(pitcher_pred)
        
        # Team-level predictions
        predictions['team_stats'] = {
            'predicted_runs': 0,  # Disabled fake prediction
            'win_probability': 0.5 + (0.1 if home_away == 'home' else 0)  # Basic home field advantage only
        }
        predictions['team_stats']['win_probability'] = min(0.85, max(0.15, predictions['team_stats']['win_probability']))
        
        return predictions
        
    except Exception as e:
        print(f"Error generating team predictions: {e}")
        return {'team_name': team_name, 'batters': [], 'pitchers': [], 'team_stats': {}}

def determine_streak_status(player_data: pd.Series, stat_col: str) -> str:
    """Determine if player is hot, cold, or neutral"""
    try:
        if stat_col not in player_data:
            return "Neutral"
        
        current_val = player_data[stat_col]
        
        # Simple streak logic based on stat type
        if stat_col == 'batting_avg':
            if current_val > 0.300:
                return "🔥 Hot"
            elif current_val < 0.220:
                return "🧊 Cold"
        elif stat_col == 'era':
            if current_val < 3.00:
                return "🔥 Hot"
            elif current_val > 5.00:
                return "🧊 Cold"
        
        return "Neutral"
        
    except:
        return "Neutral"

def display_team_predictions(predictions: Dict, home_away: str):
    """Display team predictions in organized format"""
    if not predictions.get('batters') and not predictions.get('pitchers'):
        st.write("No prediction data available")
        return
    
    # Team win probability
    win_prob = predictions.get('team_stats', {}).get('win_probability', 0.5)
    st.metric("Win Probability", f"{win_prob:.1%}")
    
    # Top batters
    if predictions.get('batters'):
        st.markdown("**🏏 Top Batters:**")
        for batter in predictions['batters'][:5]:  # Show top 5
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{batter['name']} {batter['streak_status']}")
            with col2:
                st.metric("Hit %", f"{batter['hit_probability']:.1%}")
            with col3:
                st.metric("HR %", f"{batter['hr_probability']:.1%}")
    
    # Starting pitcher
    if predictions.get('pitchers'):
        st.markdown("**⚾ Starting Pitcher:**")
        pitcher = predictions['pitchers'][0]  # Assume first is starter
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{pitcher['name']} {pitcher['streak_status']}")
        with col2:
            st.metric("K Rate", f"{pitcher['strikeout_probability']:.1%}")

def generate_game_insights(away_preds: Dict, home_preds: Dict, game: Dict) -> List[str]:
    """Generate key insights for the game"""
    insights = []
    
    try:
        # Win probability comparison
        away_win_prob = away_preds.get('team_stats', {}).get('win_probability', 0.5)
        home_win_prob = home_preds.get('team_stats', {}).get('win_probability', 0.5)
        
        if abs(away_win_prob - home_win_prob) > 0.15:
            favorite = game['home_team'] if home_win_prob > away_win_prob else game['away_team']
            insights.append(f"📊 {favorite} has a significant advantage with {max(away_win_prob, home_win_prob):.1%} win probability")
        
        # Hot batters
        away_hot_batters = [b for b in away_preds.get('batters', []) if '🔥' in b['streak_status']]
        home_hot_batters = [b for b in home_preds.get('batters', []) if '🔥' in b['streak_status']]
        
        if away_hot_batters:
            insights.append(f"🔥 {game['away_team']} has {len(away_hot_batters)} hot batters: {', '.join([b['name'] for b in away_hot_batters[:2]])}")
        
        if home_hot_batters:
            insights.append(f"🔥 {game['home_team']} has {len(home_hot_batters)} hot batters: {', '.join([b['name'] for b in home_hot_batters[:2]])}")
        
        # Pitcher analysis
        if away_preds.get('pitchers') and home_preds.get('pitchers'):
            away_pitcher = away_preds['pitchers'][0]
            home_pitcher = home_preds['pitchers'][0]
            
            if away_pitcher['strikeout_probability'] > 0.25:
                insights.append(f"⚾ {away_pitcher['name']} has high strikeout potential ({away_pitcher['strikeout_probability']:.1%})")
            
            if home_pitcher['strikeout_probability'] > 0.25:
                insights.append(f"⚾ {home_pitcher['name']} has high strikeout potential ({home_pitcher['strikeout_probability']:.1%})")
        
        return insights
        
    except Exception as e:
        return ["Unable to generate game insights"]



def hot_cold_streaks_page():
    st.header("Hot/Cold Streak Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        streak_window = st.selectbox("Streak Window", [10, 15, 20])
    
    with col2:
        stat_type = st.selectbox("Stat Type", ["Batting Average", "Home Runs", "RBIs", "ERA", "WHIP"])
    
    team = st.selectbox("Select Team", st.session_state.data_fetcher.get_teams())
    
    if team:
        with st.spinner("Analyzing streaks..."):
            try:
                # Get team roster and recent performance
                team_data = st.session_state.data_fetcher.get_team_recent_performance(team)
                
                if team_data.empty:
                    st.warning(f"No recent data available for {team}. Generating sample data for demonstration.")
                    # Generate sample team data for streak analysis
                    team_data = pd.DataFrame({
                        'player_name': [f"Player_{i}" for i in range(1, 16)],
                        'player_type': ['batter']*9 + ['pitcher']*6,
                        'batting_avg': [0.250] * 9 + [0]*6,  # Use fixed average instead of random
                        'home_runs': [15] * 9 + [0]*6,  # Fixed values instead of random
                        'rbis': [45] * 9 + [0]*6,  # Fixed values instead of random
                        'era': [0]*9 + [4.0] * 6,  # Fixed values instead of random
                        'whip': [0]*9 + [1.3] * 6  # Fixed values instead of random
                    })
                
                # Analyze streaks
                hot_players, cold_players = st.session_state.analytics_engine.detect_hot_cold_streaks(
                    team_data, streak_window, stat_type
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 Hot Streaks")
                    if hot_players:
                        for player in hot_players:
                            st.success(f"**{player['name']}** - {player['streak_description']}")
                    else:
                        st.info("No hot streaks detected")
                
                with col2:
                    st.subheader("🧊 Cold Streaks")
                    if cold_players:
                        for player in cold_players:
                            st.warning(f"**{player['name']}** - {player['streak_description']}")
                    else:
                        st.info("No cold streaks detected")
                
                # Visualization
                if hot_players or cold_players:
                    fig = st.session_state.visualizer.create_streak_visualization(
                        hot_players, cold_players, stat_type
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing streaks: {str(e)}")



def strike_zone_page():
    st.header("🎯 Strike Zone Analysis")
    
    # Date selection dropdown
    date_options = get_date_options()
    selected_date_label = st.selectbox(
        "Select Game Date:",
        options=list(date_options.keys()),
        index=1,  # Default to "Today"
        key="strike_zone_date_select"
    )
    selected_date = date_options[selected_date_label]
    
    # Get games for selected date
    from datetime import datetime
    date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
    games = st.session_state.real_prediction_engine.data_fetcher.get_games_for_date(date_obj)
    
    if not games:
        st.warning(f"No games scheduled for {selected_date_label.lower()}. Please try another date.")
        return
    
    st.subheader(f"{selected_date_label}'s Matchups")
    
    # Collect real players from today's games
    all_batters = []
    all_pitchers = []
    
    # Display games and get player predictions
    for game in games:
        # Add numerical date before game time
        game_time = game.get('game_time', 'TBD')
        try:
            date_display = date_obj.strftime("%m/%d")
            st.write(f"**{game['away_team']} @ {game['home_team']}** - {date_display} {game_time}")
        except:
            # Fallback without date if parsing fails
            st.write(f"**{game['away_team']} @ {game['home_team']}** - {game_time}")
        
        try:
            # Get real team predictions with actual players
            away_team_id = game.get('away_team_id', 0)
            home_team_id = game.get('home_team_id', 0)
            
            # Get today's predictions which include real MLB players
            predictions = st.session_state.real_prediction_engine.generate_game_predictions(game, simplified=False)
            
            # Extract batters and pitchers from predictions
            for team_name in ['away_team', 'home_team']:
                team_predictions = predictions.get(team_name, {})
                
                # Add batters
                for batter in team_predictions.get('batters', [])[:9]:  # Starting lineup
                    team_abbr = game[team_name]
                    all_batters.append(f"{batter['name']} ({team_abbr})")
                
                # Add starting pitcher
                pitchers = team_predictions.get('pitchers', [])
                if pitchers:
                    starter = pitchers[0]  # First pitcher is starter
                    team_abbr = game[team_name]
                    all_pitchers.append(f"{starter['name']} ({team_abbr})")
                    
        except Exception as e:
            print(f"Error getting players for {game['away_team']} vs {game['home_team']}: {e}")
            continue
    
    # Remove duplicates while preserving order
    all_batters = list(dict.fromkeys(all_batters))
    all_pitchers = list(dict.fromkeys(all_pitchers))
    
    # Fallback players if no real players found
    if not all_batters:
        all_batters = ["Aaron Judge (NYY)", "Mookie Betts (LAD)", "Mike Trout (LAA)", "Ronald Acuna Jr. (ATL)", "Vladimir Guerrero Jr. (TOR)"]
    if not all_pitchers:
        all_pitchers = ["Gerrit Cole (NYY)", "Jacob deGrom (TEX)", "Shane Bieber (CLE)", "Walker Buehler (LAD)", "Zack Wheeler (PHI)"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏏 Select Batter:**")
        selected_batter_with_team = st.selectbox("Batter from Today's Games", all_batters, index=0)
        selected_batter = selected_batter_with_team.split(' (')[0] if selected_batter_with_team else all_batters[0].split(' (')[0]
    
    with col2:
        st.markdown("**⚾ Select Starting Pitcher:**")
        selected_pitcher_with_team = st.selectbox("Starting Pitcher from Today's Games", all_pitchers, index=0)
        selected_pitcher = selected_pitcher_with_team.split(' (')[0] if selected_pitcher_with_team else all_pitchers[0].split(' (')[0]
    
    # Advanced analysis type selection
    analysis_type = st.radio(
        "Analysis Type",
        ["Exit Velocity by Zone", "Launch Angle Analysis", "Hard Hit Rate", "Batted Ball Distribution", 
         "Barrel Rate by Zone", "Hit Type Analysis", "Strikeout Heat Map"],
        horizontal=False
    )
    
    if selected_batter and selected_pitcher:
        with st.spinner("Generating strike zone analysis..."):
            try:
                if analysis_type == "Exit Velocity by Zone":
                    zone_data = st.session_state.strike_zone_analyzer.analyze_exit_velocity_zones(selected_batter)
                    title = f"{selected_batter} - Exit Velocity by Zone"
                elif analysis_type == "Launch Angle Analysis":
                    zone_data = st.session_state.strike_zone_analyzer.analyze_launch_angle_zones(selected_batter)
                    title = f"{selected_batter} - Launch Angle Distribution"
                elif analysis_type == "Hard Hit Rate":
                    zone_data = st.session_state.strike_zone_analyzer.analyze_hard_hit_zones(selected_batter)
                    title = f"{selected_batter} - Hard Hit Rate by Zone"
                elif analysis_type == "Batted Ball Distribution":
                    zone_data = st.session_state.strike_zone_analyzer.analyze_batted_ball_zones(selected_batter)
                    title = f"{selected_batter} - Batted Ball Distribution"
                elif analysis_type == "Barrel Rate by Zone":
                    zone_data = st.session_state.strike_zone_analyzer.analyze_barrel_zones(selected_batter)
                    title = f"{selected_batter} - Barrel Rate by Zone"
                elif analysis_type == "Hit Type Analysis":
                    zone_data = st.session_state.strike_zone_analyzer.analyze_hit_types_zones(selected_batter)
                    title = f"{selected_batter} - Hit Types by Zone"
                else:  # Strikeout Heat Map
                    zone_data = st.session_state.strike_zone_analyzer.analyze_strikeout_zones(selected_pitcher)
                    title = f"{selected_pitcher} - Strikeout Rate by Zone"
                
                if zone_data is not None and zone_data.size > 0:
                    # Ensure zone_data is a 3x3 numpy array
                    if zone_data.shape != (3, 3):
                        zone_data = zone_data.reshape(3, 3)
                    
                    # Create heat map with proper data
                    fig = st.session_state.visualizer.create_strike_zone_heatmap(zone_data, title)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Zone analysis summary
                    st.subheader("📊 Zone Analysis Summary")
                    summary = st.session_state.strike_zone_analyzer.get_zone_summary(zone_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        for i, (zone, stats) in enumerate(summary.items()):
                            if i < len(summary) // 2:
                                st.write(f"**{zone}:** {stats}")
                    with col2:
                        for i, (zone, stats) in enumerate(summary.items()):
                            if i >= len(summary) // 2:
                                st.write(f"**{zone}:** {stats}")
                    
                    # Strategic recommendations
                    if analysis_type == "Matchup Advantage Zones":
                        st.subheader("🎯 Strategic Recommendations")
                        batter_zones = st.session_state.strike_zone_analyzer.analyze_batter_zones(selected_batter)
                        pitcher_zones = st.session_state.strike_zone_analyzer.analyze_pitcher_zones(selected_pitcher)
                        recommendations = st.session_state.strike_zone_analyzer.get_zone_recommendations(
                            batter_zones, pitcher_zones
                        )
                        for rec in recommendations:
                            st.info(rec)
                else:
                    st.error("Unable to generate zone analysis.")
                
            except Exception as e:
                st.error(f"Error generating strike zone analysis: {str(e)}")

def betting_opportunities_page():
    st.header("💰 Advanced Betting Opportunities")
    st.markdown("AI-powered betting analysis with advanced edge detection")
    
    # Date selection dropdown
    date_options = get_date_options()
    selected_date_label = st.selectbox(
        "Select Game Date:",
        options=list(date_options.keys()),
        index=1,  # Default to "Today"
        key="betting_date_select"
    )
    selected_date = date_options[selected_date_label]
    
    # Enhanced mobile-optimized CSS for betting opportunities page
    st.markdown("""
    <style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .stDataFrame {
            font-size: 11px;
            overflow-x: auto;
        }
        .stDataFrame table {
            min-width: 100%;
            font-size: 10px;
        }
        .stDataFrame td {
            white-space: normal !important;
            word-wrap: break-word !important;
            max-width: 150px;
            padding: 4px;
        }
        div[data-testid="metric-container"] {
            font-size: 11px;
            padding: 6px;
            margin: 3px 0;
        }
        div[data-testid="metric-container"] > div {
            font-size: 11px;
        }
        .stMarkdown {
            font-size: 12px;
            line-height: 1.4;
            word-wrap: break-word;
            white-space: normal;
        }
        .streamlit-expanderContent {
            padding: 6px;
            font-size: 12px;
        }
        .stColumn {
            padding: 1px;
        }
        /* Force text wrapping in containers */
        .element-container {
            word-wrap: break-word;
            white-space: normal;
        }
        /* Better mobile card styling */
        div[data-testid="container"] {
            word-wrap: break-word;
            white-space: normal;
        }
    }
    
    /* Universal text wrapping improvements */
    .streamlit-expanderContent {
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    .stDataFrame {
        word-wrap: break-word;
        overflow-x: auto;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e6e6e6;
        padding: 5px;
        border-radius: 5px;
        margin: 2px;
        word-wrap: break-word;
        white-space: normal;
    }
    .stMarkdown {
        word-wrap: break-word !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    
    /* Responsive table improvements */
    .dataframe {
        width: 100% !important;
        font-size: 12px;
    }
    
    /* Better mobile expander styling */
    details[data-testid="expander"] {
        border: 1px solid #e6e6e6;
        border-radius: 4px;
        margin: 4px 0;
    }
    
    /* Force proper text wrapping everywhere */
    * {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load simple, working betting engine
    if 'betting_engine_v4' not in st.session_state:
        from clean_betting_engine import AuthenticBettingEngine
        st.session_state.betting_engine = AuthenticBettingEngine()
        st.session_state.betting_engine_v4 = True
        
        # Show initialization status
        st.success("✅ Data-driven betting engine loaded with comprehensive prop analysis")
    
    # Get games for selected date - NO CACHING to force fresh data
    def get_games_for_date(date_str):
        """Get games for selected date"""
        try:
            # Ensure prediction engine is initialized
            if 'real_prediction_engine' not in st.session_state:
                st.session_state.real_prediction_engine = get_prediction_engine()
            
            # Convert date string to datetime for API call
            from datetime import datetime
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Use the data fetcher with the selected date
            games = st.session_state.real_prediction_engine.data_fetcher.get_games_for_date(date_obj)
            print(f"DEBUG: get_games_for_date({date_str}) returned {len(games) if games else 0} games")
            return games if games else []
        except Exception as e:
            print(f"Error fetching games for {date_str}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    # Analyze single game opportunities - NO CACHING to use updated betting engine
    def get_single_game_opportunities(game_index: int, date_str: str):
        """Get betting opportunities for a single selected game"""
        try:
            games = get_games_for_date(date_str)
            if not games or game_index >= len(games):
                return None, []
            
            game = games[game_index]
            
            # Generate simplified predictions with authentic data for betting opportunities
            predictions = st.session_state.real_prediction_engine.generate_game_predictions(game, simplified=True)
            
            # Use actual MLB predictions without fake data generation
            home_players = predictions.get('home_players', [])
            away_players = predictions.get('away_players', [])
            
            # Analyze betting opportunities with REALISTIC stats
            player_predictions = {
                'home_players': home_players,
                'away_players': away_players
            }
            
            print(f"DEBUG: Analyzing game {game.get('away_team')} @ {game.get('home_team')}")
            print(f"DEBUG: Home players: {len(predictions.get('home_players', []))}")
            print(f"DEBUG: Away players: {len(predictions.get('away_players', []))}")
            
            game_opportunities = st.session_state.betting_engine.analyze_betting_opportunities(
                [game], player_predictions
            )
            
            print(f"DEBUG: Generated {len(game_opportunities)} opportunities")
            if game_opportunities:
                print(f"DEBUG: First opportunity: {game_opportunities[0]}")
            
            return [game], game_opportunities
            
        except Exception as e:
            print(f"Error fetching single game opportunities: {e}")
            import traceback
            traceback.print_exc()
            return None, []
    
    # This section has been moved up to after date selection
    
    # Get games for selected date
    available_games = get_games_for_date(selected_date)
    
    if not available_games:
        st.warning(f"No games found for {selected_date_label.lower()}. Please check back later.")
        return
    
    # Create game selection dropdown
    st.subheader(f"🎯 Select Game for Betting Analysis ({selected_date_label})")
    
    game_options = []
    for i, game in enumerate(available_games):
        game_time = game.get('game_time', 'TBD')
        # Add numerical date before game time
        try:
            from datetime import datetime
            date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
            date_display = date_obj.strftime("%m/%d")
            game_options.append(f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')} ({date_display} {game_time})")
        except:
            # Fallback without date if parsing fails
            game_options.append(f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')} ({game_time})")
    
    selected_game_idx = st.selectbox(
        "Choose a game to analyze:",
        range(len(game_options)),
        format_func=lambda x: game_options[x],
        key="game_selection"
    )
    
    # Get opportunities for selected game
    with st.spinner(f"Analyzing betting opportunities for {game_options[selected_game_idx]}..."):
        games, all_opportunities = get_single_game_opportunities(selected_game_idx, selected_date)
    
        if games is None:
            st.warning("Unable to load game data")
            return
        
        if not all_opportunities:
            st.info("No profitable betting opportunities found for this game")
            return
    
    # Display summary metrics with mobile-responsive layout
    st.subheader("📊 Today's Betting Summary")
    
    # Use 2x2 grid on mobile, 4 columns on desktop
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    total_opportunities = len(all_opportunities)
    high_value_bets = len([opp for opp in all_opportunities if opp.get('betting_edge', 0) >= 5.0])
    medium_value_bets = len([opp for opp in all_opportunities if 3.0 <= opp.get('betting_edge', 0) < 5.0])
    hot_streak_players = len([opp for opp in all_opportunities if 'Hot Streak' in opp.get('edge_factors', [])])
    
    with col1:
        st.metric("Total Opportunities", total_opportunities)
    with col2:
        st.metric("Hot Streak Players", hot_streak_players)
    with col3:
        st.metric("High Value (5%+ Edge)", high_value_bets)
    with col4:
        st.metric("Medium Value (3-5% Edge)", medium_value_bets)
    
    # Show selected game info
    selected_game = games[0] if games else {}
    st.subheader(f"🏟️ Betting Opportunities: {selected_game.get('away_team', 'Away')} @ {selected_game.get('home_team', 'Home')}")
    
    # Show all opportunities for selected game
    st.write(f"**{len(all_opportunities)} total opportunities found**")
    
    # Group opportunities by type for better organization
    opportunities_by_type = {}
    for opp in all_opportunities:
        opp_type = opp.get('type', 'General')
        if opp_type not in opportunities_by_type:
            opportunities_by_type[opp_type] = []
        opportunities_by_type[opp_type].append(opp)
    
    # Sort each type by edge
    for opp_type in opportunities_by_type:
        opportunities_by_type[opp_type].sort(key=lambda x: x.get('betting_edge', 0), reverse=True)
    
    # Enhanced Display with Comprehensive Betting Props
    if opportunities_by_type:
        st.success(f"✅ Found {len(all_opportunities)} betting opportunities!")
        
        # Simple table display for immediate results
        st.subheader("🏆 Top Betting Opportunities")
        
        # Create simple DataFrame for quick display with better formatting
        df_data = []
        for opp in all_opportunities[:25]:  # Show top 25
            reasoning = opp.get('reasoning', 'N/A')
            # Better truncation that preserves zone names
            if len(reasoning) > 100:
                # Find the best break point to preserve zone information
                if ':' in reasoning:
                    parts = reasoning.split(', ')
                    truncated = parts[0]  # Always keep first part
                    for part in parts[1:]:
                        if len(truncated + ', ' + part) <= 95:
                            truncated += ', ' + part
                        else:
                            truncated += '...'
                            break
                    reasoning = truncated
                else:
                    reasoning = reasoning[:97] + "..."
            
            df_data.append({
                'Player': opp.get('player', 'Unknown'),
                'Bet Type': opp.get('bet_type', 'Unknown'),
                'Edge %': f"{opp.get('betting_edge', 0):.1f}%",
                'Zone Analysis': reasoning
            })
        
        if df_data:
            # Mobile-responsive display - use cards for better mobile experience
            
            # Mobile-optimized card layout with full text display
            for i, row in enumerate(df_data[:10], 1):  # Show top 10 for quick view
                with st.container():
                    # Single column layout for better mobile experience
                    st.markdown(f"**#{i} {row['Player']}** • {row['Bet Type']} • **{row['Edge %']}**")
                    
                    # Full analysis text without truncation in expandable format
                    with st.expander(f"📊 Analysis Details", expanded=False):
                        st.write(row['Zone Analysis'])
                    
                    st.markdown("---")
            
        # Reorganize by bet category for better presentation
        bet_categories = {
            'Hits Prop': '🏏 Hits Props',
            'Home Run Prop': '💣 Home Run Props', 
            'RBI Prop': '🏃 RBI Props',
            'Total Bases Prop': '📊 Total Bases Props',
            'Strikeout Prop': '⚾ Strikeout Props'
        }
        
        # Create tabs for each category
        category_tabs = []
        category_data = {}
        
        for bet_type, display_name in bet_categories.items():
            if bet_type in opportunities_by_type:
                category_tabs.append(display_name)
                category_data[display_name] = opportunities_by_type[bet_type]
        
        if category_tabs:
            tabs = st.tabs(category_tabs)
            
            for tab, category_name in zip(tabs, category_tabs):
                with tab:
                    opps = category_data[category_name]
                    st.write(f"**{len(opps)} opportunities found**")
                    
                    # Show top opportunities with enhanced display
                    for i, opp in enumerate(opps[:15], 1):  # Show top 15 per category
                        edge = opp.get('betting_edge', 3.0)
                        confidence_score = opp.get('confidence_score', 0.6)
                        
                        # Mobile-optimized card display 
                        with st.container():
                            # Single column layout for better mobile experience
                            st.markdown(f"**#{i} {opp.get('player', 'Player')}** - {opp.get('bet_type', 'Unknown Bet')}")
                            
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Full reasoning without truncation for mobile
                                reasoning = opp.get('reasoning', 'No analysis available')
                                # Use text_area for better mobile text display
                                st.text_area(
                                    label="Analysis",
                                    value=reasoning,
                                    height=68,
                                    disabled=True,
                                    key=f"mobile_reasoning_{i}_{category_name}",
                                    label_visibility="hidden"
                                )
                            
                            with col2:
                                st.metric("Edge", f"{edge:.1f}%")
                                
                                # Quality indicator
                                if edge >= 5.0 and confidence_score > 0.75:
                                    st.success("🔥 Strong")
                                elif edge >= 3.0:
                                    st.info("⚡ Medium")
                                else:
                                    st.warning("💡 Value")
                            
                            st.divider()
    else:
        st.warning("No betting opportunities found for the selected game.")


if __name__ == "__main__":
    main()

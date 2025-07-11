import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_fetcher import MLBDataFetcher
from analytics_engine import AnalyticsEngine
from visualization import Visualizer
from ml_predictor import MLPredictor
from strike_zone_analyzer import StrikeZoneAnalyzer
from real_prediction_engine import RealMLBPredictionEngine
from advanced_betting_engine import AdvancedBettingEngine

# Page configuration
st.set_page_config(
    page_title="MLB Analytics Pro",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = MLBDataFetcher()
if 'analytics_engine' not in st.session_state:
    st.session_state.analytics_engine = AnalyticsEngine()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = MLPredictor()
if 'strike_zone_analyzer' not in st.session_state:
    st.session_state.strike_zone_analyzer = StrikeZoneAnalyzer()
if 'real_prediction_engine' not in st.session_state:
    st.session_state.real_prediction_engine = RealMLBPredictionEngine()
if 'betting_engine' not in st.session_state:
    st.session_state.betting_engine = AdvancedBettingEngine()

def main():
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
    st.header("📅 Today's Real MLB Games & Predictions")
    
    # Show data source status with timeout handling
    try:
        data_status = st.session_state.real_prediction_engine.get_data_source_status()
        st.markdown(f"**Data Source:** {data_status}")
    except Exception as e:
        st.markdown("**Data Source:** ⚠️ API connection issues - using sample data")
    
    # Get today's games
    today = datetime.now().date()
    
    with st.spinner("Loading real MLB games from MLB Stats API..."):
        games = st.session_state.real_prediction_engine.data_fetcher.get_todays_games()
    
    if not games:
        st.warning("No MLB games scheduled for today. Check your internet connection.")
        st.info("This app connects to MLB Stats API (statsapi.mlb.com) and Baseball Savant for real data.")
        return
    
    st.subheader(f"Real Games for {today.strftime('%B %d, %Y')} - {len(games)} games")
    
    # Display each game with real predictions
    for i, game in enumerate(games):
        game_title = f"🏟️ {game['away_team']} @ {game['home_team']}"
        if game.get('game_time'):
            try:
                # Convert UTC time to Central Time
                utc_time = datetime.fromisoformat(game['game_time'].replace('Z', '+00:00'))
                central_tz = pytz.timezone('US/Central')
                central_time = utc_time.astimezone(central_tz)
                game_title += f" ({central_time.strftime('%I:%M %p CT')})"
            except:
                pass
        
        # Add venue info if available
        if game.get('venue'):
            game_title += f" - {game['venue']}"
        
        # Show expanded for first 3 games, collapsed for rest
        expanded = i < 3
        with st.expander(game_title, expanded=expanded):
            with st.spinner("Generating real predictions from MLB data..."):
                game_predictions = st.session_state.real_prediction_engine.generate_game_predictions(game)
            
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
                hit_prob = predictions.get('hit_probability', 0)
                st.write(f"Hit: {hit_prob:.1%}")
            with col3:
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
            
            # Show season stats if available
            if season_stats:
                avg = season_stats.get('avg', 0)
                hr = season_stats.get('homeRuns', 0)
                if avg or hr:
                    try:
                        avg_float = float(avg) if avg else 0.0
                        hr_int = int(hr) if hr else 0
                        st.caption(f"Season: {avg_float:.3f} AVG, {hr_int} HR")
                    except (ValueError, TypeError):
                        st.caption(f"Season: {avg} AVG, {hr} HR")
            
            # Display batter vs pitcher history if available
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
        
        # Pitching predictions in columns
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            k_proj = predictions.get('predicted_strikeouts', 0)
            st.write(f"**K's:** {k_proj:.1f}")
        with col2:
            bb_proj = predictions.get('predicted_walks', 0)
            st.write(f"**BB's:** {bb_proj:.1f}")
        with col3:
            era_proj = predictions.get('predicted_era', 0)
            st.write(f"**ERA:** {era_proj:.2f}")
        with col4:
            er_proj = predictions.get('predicted_earned_runs', 0)
            st.write(f"**ER:** {er_proj:.1f}")
        
        # Additional pitcher stats
        col1, col2 = st.columns(2)
        with col1:
            hits_proj = predictions.get('predicted_hits_allowed', 0)
            st.write(f"**Hits Allowed:** {hits_proj:.1f}")
        with col2:
            whip_proj = predictions.get('predicted_whip', 0)
            if whip_proj > 0:
                st.write(f"**WHIP:** {whip_proj:.2f}")
        
        # Show vs handedness splits (simplified to avoid API errors)
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
        
        # Last 5 starts
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
                'predicted_avg': min(0.450, max(0.150, batter.get('batting_avg', 0.250) + np.random.normal(0, 0.020))),
                'hr_probability': min(0.15, max(0.01, np.random.beta(2, 20))),
                'hit_probability': min(0.45, max(0.15, batter.get('batting_avg', 0.250) + np.random.normal(0, 0.030))),
                'streak_status': determine_streak_status(batter, 'batting_avg')
            }
            predictions['batters'].append(batter_pred)
        
        # Generate pitcher predictions (starting pitcher + key relievers)
        for _, pitcher in pitchers.head(4).iterrows():
            pitcher_pred = {
                'name': pitcher['player_name'],
                'current_era': pitcher.get('era', 4.00),
                'predicted_era': min(8.00, max(1.00, pitcher.get('era', 4.00) + np.random.normal(0, 0.200))),
                'strikeout_probability': min(0.35, max(0.10, np.random.beta(8, 12))),
                'quality_start_probability': min(0.70, max(0.20, np.random.beta(6, 4))),
                'streak_status': determine_streak_status(pitcher, 'era')
            }
            predictions['pitchers'].append(pitcher_pred)
        
        # Team-level predictions
        predictions['team_stats'] = {
            'predicted_runs': max(0, np.random.poisson(4.5)),
            'win_probability': 0.5 + (0.1 if home_away == 'home' else 0) + np.random.normal(0, 0.15)
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
                        'batting_avg': [0.250 + np.random.normal(0, 0.050) for _ in range(9)] + [0]*6,
                        'home_runs': [15 + np.random.randint(-5, 10) for _ in range(9)] + [0]*6,
                        'rbis': [45 + np.random.randint(-15, 20) for _ in range(9)] + [0]*6,
                        'era': [0]*9 + [4.0 + np.random.normal(0, 1.0) for _ in range(6)],
                        'whip': [0]*9 + [1.3 + np.random.normal(0, 0.2) for _ in range(6)]
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
    
    # Get today's games and real player data
    today = datetime.now().date()
    games = st.session_state.real_prediction_engine.data_fetcher.get_todays_games()
    
    if not games:
        st.warning("No games scheduled today. Using recent games for analysis.")
        # Get sample games for analysis
        from datetime import timedelta
        yesterday = today - timedelta(days=1)
        games = st.session_state.real_prediction_engine.data_fetcher.get_todays_games()
    
    st.subheader("Today's Matchups")
    
    # Collect real players from today's games
    all_batters = []
    all_pitchers = []
    
    # Display games and get player predictions
    for game in games:
        st.write(f"**{game['away_team']} @ {game['home_team']}** - {game.get('game_time', 'TBD')}")
        
        try:
            # Get real team predictions with actual players
            away_team_id = game.get('away_team_id', 0)
            home_team_id = game.get('home_team_id', 0)
            
            # Get today's predictions which include real MLB players
            predictions = st.session_state.real_prediction_engine.generate_game_predictions(game)
            
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
    
    # Get today's games and predictions
    with st.spinner("Analyzing betting opportunities across today's games..."):
        try:
            games = st.session_state.real_prediction_engine.data_fetcher.get_todays_games()
            
            if not games:
                st.warning("No MLB games found for today")
                return
            
            # Get predictions for all games
            all_predictions = {}
            for game in games:
                try:
                    predictions = st.session_state.real_prediction_engine.generate_game_predictions(game)
                    all_predictions[f"{game['away_team']}@{game['home_team']}"] = {
                        'game': game,
                        'predictions': predictions
                    }
                except Exception as e:
                    print(f"Error getting predictions for {game['away_team']} @ {game['home_team']}: {e}")
                    continue
            
            # Analyze betting opportunities
            all_opportunities = []
            for game_key, data in all_predictions.items():
                try:
                    # Pass the predictions dict with team player data
                    player_predictions = {
                        'home_players': data['predictions'].get('home_players', []),
                        'away_players': data['predictions'].get('away_players', [])
                    }
                    
                    game_opportunities = st.session_state.betting_engine.analyze_betting_opportunities(
                        [data['game']], player_predictions
                    )
                    all_opportunities.extend(game_opportunities)
                    
                    # Debug output to see what we're getting
                    if game_opportunities:
                        print(f"Found {len(game_opportunities)} opportunities for {game_key}")
                    else:
                        print(f"No opportunities found for {game_key}")
                        
                except Exception as e:
                    print(f"Error analyzing opportunities for {game_key}: {e}")
                    continue
            
            if not all_opportunities:
                st.warning("No profitable betting opportunities found for today's games")
                return
            
            # Sort opportunities by edge and confidence
            all_opportunities.sort(key=lambda x: (x.get('betting_edge', 0) * x.get('confidence_score', 0.5)), reverse=True)
            
            # Display summary metrics
            st.subheader("📊 Today's Betting Summary")
            
            high_value_bets = len([o for o in all_opportunities if o.get('betting_edge', 0) >= 5.0])
            medium_value_bets = len([o for o in all_opportunities if 3.0 <= o.get('betting_edge', 0) < 5.0])
            hot_streak_bets = len([o for o in all_opportunities if o['type'] == 'Hot Streak Player'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Opportunities", len(all_opportunities))
            with col2:
                st.metric("Hot Streak Players", hot_streak_bets)
            with col3:
                st.metric("High Value (5%+ Edge)", high_value_bets)
            with col4:
                st.metric("Medium Value (3-5% Edge)", medium_value_bets)
            
            # Top opportunities
            st.subheader("🎯 Top Betting Opportunities")
            
            # Add slider to control how many opportunities to show
            num_to_show = st.slider("Number of top opportunities to display", min_value=10, max_value=50, value=25, step=5)
            
            for i, opp in enumerate(all_opportunities[:num_to_show], 1):
                # Handle new data structure
                edge = opp.get('betting_edge', opp.get('edge', 5.0))
                confidence_score = opp.get('confidence_score', 0.6)
                confidence_text = "High" if confidence_score > 0.75 else "Medium" if confidence_score > 0.6 else "Low"
                
                with st.expander(f"#{i} - {opp['type']}: {opp['bet_type']} (Edge: {edge:.1f}%)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Game:** {opp['game']}")
                        st.write(f"**Player:** {opp.get('player', 'N/A')}")
                        st.write(f"**Projection:** {opp['projection']}")
                        st.write(f"**Reasoning:** {opp['reasoning']}")
                        
                        # Show edge factors
                        if 'edge_factors' in opp:
                            st.write(f"**Edge Factors:** {', '.join(opp['edge_factors'])}")
                    
                    with col2:
                        st.metric("Edge", f"{edge:.1f}%")
                        st.metric("Confidence Score", f"{confidence_score:.1%}")
                        st.metric("Units", f"{opp.get('recommended_units', 1.0):.1f}")
                        
                        # Color code by edge and confidence
                        if edge >= 5.0 and confidence_score > 0.75:
                            st.success("🟢 Strong Bet")
                        elif edge >= 3.0 and confidence_score > 0.6:
                            st.info("🟡 Medium Bet")
                        else:
                            st.warning("🟠 Low Confidence")
            
            # Detailed breakdown by category
            st.subheader("📈 Opportunities by Category")
            
            # Group opportunities by type
            opportunities_by_type = {}
            for opp in all_opportunities:
                opp_type = opp['type']
                if opp_type not in opportunities_by_type:
                    opportunities_by_type[opp_type] = []
                opportunities_by_type[opp_type].append(opp)
            
            tabs = st.tabs(list(opportunities_by_type.keys()))
            
            for tab, (opp_type, opps) in zip(tabs, opportunities_by_type.items()):
                with tab:
                    st.write(f"**{len(opps)} {opp_type} opportunities found**")
                    
                    # Create DataFrame for better display
                    df_data = []
                    max_per_category = min(len(opps), 50)  # Show up to 50 per category
                    for opp in opps[:max_per_category]:
                        edge = opp.get('betting_edge', opp.get('edge', 5.0))
                        confidence_score = opp.get('confidence_score', 0.6)
                        confidence_text = "High" if confidence_score > 0.75 else "Medium" if confidence_score > 0.6 else "Low"
                        
                        df_data.append({
                            'Player': opp.get('player', 'Game Total'),
                            'Bet': opp.get('bet_type', opp.get('bet', 'Unknown')),
                            'Edge %': f"{edge:.1f}%",
                            'Confidence': confidence_text,
                            'Category': opp.get('category', 'General'),
                            'Reasoning': opp['reasoning'][:50] + "..." if len(opp['reasoning']) > 50 else opp['reasoning']
                        })
                    
                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
            
            # Expert insights
            st.subheader("🧠 AI Insights & Strategy")
            
            with st.expander("💡 Today's Key Insights"):
                insights = []
                
                # Generate dynamic insights based on opportunities
                high_edge_bets = [o for o in all_opportunities if o.get('betting_edge', 0) >= 10]
                if high_edge_bets:
                    insights.append(f"🎯 {len(high_edge_bets)} high-edge opportunities (10%+ edge) identified")
                
                player_props = [o for o in all_opportunities if o['type'] == 'Player Prop']
                if len(player_props) >= 5:
                    insights.append(f"🏏 Strong player prop market today with {len(player_props)} opportunities")
                
                total_bets = [o for o in all_opportunities if o['type'] == 'Total Runs']
                if len(total_bets) >= 3:
                    insights.append(f"🎲 Multiple total runs opportunities suggest weather/ballpark edges")
                
                if not insights:
                    insights = ["📊 Market appears efficient today - focus on highest confidence plays"]
                
                for insight in insights:
                    st.info(insight)
            
            with st.expander("📚 Betting Strategy Guide"):
                st.markdown("""
                **Unit-Based Sizing:**
                - Conservative bet sizing with maximum 3 units per opportunity
                - Units calculated based on edge strength and confidence
                - Focus on consistent long-term profitability
                
                **Edge Detection:**
                - High Edge (5%+): Strong statistical advantage
                - Medium Edge (3-5%): Solid value opportunity  
                - Low Edge (2-3%): Marginal value, high confidence required
                
                **Confidence Levels:**
                - High: Multiple supporting factors, strong statistical edge
                - Medium: Solid edge with good supporting data
                - Low: Marginal opportunity, proceed with caution
                
                **Key Factors:**
                - Hot streaks: Recent performance significantly above season average
                - Ballpark edges: Venue-specific advantages for power/strikeouts
                - Matchup edges: Platoon advantages and style mismatches
                """)
        
        except Exception as e:
            st.error(f"Error analyzing betting opportunities: {str(e)}")
            st.write("Please check your internet connection and try again.")


if __name__ == "__main__":
    main()

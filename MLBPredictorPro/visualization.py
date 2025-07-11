import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import plotly.figure_factory as ff

class Visualizer:
    """
    Creates visualizations for MLB analytics
    """
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'hot': '#ff4444',
            'cold': '#4444ff'
        }
    
    def create_rolling_averages_chart(self, rolling_data: pd.DataFrame, player_name: str, player_type: str) -> go.Figure:
        """Create rolling averages chart"""
        try:
            fig = go.Figure()
            
            # Select appropriate stats based on player type
            if player_type.lower() == 'batter':
                stats_to_plot = ['batting_avg_rolling', 'ops_rolling']
                titles = ['Batting Average', 'OPS']
            else:
                stats_to_plot = ['era_rolling', 'whip_rolling']
                titles = ['ERA', 'WHIP']
            
            # Create subplot traces
            for i, (stat, title) in enumerate(zip(stats_to_plot, titles)):
                if stat in rolling_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=rolling_data['date'] if 'date' in rolling_data.columns else rolling_data.index,
                            y=rolling_data[stat],
                            mode='lines+markers',
                            name=title,
                            line=dict(color=self.color_scheme['primary'] if i == 0 else self.color_scheme['secondary']),
                            hovertemplate=f'<b>{title}</b><br>' + 
                                        'Date: %{x}<br>' + 
                                        'Value: %{y:.3f}<extra></extra>'
                        )
                    )
            
            fig.update_layout(
                title=f'{player_name} - Rolling Averages',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure on error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
    
    def create_streak_visualization(self, hot_players: List[Dict], cold_players: List[Dict], stat_type: str) -> go.Figure:
        """Create streak visualization"""
        try:
            fig = go.Figure()
            
            # Hot players
            if hot_players:
                hot_names = [p['name'] for p in hot_players]
                hot_values = [p['recent_avg'] for p in hot_players]
                hot_percentiles = [p['percentile'] for p in hot_players]
                
                fig.add_trace(
                    go.Bar(
                        x=hot_names,
                        y=hot_values,
                        name='Hot Streaks',
                        marker_color=self.color_scheme['hot'],
                        customdata=hot_percentiles,
                        hovertemplate='<b>%{x}</b><br>' + 
                                    f'{stat_type}: %{{y:.3f}}<br>' + 
                                    'Percentile: %{customdata:.1%}<extra></extra>'
                    )
                )
            
            # Cold players
            if cold_players:
                cold_names = [p['name'] for p in cold_players]
                cold_values = [p['recent_avg'] for p in cold_players]
                cold_percentiles = [p['percentile'] for p in cold_players]
                
                fig.add_trace(
                    go.Bar(
                        x=cold_names,
                        y=cold_values,
                        name='Cold Streaks',
                        marker_color=self.color_scheme['cold'],
                        customdata=cold_percentiles,
                        hovertemplate='<b>%{x}</b><br>' + 
                                    f'{stat_type}: %{{y:.3f}}<br>' + 
                                    'Percentile: %{customdata:.1%}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title=f'Hot/Cold Streaks - {stat_type}',
                xaxis_title='Players',
                yaxis_title=stat_type,
                template='plotly_white',
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating streak visualization: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
    
    def create_strike_zone_heatmap(self, zone_data: np.ndarray, title: str) -> go.Figure:
        """Create strike zone heat map"""
        try:
            # Create 3x3 grid for strike zone
            if zone_data is None:
                zone_data = np.random.uniform(0.200, 0.400, (3, 3))
            
            # Reshape if needed
            if zone_data.shape != (3, 3):
                zone_data = zone_data.reshape(3, 3)
            
            # Create custom colorscale
            colorscale = [
                [0, '#313695'],      # Dark blue (cold)
                [0.25, '#4575b4'],   # Blue
                [0.5, '#ffffbf'],    # Yellow (neutral)
                [0.75, '#fe9929'],   # Orange
                [1, '#d73027']       # Red (hot)
            ]
            
            fig = go.Figure(data=go.Heatmap(
                z=zone_data,
                colorscale=colorscale,
                showscale=True,
                hovertemplate='Zone %{x}-%{y}<br>Value: %{z:.3f}<extra></extra>',
                colorbar=dict(
                    title="Performance",
                    titleside="right"
                )
            ))
            
            # Add zone labels
            annotations = []
            for i in range(3):
                for j in range(3):
                    zone_num = i * 3 + j + 1
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=f'Zone {zone_num}',
                            showarrow=False,
                            font=dict(color='white', size=12, family='Arial Black')
                        )
                    )
            
            fig.update_layout(
                title=title,
                xaxis=dict(
                    title='Strike Zone (Left-Right)',
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['Inside', 'Middle', 'Outside'],
                    showgrid=False
                ),
                yaxis=dict(
                    title='Strike Zone (Top-Bottom)',
                    tickmode='array',
                    tickvals=[0, 1, 2],
                    ticktext=['High', 'Middle', 'Low'],
                    showgrid=False
                ),
                annotations=annotations,
                template='plotly_white',
                height=500,
                width=500
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating heat map: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
    
    def create_matchup_comparison(self, batter_stats: Dict, pitcher_stats: Dict, matchup_score: float) -> go.Figure:
        """Create matchup comparison visualization"""
        try:
            fig = go.Figure()
            
            # Create radar chart data
            categories = list(batter_stats.keys())
            batter_values = list(batter_stats.values())
            pitcher_values = list(pitcher_stats.values())
            
            # Normalize values to 0-1 scale for radar chart
            batter_norm = [(float(str(v).replace('%', '')) / 100) if '%' in str(v) else 
                          (float(v) if isinstance(v, (int, float)) else 0.5) 
                          for v in batter_values]
            pitcher_norm = [(float(str(v).replace('%', '')) / 100) if '%' in str(v) else 
                           (float(v) if isinstance(v, (int, float)) else 0.5) 
                           for v in pitcher_values]
            
            # Add batter trace
            fig.add_trace(go.Scatterpolar(
                r=batter_norm,
                theta=categories,
                fill='toself',
                name='Batter',
                line_color=self.color_scheme['primary']
            ))
            
            # Add pitcher trace (inverted for comparison)
            fig.add_trace(go.Scatterpolar(
                r=[1-x for x in pitcher_norm],  # Invert for comparison
                theta=categories,
                fill='toself',
                name='Pitcher (Inverted)',
                line_color=self.color_scheme['secondary']
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=f'Matchup Comparison (Score: {matchup_score:.2f})',
                template='plotly_white',
                height=500
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating matchup comparison: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
    
    def create_prediction_confidence_chart(self, predictions: Dict, confidences: Dict) -> go.Figure:
        """Create prediction confidence visualization"""
        try:
            stats = list(predictions.keys())
            pred_values = list(predictions.values())
            conf_values = list(confidences.values())
            
            fig = go.Figure()
            
            # Add prediction bars
            fig.add_trace(go.Bar(
                x=stats,
                y=pred_values,
                name='Predictions',
                marker_color=self.color_scheme['primary'],
                yaxis='y'
            ))
            
            # Add confidence line
            fig.add_trace(go.Scatter(
                x=stats,
                y=conf_values,
                mode='lines+markers',
                name='Confidence',
                line=dict(color=self.color_scheme['secondary']),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='AI Predictions with Confidence',
                xaxis_title='Statistics',
                yaxis=dict(
                    title='Predicted Value',
                    side='left'
                ),
                yaxis2=dict(
                    title='Confidence',
                    side='right',
                    overlaying='y',
                    range=[0, 1]
                ),
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating prediction chart: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig
    
    def create_team_performance_dashboard(self, team_data: pd.DataFrame) -> go.Figure:
        """Create team performance dashboard"""
        try:
            fig = go.Figure()
            
            # Example team performance metrics
            if not team_data.empty:
                # Group by date and calculate team averages
                daily_stats = team_data.groupby('date').agg({
                    'batting_avg': 'mean',
                    'era': 'mean',
                    'runs': 'sum'
                }).reset_index()
                
                # Add team batting average
                fig.add_trace(go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['batting_avg'],
                    mode='lines+markers',
                    name='Team Batting Average',
                    line=dict(color=self.color_scheme['primary'])
                ))
                
                # Add team ERA on secondary axis
                fig.add_trace(go.Scatter(
                    x=daily_stats['date'],
                    y=daily_stats['era'],
                    mode='lines+markers',
                    name='Team ERA',
                    line=dict(color=self.color_scheme['secondary']),
                    yaxis='y2'
                ))
            
            fig.update_layout(
                title='Team Performance Dashboard',
                xaxis_title='Date',
                yaxis=dict(
                    title='Batting Average',
                    side='left'
                ),
                yaxis2=dict(
                    title='ERA',
                    side='right',
                    overlaying='y'
                ),
                template='plotly_white',
                height=400
            )
            
            return fig
            
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating team dashboard: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
            return fig

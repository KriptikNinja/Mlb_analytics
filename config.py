"""
Configuration file for MLB Analytics App
Contains authentication settings and other app configuration
"""

# Authentication Settings
AUTH_CONFIG = {
    # Customize your private credentials here
    "users": {
        "mlb": "Fuckthebooks",
        # Add your custom users here:
        # "your_username": "your_secure_password",
    },
    
    # Session settings
    "session_timeout_hours": 24,
    "require_auth": True,  # Set to False to disable authentication
}

# App Configuration
APP_CONFIG = {
    "app_name": "MLB Analytics Pro",
    "version": "2.0",
    "timezone": "US/Central",
    "max_opportunities_display": 50,
    "default_opportunities_display": 25,
}

# API Configuration
API_CONFIG = {
    "mlb_stats_api": "https://statsapi.mlb.com/api/v1",
    "cache_timeout_minutes": 5,
    "request_timeout_seconds": 30,
}
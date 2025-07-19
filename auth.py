"""
Authentication module for MLB Analytics App
Provides simple username/password protection with mobile-friendly persistent sessions
"""
import streamlit as st
import hashlib
import os
import time
from config import AUTH_CONFIG

class SimpleAuth:
    """Simple authentication system for Streamlit apps"""
    
    def __init__(self):
        # Load credentials from config
        self.users = {}
        for username, password in AUTH_CONFIG["users"].items():
            self.users[username] = self._hash_password(password)
        
        self.auth_enabled = AUTH_CONFIG.get("require_auth", True)
        # Mobile-friendly session timeout: 8 hours (28800 seconds)
        self.session_timeout = 28800
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials"""
        if username in self.users:
            return self.users[username] == self._hash_password(password)
        return False
    
    def login_form(self):
        """Display login form and handle authentication"""
        st.title("🔐 MLB Analytics - Access Required")
        st.markdown("Please enter your credentials to continue")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if self.authenticate(username, password):
                    # Store authentication with timestamp for mobile persistence
                    current_time = time.time()
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.session_state['login_time'] = current_time
                    st.session_state['last_activity'] = current_time
                    st.success("✅ Login successful! Session will stay active for 8 hours.")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        

    
    def logout(self):
        """Handle user logout"""
        if st.sidebar.button("🚪 Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.rerun()
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated with session timeout handling"""
        if not st.session_state.get('authenticated', False):
            return False
            
        # Check session timeout for mobile-friendly experience
        current_time = time.time()
        last_activity = st.session_state.get('last_activity', 0)
        
        if current_time - last_activity > self.session_timeout:
            # Session expired
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            return False
            
        # Update last activity time
        st.session_state['last_activity'] = current_time
        return True
    
    def get_username(self) -> str:
        """Get current username"""
        return st.session_state.get('username', 'Unknown')
    
    def require_auth(self):
        """Decorator-like function to require authentication"""
        # Check if authentication is enabled
        if not self.auth_enabled:
            return
            
        if not self.is_authenticated():
            self.login_form()
            st.stop()
        else:
            # Show user info in sidebar
            st.sidebar.success(f"👤 Welcome, {self.get_username()}!")
            self.logout()

# Global auth instance
auth = SimpleAuth()
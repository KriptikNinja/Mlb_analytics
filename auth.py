"""
Authentication module for MLB Analytics App
Provides simple username/password protection
"""
import streamlit as st
import hashlib
import os
from config import AUTH_CONFIG

class SimpleAuth:
    """Simple authentication system for Streamlit apps"""
    
    def __init__(self):
        # Load credentials from config
        self.users = {}
        for username, password in AUTH_CONFIG["users"].items():
            self.users[username] = self._hash_password(password)
        
        self.auth_enabled = AUTH_CONFIG.get("require_auth", True)
    
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
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        

    
    def logout(self):
        """Handle user logout"""
        if st.sidebar.button("🚪 Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.rerun()
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.get('authenticated', False)
    
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
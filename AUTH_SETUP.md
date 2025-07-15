# Authentication Setup

## How to Set Your Private Login Credentials

1. **Edit the config.py file**
2. **Find the AUTH_CONFIG section**
3. **Replace the credentials with your own:**

```python
"users": {
    "your_username": "your_secure_password",
    "another_user": "another_password",
},
```

## Security Notes

- Passwords are automatically encrypted when stored
- No credentials are displayed anywhere in the app
- Only people you give the username/password can access the betting analytics
- You can add multiple users by adding more lines

## To Disable Authentication (Optional)

Set `"require_auth": False` in the AUTH_CONFIG section of config.py

## Current Status

- Authentication is ENABLED
- Clean login form with no visible credentials
- Only authorized users you specify can access the app
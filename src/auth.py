import hashlib
from .database import add_user, get_user

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(password, hashed_password):
    """Check if the provided password matches the hash."""
    return hash_password(password) == hashed_password

def signup(username, password):
    """Handle user registration."""
    if not username or not password:
        return False, "Username and password are required."
    
    hashed = hash_password(password)
    success = add_user(username, hashed)
    if success:
        return True, "Registration successful! You can now log in."
    else:
        return False, "Username already exists."

def login(username, password):
    """Handle user login."""
    user = get_user(username)
    if user and verify_password(password, user['password']):
        return True, user
    return False, "Invalid username or password."

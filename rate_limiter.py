"""
Shared Rate Limiter Instance

This module provides a shared Flask-Limiter instance that can be imported
and used across the application.
"""

from flask_limiter import Limiter
from rate_limit_config import get_user_identifier

# Global limiter instance (will be initialized in main.py)
limiter = None


def init_limiter(app):
    """
    Initialize the rate limiter with the Flask app.

    Args:
        app: Flask application instance

    Returns:
        Limiter instance
    """
    global limiter

    from rate_limit_config import create_limiter, configure_rate_limiting

    limiter = create_limiter(app)
    configure_rate_limiting(app, limiter)

    return limiter


def get_limiter():
    """Get the limiter instance"""
    return limiter

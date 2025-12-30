"""
Rate limiting utilities for agentic endpoints

This module provides rate limit decorators that can be applied to agentic API endpoints.
The decorators are initialized when the limiter instance is set.
"""

from functools import wraps
from flask import jsonify
import logging

logger = logging.getLogger(__name__)

# Global limiter instance
_limiter = None

# Rate limit decorators (initialized when limiter is set)
workflow_limit = None
tool_limit = None
session_limit = None
router_limit = None
health_limit = None


def initialize_rate_limits(limiter):
    """
    Initialize rate limit decorators with the Flask-Limiter instance.

    Args:
        limiter: Flask-Limiter instance
    """
    global _limiter, workflow_limit, tool_limit, session_limit, router_limit, health_limit
    _limiter = limiter

    if not limiter:
        logger.warning("Rate limiter not available - using no-op decorators")
        # Create no-op decorators
        workflow_limit = lambda f: f
        tool_limit = lambda f: f
        session_limit = lambda f: f
        router_limit = lambda f: f
        health_limit = lambda f: f
        return

    try:
        from rate_limit_config import RateLimitConfig

        # Create rate limit decorators with specific limits
        workflow_limit = limiter.limit(
            RateLimitConfig.LIMITS['agentic_workflow'],
            error_message="Workflow rate limit exceeded. These operations are resource-intensive. Please wait before making another request."
        )

        tool_limit = limiter.limit(
            RateLimitConfig.LIMITS['agentic_tool'],
            error_message="Tool rate limit exceeded. Please wait before making another request."
        )

        session_limit = limiter.limit(
            RateLimitConfig.LIMITS['session_management'],
            error_message="Session management rate limit exceeded."
        )

        router_limit = limiter.limit(
            RateLimitConfig.LIMITS['router'],
            error_message="Router rate limit exceeded. Please wait before making another request."
        )

        health_limit = limiter.limit(
            RateLimitConfig.LIMITS['health'],
            error_message="Health check rate limit exceeded."
        )

        logger.info("Rate limit decorators initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing rate limits: {e}")
        # Fallback to no-op decorators
        workflow_limit = lambda f: f
        tool_limit = lambda f: f
        session_limit = lambda f: f
        router_limit = lambda f: f
        health_limit = lambda f: f


# Decorator factories that can be used as @decorators
def workflow_limited(f):
    """Decorator for workflow endpoints with rate limiting"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if workflow_limit and callable(workflow_limit):
            # Apply the rate limit decorator
            try:
                limited_f = workflow_limit(f)
                return limited_f(*args, **kwargs)
            except:
                # If rate limiting fails, just execute the function
                return f(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated


def tool_limited(f):
    """Decorator for tool endpoints with rate limiting"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if tool_limit and callable(tool_limit):
            try:
                limited_f = tool_limit(f)
                return limited_f(*args, **kwargs)
            except:
                return f(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated


def session_limited(f):
    """Decorator for session endpoints with rate limiting"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if session_limit and callable(session_limit):
            try:
                limited_f = session_limit(f)
                return limited_f(*args, **kwargs)
            except:
                return f(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated


def router_limited(f):
    """Decorator for router endpoints with rate limiting"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if router_limit and callable(router_limit):
            try:
                limited_f = router_limit(f)
                return limited_f(*args, **kwargs)
            except:
                return f(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated


def health_limited(f):
    """Decorator for health check endpoints with rate limiting"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if health_limit and callable(health_limit):
            try:
                limited_f = health_limit(f)
                return limited_f(*args, **kwargs)
            except:
                return f(*args, **kwargs)
        return f(*args, **kwargs)
    return decorated

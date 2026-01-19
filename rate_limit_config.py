"""
Rate Limiting Configuration for Flask API

This module provides comprehensive rate limiting configuration for the Flask API,
protecting against abuse, DoS attacks, and ensuring fair resource allocation.

Features:
- Redis-backed rate limiting with fallback to in-memory
- Tiered rate limits for different endpoint types
- User-based and IP-based rate limiting
- Custom error responses with retry-after headers
- Exemption support for trusted services
- Rate limit headers in responses

Author: AI Product Recommender Backend Team
"""

import os
import logging
from typing import Optional, Callable
from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logger = logging.getLogger(__name__)


class RateLimitConfig:
    """
    Rate limiting configuration with environment-based settings
    """

    # Storage Backend Configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    USE_REDIS = os.getenv('USE_REDIS_RATE_LIMIT', 'true').lower() == 'true'

    # Storage URI (Redis or in-memory)
    STORAGE_URI = REDIS_URL if USE_REDIS else 'memory://'

    # Rate Limit Strategy
    STRATEGY = 'fixed-window'  # Options: fixed-window, moving-window

    # Default Rate Limits (per minute unless specified)
    DEFAULT_LIMITS = ["200 per minute", "3000 per hour", "10000 per day"]

    # Tiered Limits for Different Endpoint Types
    LIMITS = {
        # Strict limits for expensive agentic workflows
        'agentic_workflow': [
            "10 per minute",    # Prevent abuse of expensive LLM calls
            "100 per hour",     # Reasonable limit for active users
            "500 per day"       # Daily cap for heavy users
        ],

        # Moderate limits for tool/agent endpoints (lighter operations)
        'agentic_tool': [
            "30 per minute",
            "300 per hour",
            "1500 per day"
        ],

        # Light limits for session management and health checks
        'session_management': [
            "60 per minute",
            "600 per hour",
            "3000 per day"
        ],

        # Very strict limits for authentication endpoints (prevent brute force)
        'auth': [
            "5 per minute",
            "20 per hour",
            "100 per day"
        ],

        # Generous limits for health checks and status endpoints
        'health': [
            "120 per minute",
            "1000 per hour",
            "5000 per day"
        ],

        # Router endpoint (moderate - used frequently but lightweight)
        'router': [
            "30 per minute",
            "300 per hour",
            "1500 per day"
        ]
    }

    # Header Configuration
    HEADERS_ENABLED = True
    HEADER_LIMIT = "X-RateLimit-Limit"
    HEADER_REMAINING = "X-RateLimit-Remaining"
    HEADER_RESET = "X-RateLimit-Reset"
    HEADER_RETRY_AFTER = "Retry-After"

    # Exemptions (IP addresses or user IDs to exempt from rate limiting)
    EXEMPT_IPS = os.getenv('RATE_LIMIT_EXEMPT_IPS', '').split(',')
    EXEMPT_IPS = [ip.strip() for ip in EXEMPT_IPS if ip.strip()]

    # Enable/Disable Rate Limiting Globally
    ENABLED = os.getenv('RATE_LIMITING_ENABLED', 'true').lower() == 'true'

    # Swallow errors (if True, rate limiting failures won't break the app)
    SWALLOW_ERRORS = os.getenv('RATE_LIMIT_SWALLOW_ERRORS', 'true').lower() == 'true'


def get_user_identifier() -> str:
    """
    Get unique identifier for rate limiting.
    Prioritizes authenticated user ID over IP address.

    Returns:
        str: User identifier (user_id or IP address)
    """
    # Try to get authenticated user ID from session
    from flask import session
    user_id = session.get('user_id')

    if user_id:
        return f"user:{user_id}"

    # Fallback to IP address
    return f"ip:{get_remote_address()}"


def is_exempt() -> bool:
    """
    Check if the current request should be exempt from rate limiting.

    Returns:
        bool: True if request should be exempt
    """
    # Check if IP is in exemption list
    remote_addr = get_remote_address()
    if remote_addr in RateLimitConfig.EXEMPT_IPS:
        logger.info(f"Rate limit exemption for IP: {remote_addr}")
        return True

    # Check if user has exemption flag (if using database)
    from flask import session
    user_id = session.get('user_id')
    if user_id:
        # Could add database check for premium/admin users
        # Example: user = User.query.get(user_id)
        # return user.rate_limit_exempt if user else False
        pass

    return False


def create_limiter(app: Flask) -> Limiter:
    """
    Create and configure Flask-Limiter instance.
    FIX #6: Automatically falls back to in-memory storage if Redis is unavailable.

    Args:
        app: Flask application instance

    Returns:
        Limiter: Configured limiter instance
    """
    storage_uri = RateLimitConfig.STORAGE_URI
    redis_status = "not configured"

    # FIX #6: Test Redis connection if configured, fallback to memory if unavailable
    if RateLimitConfig.USE_REDIS and 'redis' in storage_uri:
        try:
            import redis
            logger.info(f"[FIX6] Testing Redis connection: {RateLimitConfig.REDIS_URL}")

            # Quick connection test with 1 second timeout
            try:
                r = redis.from_url(
                    RateLimitConfig.REDIS_URL,
                    socket_timeout=1,
                    socket_connect_timeout=1,
                    decode_responses=True,
                    retry_on_timeout=False
                )
                # Perform ping with explicit timeout handling
                r.ping()
                logger.info(f"[FIX6] ✅ Redis connection successful")
                redis_status = "connected"

            except redis.ConnectionError as e:
                logger.warning(f"[FIX6] ❌ Redis connection failed (ConnectionError): {e}")
                logger.warning(f"[FIX6] Falling back to in-memory rate limiting")
                storage_uri = 'memory://'
                redis_status = "connection_error_fallback"

            except redis.TimeoutError as e:
                logger.warning(f"[FIX6] ❌ Redis timeout (socket/connect timeout): {e}")
                logger.warning(f"[FIX6] Falling back to in-memory rate limiting")
                storage_uri = 'memory://'
                redis_status = "timeout_fallback"

        except ImportError:
            logger.warning(f"[FIX6] Redis client library not installed, using in-memory storage")
            storage_uri = 'memory://'
            redis_status = "no_redis_library"

        except Exception as e:
            logger.warning(f"[FIX6] ❌ Unexpected error testing Redis ({type(e).__name__}): {e}")
            logger.warning(f"[FIX6] Falling back to in-memory rate limiting")
            storage_uri = 'memory://'
            redis_status = "error_fallback"

    limiter = Limiter(
        app=app,
        key_func=get_user_identifier,
        default_limits=RateLimitConfig.DEFAULT_LIMITS,
        storage_uri=storage_uri,
        storage_options={
            'socket_connect_timeout': 2,
            'socket_timeout': 2
        },
        strategy=RateLimitConfig.STRATEGY,
        headers_enabled=RateLimitConfig.HEADERS_ENABLED,
        swallow_errors=RateLimitConfig.SWALLOW_ERRORS,
        enabled=RateLimitConfig.ENABLED
    )

    logger.info(f"[FIX6] Rate limiter initialized")
    logger.info(f"[FIX6]   Storage backend: {storage_uri}")
    logger.info(f"[FIX6]   Redis status: {redis_status}")
    logger.info(f"[FIX6]   Rate limiting enabled: {RateLimitConfig.ENABLED}")

    return limiter


def rate_limit_error_handler(error):
    """
    Custom error handler for rate limit exceeded errors.

    Args:
        error: RateLimitExceeded error

    Returns:
        JSON response with rate limit information
    """
    logger.warning(
        f"Rate limit exceeded for {get_user_identifier()} "
        f"on {request.method} {request.path}"
    )

    response = jsonify({
        'success': False,
        'error': 'Rate limit exceeded',
        'message': f'Too many requests. Please try again later.',
        'retry_after': error.description if hasattr(error, 'description') else None,
        'details': {
            'limit': str(error.limit) if hasattr(error, 'limit') else None,
            'reset_time': error.reset_time if hasattr(error, 'reset_time') else None
        }
    })

    response.status_code = 429

    # Add retry-after header
    if hasattr(error, 'reset_time'):
        retry_after = int(error.reset_time - error.current_time) if error.reset_time else 60
        response.headers['Retry-After'] = str(max(1, retry_after))

    return response


def configure_rate_limiting(app: Flask, limiter: Limiter):
    """
    Configure rate limiting for the Flask application.

    Args:
        app: Flask application instance
        limiter: Limiter instance
    """
    # Register custom error handler
    app.register_error_handler(429, rate_limit_error_handler)

    # Add exemption function
    limiter.request_filter(is_exempt)

    logger.info("Rate limiting configured successfully")


# Decorators for easy application to endpoints

def workflow_limit(limiter: Limiter) -> Callable:
    """Decorator for expensive agentic workflow endpoints"""
    return limiter.limit(
        RateLimitConfig.LIMITS['agentic_workflow'],
        error_message="Workflow rate limit exceeded. Please wait before making another request."
    )


def tool_limit(limiter: Limiter) -> Callable:
    """Decorator for tool/agent endpoints"""
    return limiter.limit(
        RateLimitConfig.LIMITS['agentic_tool'],
        error_message="Tool rate limit exceeded. Please wait before making another request."
    )


def session_limit(limiter: Limiter) -> Callable:
    """Decorator for session management endpoints"""
    return limiter.limit(
        RateLimitConfig.LIMITS['session_management'],
        error_message="Session management rate limit exceeded."
    )


def auth_limit(limiter: Limiter) -> Callable:
    """Decorator for authentication endpoints"""
    return limiter.limit(
        RateLimitConfig.LIMITS['auth'],
        error_message="Authentication rate limit exceeded. Too many login attempts."
    )


def health_limit(limiter: Limiter) -> Callable:
    """Decorator for health check endpoints"""
    return limiter.limit(
        RateLimitConfig.LIMITS['health'],
        error_message="Health check rate limit exceeded."
    )


def router_limit(limiter: Limiter) -> Callable:
    """Decorator for router endpoints"""
    return limiter.limit(
        RateLimitConfig.LIMITS['router'],
        error_message="Router rate limit exceeded. Please wait before making another request."
    )


# Utility functions for monitoring and management

def get_rate_limit_stats(limiter: Limiter, identifier: str = None) -> dict:
    """
    Get current rate limit statistics for a user/IP.

    Args:
        limiter: Limiter instance
        identifier: User identifier (defaults to current user)

    Returns:
        dict: Rate limit statistics
    """
    if identifier is None:
        identifier = get_user_identifier()

    try:
        # This would require implementing custom tracking
        # Placeholder for future enhancement
        return {
            'identifier': identifier,
            'status': 'active',
            'message': 'Rate limit statistics not yet implemented'
        }
    except Exception as e:
        logger.error(f"Error getting rate limit stats: {e}")
        return {
            'identifier': identifier,
            'status': 'error',
            'message': str(e)
        }


def reset_rate_limit(limiter: Limiter, identifier: str):
    """
    Reset rate limit for a specific user/IP (admin function).

    Args:
        limiter: Limiter instance
        identifier: User identifier to reset
    """
    try:
        # This would require implementing custom reset logic
        # Placeholder for future enhancement
        logger.info(f"Rate limit reset requested for: {identifier}")
        return True
    except Exception as e:
        logger.error(f"Error resetting rate limit: {e}")
        return False


# Environment variable documentation
"""
Environment Variables for Rate Limiting:

REDIS_URL: Redis connection URL (default: redis://localhost:6379)
USE_REDIS_RATE_LIMIT: Use Redis for rate limiting (default: true)
RATE_LIMITING_ENABLED: Enable/disable rate limiting globally (default: true)
RATE_LIMIT_SWALLOW_ERRORS: Swallow rate limiting errors (default: true)
RATE_LIMIT_EXEMPT_IPS: Comma-separated list of exempt IP addresses

Example .env configuration:
REDIS_URL=redis://localhost:6379
USE_REDIS_RATE_LIMIT=true
RATE_LIMITING_ENABLED=true
RATE_LIMIT_SWALLOW_ERRORS=true
RATE_LIMIT_EXEMPT_IPS=127.0.0.1,192.168.1.100
"""

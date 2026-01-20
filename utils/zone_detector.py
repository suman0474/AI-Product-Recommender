# utils/zone_detector.py
# IP-based Zone Detection Utility for Hierarchical Thread Management
#
# Detects user zone from IP address for zone-based storage partitioning
# in the tree-based thread-ID workflow system.

import logging
import os
from enum import Enum
from typing import Optional, Dict, Any
from functools import lru_cache
import requests

logger = logging.getLogger(__name__)


# ============================================================================
# ZONE DEFINITIONS
# ============================================================================

class ThreadZone(str, Enum):
    """Available zones for thread partitioning."""

    US_WEST = "US-WEST"
    US_EAST = "US-EAST"
    EU_CENTRAL = "EU-CENTRAL"
    EU_WEST = "EU-WEST"
    ASIA_PACIFIC = "ASIA-PACIFIC"
    ASIA_SOUTH = "ASIA-SOUTH"
    DEFAULT = "DEFAULT"

    @classmethod
    def from_string(cls, value: str) -> "ThreadZone":
        """Convert string to ThreadZone with fallback to DEFAULT."""
        try:
            return cls(value.upper().replace(" ", "-").replace("_", "-"))
        except ValueError:
            return cls.DEFAULT


# ============================================================================
# IP TO ZONE MAPPING
# ============================================================================

# Country/region to zone mapping
REGION_TO_ZONE: Dict[str, ThreadZone] = {
    # US West
    "US-CA": ThreadZone.US_WEST,  # California
    "US-WA": ThreadZone.US_WEST,  # Washington
    "US-OR": ThreadZone.US_WEST,  # Oregon
    "US-NV": ThreadZone.US_WEST,  # Nevada
    "US-AZ": ThreadZone.US_WEST,  # Arizona
    "US-CO": ThreadZone.US_WEST,  # Colorado
    "US-UT": ThreadZone.US_WEST,  # Utah
    "US-HI": ThreadZone.US_WEST,  # Hawaii
    "US-AK": ThreadZone.US_WEST,  # Alaska

    # US East
    "US-NY": ThreadZone.US_EAST,  # New York
    "US-TX": ThreadZone.US_EAST,  # Texas
    "US-FL": ThreadZone.US_EAST,  # Florida
    "US-IL": ThreadZone.US_EAST,  # Illinois
    "US-PA": ThreadZone.US_EAST,  # Pennsylvania
    "US-OH": ThreadZone.US_EAST,  # Ohio
    "US-GA": ThreadZone.US_EAST,  # Georgia
    "US-NC": ThreadZone.US_EAST,  # North Carolina
    "US-MI": ThreadZone.US_EAST,  # Michigan
    "US-NJ": ThreadZone.US_EAST,  # New Jersey
    "US-VA": ThreadZone.US_EAST,  # Virginia
    "US-MA": ThreadZone.US_EAST,  # Massachusetts
    "US-DC": ThreadZone.US_EAST,  # Washington DC

    # EU Central
    "DE": ThreadZone.EU_CENTRAL,  # Germany
    "AT": ThreadZone.EU_CENTRAL,  # Austria
    "CH": ThreadZone.EU_CENTRAL,  # Switzerland
    "PL": ThreadZone.EU_CENTRAL,  # Poland
    "CZ": ThreadZone.EU_CENTRAL,  # Czech Republic
    "HU": ThreadZone.EU_CENTRAL,  # Hungary
    "SK": ThreadZone.EU_CENTRAL,  # Slovakia

    # EU West
    "GB": ThreadZone.EU_WEST,  # United Kingdom
    "FR": ThreadZone.EU_WEST,  # France
    "ES": ThreadZone.EU_WEST,  # Spain
    "IT": ThreadZone.EU_WEST,  # Italy
    "NL": ThreadZone.EU_WEST,  # Netherlands
    "BE": ThreadZone.EU_WEST,  # Belgium
    "IE": ThreadZone.EU_WEST,  # Ireland
    "PT": ThreadZone.EU_WEST,  # Portugal
    "SE": ThreadZone.EU_WEST,  # Sweden
    "NO": ThreadZone.EU_WEST,  # Norway
    "DK": ThreadZone.EU_WEST,  # Denmark
    "FI": ThreadZone.EU_WEST,  # Finland

    # Asia Pacific
    "JP": ThreadZone.ASIA_PACIFIC,  # Japan
    "KR": ThreadZone.ASIA_PACIFIC,  # South Korea
    "AU": ThreadZone.ASIA_PACIFIC,  # Australia
    "NZ": ThreadZone.ASIA_PACIFIC,  # New Zealand
    "SG": ThreadZone.ASIA_PACIFIC,  # Singapore
    "HK": ThreadZone.ASIA_PACIFIC,  # Hong Kong
    "TW": ThreadZone.ASIA_PACIFIC,  # Taiwan
    "PH": ThreadZone.ASIA_PACIFIC,  # Philippines
    "MY": ThreadZone.ASIA_PACIFIC,  # Malaysia
    "ID": ThreadZone.ASIA_PACIFIC,  # Indonesia
    "TH": ThreadZone.ASIA_PACIFIC,  # Thailand
    "VN": ThreadZone.ASIA_PACIFIC,  # Vietnam

    # Asia South
    "IN": ThreadZone.ASIA_SOUTH,  # India
    "PK": ThreadZone.ASIA_SOUTH,  # Pakistan
    "BD": ThreadZone.ASIA_SOUTH,  # Bangladesh
    "LK": ThreadZone.ASIA_SOUTH,  # Sri Lanka
    "NP": ThreadZone.ASIA_SOUTH,  # Nepal
    "AE": ThreadZone.ASIA_SOUTH,  # UAE
    "SA": ThreadZone.ASIA_SOUTH,  # Saudi Arabia
    "QA": ThreadZone.ASIA_SOUTH,  # Qatar
    "KW": ThreadZone.ASIA_SOUTH,  # Kuwait
    "OM": ThreadZone.ASIA_SOUTH,  # Oman
    "BH": ThreadZone.ASIA_SOUTH,  # Bahrain
}

# Continent fallback mapping
CONTINENT_TO_ZONE: Dict[str, ThreadZone] = {
    "NA": ThreadZone.US_EAST,      # North America -> US East (default)
    "SA": ThreadZone.US_EAST,      # South America -> US East
    "EU": ThreadZone.EU_CENTRAL,   # Europe -> EU Central (default)
    "AS": ThreadZone.ASIA_PACIFIC, # Asia -> Asia Pacific (default)
    "OC": ThreadZone.ASIA_PACIFIC, # Oceania -> Asia Pacific
    "AF": ThreadZone.EU_WEST,      # Africa -> EU West
}


# ============================================================================
# ZONE DETECTOR CLASS
# ============================================================================

class ZoneDetector:
    """
    Detects user zone from IP address using IP geolocation services.

    Supports multiple geolocation providers with fallback:
    1. ipinfo.io (default)
    2. ip-api.com (fallback)

    Results are cached to minimize API calls.
    """

    IPINFO_API_URL = "https://ipinfo.io/{ip}/json"
    IPAPI_URL = "http://ip-api.com/json/{ip}"

    def __init__(
        self,
        ipinfo_token: Optional[str] = None,
        cache_ttl_seconds: int = 3600,
        timeout_seconds: int = 5,
    ):
        """
        Initialize zone detector.

        Args:
            ipinfo_token: Optional token for ipinfo.io (increases rate limits)
            cache_ttl_seconds: Cache TTL for IP lookups (default 1 hour)
            timeout_seconds: Request timeout in seconds
        """
        self.ipinfo_token = ipinfo_token or os.getenv("IPINFO_TOKEN")
        self.timeout = timeout_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def detect_zone(self, ip_address: str) -> ThreadZone:
        """
        Detect zone from IP address.

        Args:
            ip_address: The client IP address

        Returns:
            ThreadZone based on IP geolocation
        """
        # Handle localhost/private IPs
        if self._is_local_ip(ip_address):
            logger.debug(f"[ZONE_DETECTOR] Local IP detected: {ip_address}")
            return self._get_default_zone()

        # Check cache
        cached = self._cache.get(ip_address)
        if cached:
            logger.debug(f"[ZONE_DETECTOR] Cache hit for IP: {ip_address}")
            return cached.get("zone", ThreadZone.DEFAULT)

        # Try geolocation lookup
        geo_data = self._lookup_ip(ip_address)
        if geo_data:
            zone = self._map_to_zone(geo_data)
            self._cache[ip_address] = {"zone": zone, "geo_data": geo_data}
            logger.info(f"[ZONE_DETECTOR] IP {ip_address} -> Zone {zone.value}")
            return zone

        logger.warning(f"[ZONE_DETECTOR] Failed to detect zone for IP: {ip_address}")
        return ThreadZone.DEFAULT

    def _is_local_ip(self, ip: str) -> bool:
        """Check if IP is localhost or private."""
        local_prefixes = [
            "127.",
            "10.",
            "172.16.", "172.17.", "172.18.", "172.19.",
            "172.20.", "172.21.", "172.22.", "172.23.",
            "172.24.", "172.25.", "172.26.", "172.27.",
            "172.28.", "172.29.", "172.30.", "172.31.",
            "192.168.",
            "::1",
            "fe80:",
        ]
        return any(ip.startswith(prefix) for prefix in local_prefixes) or ip == "localhost"

    def _get_default_zone(self) -> ThreadZone:
        """Get default zone from environment or config."""
        default = os.getenv("DEFAULT_THREAD_ZONE", "DEFAULT")
        return ThreadZone.from_string(default)

    def _lookup_ip(self, ip: str) -> Optional[Dict[str, Any]]:
        """
        Look up IP geolocation using available services.

        Tries ipinfo.io first, then falls back to ip-api.com.
        """
        # Try ipinfo.io
        geo_data = self._lookup_ipinfo(ip)
        if geo_data:
            return geo_data

        # Fallback to ip-api.com
        geo_data = self._lookup_ipapi(ip)
        if geo_data:
            return geo_data

        return None

    def _lookup_ipinfo(self, ip: str) -> Optional[Dict[str, Any]]:
        """Look up IP using ipinfo.io."""
        try:
            url = self.IPINFO_API_URL.format(ip=ip)
            params = {}
            if self.ipinfo_token:
                params["token"] = self.ipinfo_token

            response = requests.get(url, params=params, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                return {
                    "country": data.get("country", ""),
                    "region": data.get("region", ""),
                    "city": data.get("city", ""),
                    "loc": data.get("loc", ""),
                    "timezone": data.get("timezone", ""),
                    "source": "ipinfo.io",
                }
        except requests.RequestException as e:
            logger.warning(f"[ZONE_DETECTOR] ipinfo.io lookup failed: {e}")

        return None

    def _lookup_ipapi(self, ip: str) -> Optional[Dict[str, Any]]:
        """Look up IP using ip-api.com (free fallback)."""
        try:
            url = self.IPAPI_URL.format(ip=ip)
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return {
                        "country": data.get("countryCode", ""),
                        "region": data.get("region", ""),
                        "city": data.get("city", ""),
                        "timezone": data.get("timezone", ""),
                        "continent": data.get("continent", ""),
                        "source": "ip-api.com",
                    }
        except requests.RequestException as e:
            logger.warning(f"[ZONE_DETECTOR] ip-api.com lookup failed: {e}")

        return None

    def _map_to_zone(self, geo_data: Dict[str, Any]) -> ThreadZone:
        """Map geolocation data to a ThreadZone."""
        country = geo_data.get("country", "")
        region = geo_data.get("region", "")
        continent = geo_data.get("continent", "")

        # Try country + region (for US states)
        if country == "US" and region:
            region_key = f"US-{region[:2].upper()}"
            if region_key in REGION_TO_ZONE:
                return REGION_TO_ZONE[region_key]

        # Try country only
        if country in REGION_TO_ZONE:
            return REGION_TO_ZONE[country]

        # Try continent fallback
        if continent in CONTINENT_TO_ZONE:
            return CONTINENT_TO_ZONE[continent]

        # Default
        return ThreadZone.DEFAULT


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global detector instance (lazy initialization)
_detector: Optional[ZoneDetector] = None


def get_zone_detector() -> ZoneDetector:
    """Get or create the global zone detector instance."""
    global _detector
    if _detector is None:
        _detector = ZoneDetector()
    return _detector


def detect_zone_from_ip(ip_address: str) -> ThreadZone:
    """
    Convenience function to detect zone from IP address.

    Args:
        ip_address: The client IP address

    Returns:
        ThreadZone based on IP geolocation
    """
    return get_zone_detector().detect_zone(ip_address)


def detect_zone_from_request(request) -> ThreadZone:
    """
    Detect zone from Flask request object.

    Handles proxied requests by checking X-Forwarded-For header.

    Args:
        request: Flask request object

    Returns:
        ThreadZone based on client IP
    """
    # Check for forwarded IP (behind proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.remote_addr or "127.0.0.1"

    return detect_zone_from_ip(ip)


def get_zone_from_header(request) -> Optional[ThreadZone]:
    """
    Get zone from explicit header (if client specifies preferred zone).

    Args:
        request: Flask request object

    Returns:
        ThreadZone if header present and valid, None otherwise
    """
    zone_header = request.headers.get("X-Thread-Zone")
    if zone_header:
        return ThreadZone.from_string(zone_header)
    return None


def resolve_zone(request) -> ThreadZone:
    """
    Resolve zone using the following priority:
    1. Explicit X-Thread-Zone header
    2. IP-based detection
    3. Default zone

    Args:
        request: Flask request object

    Returns:
        ThreadZone to use for the request
    """
    # Check for explicit zone header
    explicit_zone = get_zone_from_header(request)
    if explicit_zone and explicit_zone != ThreadZone.DEFAULT:
        logger.debug(f"[ZONE_DETECTOR] Using explicit zone from header: {explicit_zone.value}")
        return explicit_zone

    # Fall back to IP detection
    return detect_zone_from_request(request)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ThreadZone",
    "ZoneDetector",
    "get_zone_detector",
    "detect_zone_from_ip",
    "detect_zone_from_request",
    "get_zone_from_header",
    "resolve_zone",
]

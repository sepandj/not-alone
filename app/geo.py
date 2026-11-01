"""City/country from an IP address, using an optional offline GeoLite2 database.

If no database is present, everything is recorded as Unknown — the service still
works, it just can't show a geographic breakdown until the db is added.
"""
import hashlib
import hmac
import os
import secrets

import geoip2.database

from .config import settings

_reader = None
if os.path.exists(settings.geoip_db_path):
    try:
        _reader = geoip2.database.Reader(settings.geoip_db_path)
    except Exception:
        _reader = None

# Per-process secret: visitor hashes cannot be reversed to IPs or linked across
# restarts. Set VISITOR_SALT in the environment to keep counts stable across
# restarts; otherwise a fresh random salt is used each boot.
_SALT = os.environ.get("VISITOR_SALT") or secrets.token_hex(16)


def locate(ip: str) -> tuple[str, str | None]:
    """Return (country, city-or-None)."""
    if not _reader or not ip:
        return "Unknown", None
    try:
        r = _reader.city(ip)
        country = r.country.name or "Unknown"
        city = r.city.name or None
        return country, city
    except Exception:
        return "Unknown", None


def visitor_hash(ip: str) -> str:
    return hashlib.sha256(f"{_SALT}:{ip}".encode()).hexdigest()


def sign(message: str) -> str:
    """Unforgeable token for `message` (used to prove authorship of a thought)."""
    return hmac.new(_SALT.encode(), message.encode(), hashlib.sha256).hexdigest()[:32]

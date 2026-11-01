"""Current matching thresholds.

The nightly tuner (scripts/autotune.py) may nudge the thresholds based on how
real thoughts spread out. Each worker re-reads the newest applied value every
few minutes; if the table is empty or unreachable, the calibrated defaults in
config.py are used, so the site never depends on the tuner having run.
"""
import time

from sqlalchemy import text as sql

from .config import settings
from .db import engine

_TTL = 300
_cache: tuple[float, float, float] | None = None  # (fetched_at, same, suggest)


def thresholds() -> tuple[float, float]:
    """(same_distance, suggest_distance) to use right now."""
    global _cache
    now = time.monotonic()
    if _cache and now - _cache[0] < _TTL:
        return _cache[1], _cache[2]
    same, sugg = settings.same_distance, settings.suggest_distance
    try:
        with engine.connect() as c:
            row = c.execute(sql(
                "SELECT same_distance, suggest_distance FROM tuning "
                "WHERE applied ORDER BY id DESC LIMIT 1")).first()
        if row:
            same, sugg = float(row[0]), float(row[1])
    except Exception:
        if _cache:  # keep the last good value through a DB blip
            return _cache[1], _cache[2]
    _cache = (now, same, sugg)
    return same, sugg

"""Core logic: given a new thought, find how many others share it, where they
are, and related thoughts to suggest — all meaning-based, not word-based."""
from collections import defaultdict

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from .config import settings
from .safety import has_crisis_keywords


def _fetch_neighbours(db: Session, vec: list[float], limit: int = 2000):
    """Candidate thoughts within the suggestion radius, nearest first."""
    rows = db.execute(
        sql(
            """
            SELECT id, text, country, city, visitor_hash,
                   embedding <=> CAST(:v AS vector) AS distance
            FROM thoughts
            WHERE embedding <=> CAST(:v AS vector) < :maxd
            ORDER BY distance
            LIMIT :lim
            """
        ),
        {"v": str(vec), "maxd": settings.suggest_distance, "lim": limit},
    ).mappings().all()
    return rows


def _geo_breakdown(same_rows):
    """Distinct-people counts per country/city, enforcing k-anonymity on cities."""
    people = defaultdict(set)          # (country, city) -> {visitor_hash}
    country_people = defaultdict(set)  # country -> {visitor_hash}
    for r in same_rows:
        country = r["country"] or "Unknown"
        country_people[country].add(r["visitor_hash"])
        if r["city"]:
            people[(country, r["city"])].add(r["visitor_hash"])

    countries = sorted(
        ({"country": c, "people": len(v)} for c, v in country_people.items()),
        key=lambda d: d["people"],
        reverse=True,
    )
    cities = []
    for (country, city), v in people.items():
        if len(v) >= settings.city_k_anonymity:  # hide tiny cities
            cities.append({"country": country, "city": city, "people": len(v)})
    cities.sort(key=lambda d: d["people"], reverse=True)
    return countries, cities


def _suggestions(sugg_rows, exclude_hash):
    """Related phrasings, each backed by >=k distinct people and crisis-safe.

    We never surface a thought only one person has written, so no individual's
    personal disclosure is exposed. The shortest phrasing in a cluster is used
    as the label."""
    clusters = defaultdict(lambda: {"people": set(), "label": None})
    for r in sugg_rows:
        # Cluster suggestions by a coarse key (their own near-duplicates collapse
        # because they sit close in meaning; we approximate with lowercased text).
        key = r["text"].strip().lower()
        c = clusters[key]
        c["people"].add(r["visitor_hash"])
        if c["label"] is None or len(r["text"]) < len(c["label"]):
            c["label"] = r["text"].strip()

    out = []
    for c in clusters.values():
        if len(c["people"]) < settings.city_k_anonymity:
            continue
        label = c["label"]
        if has_crisis_keywords(label):  # never suggest a crisis phrasing to others
            continue
        out.append({"thought": label, "people": len(c["people"])})
    out.sort(key=lambda d: d["people"], reverse=True)
    return out[:6]


def analyse(db: Session, text_in: str, vec: list[float], visitor_hash: str):
    rows = _fetch_neighbours(db, vec)
    same_rows = [r for r in rows if r["distance"] < settings.same_distance]
    sugg_rows = [r for r in rows if settings.same_distance <= r["distance"] < settings.suggest_distance]

    # Count distinct people (not submissions) who share the exact thought.
    same_people = {r["visitor_hash"] for r in same_rows}
    countries, cities = _geo_breakdown(same_rows)
    suggestions = _suggestions(sugg_rows, visitor_hash)

    return {
        "same_count": len(same_people),
        "countries": countries,
        "cities": cities,
        "suggestions": suggestions,
    }

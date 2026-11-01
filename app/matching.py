"""Core logic: given a new thought, find how many others share it, where they
are, and related thoughts to suggest — all meaning-based, not word-based."""
from collections import defaultdict

from sqlalchemy import text as sql
from sqlalchemy.orm import Session

from .config import settings
from .safety import has_crisis_keywords
from .tuning import thresholds


def _fetch_neighbours(db: Session, vec: list[float], limit: int = 2000):
    """Candidate thoughts within the suggestion radius, nearest first."""
    rows = db.execute(
        sql(
            """
            SELECT t.id, t.text, t.country, t.city, t.visitor_hash, t.status,
                   t.is_seed, t.cluster_id, c.label AS cluster_label,
                   t.embedding <=> CAST(:v AS vector) AS distance
            FROM thoughts t
            LEFT JOIN clusters c ON c.id = t.cluster_id
            WHERE t.status <> 'blocked'
              AND t.embedding <=> CAST(:v AS vector) < :maxd
            ORDER BY distance
            LIMIT :lim
            """
        ),
        {"v": str(vec), "maxd": thresholds()[1], "lim": limit},
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
    clusters = defaultdict(lambda: {"people": set(), "label": None, "seed": False, "fixed": False})
    for r in sugg_rows:
        # Group by the nightly meaning-group when the thought has one, so
        # different wordings of one feeling become one suggestion. Thoughts
        # posted since the last run fall back to their lowercased text.
        key = ("c", r["cluster_id"]) if r["cluster_id"] is not None else ("t", r["text"].strip().lower())
        c = clusters[key]
        if r["cluster_label"] and not c["fixed"]:
            c["label"], c["fixed"] = r["cluster_label"], True
        if r["is_seed"]:
            # Starter phrasing written by the operator: not a personal
            # disclosure, so it may be shown, but it is not a person.
            c["seed"] = True
        else:
            c["people"].add(r["visitor_hash"])
        if not c["fixed"] and (c["label"] is None or len(r["text"]) < len(c["label"])):
            c["label"] = r["text"].strip()

    out = []
    for c in clusters.values():
        if not c["seed"] and len(c["people"]) < settings.city_k_anonymity:
            continue
        label = c["label"]
        if has_crisis_keywords(label):  # never suggest a crisis phrasing to others
            continue
        out.append({"thought": label, "people": len(c["people"])})
    out.sort(key=lambda d: d["people"], reverse=True)
    return out[:6]


def same_people_count(db: Session, vec: list[float]) -> int:
    """Distinct real people (not seeds, not blocked) who share this thought."""
    return db.execute(
        sql(
            """
            SELECT COUNT(DISTINCT visitor_hash) FROM thoughts
            WHERE status <> 'blocked' AND NOT is_seed
              AND embedding <=> CAST(:v AS vector) < :d
            """
        ),
        {"v": str(vec), "d": thresholds()[0]},
    ).scalar() or 0


def analyse(db: Session, text_in: str, vec: list[float], visitor_hash: str):
    same_d, sugg_d = thresholds()
    rows = _fetch_neighbours(db, vec)
    # Seeds are never people: they don't count and don't appear in geo.
    # Your own earlier posts are not "others": without this, saying something
    # twice told you one other person felt it, while your share card (which
    # counts everyone once, you included) said you were the first.
    same_rows = [r for r in rows if r["distance"] < same_d and not r["is_seed"]
                 and r["visitor_hash"] != visitor_hash]
    # Suggestions show text to other people, so only moderated ('ok') thoughts
    # are eligible. Counts/geo may include 'pending' (not yet checked) so people
    # see each other immediately; a later 'blocked' verdict simply drops the row.
    sugg_rows = [r for r in rows
                 if r["status"] == "ok"
                 and same_d <= r["distance"] < sugg_d]

    # Count distinct people (not submissions) who share the exact thought.
    same_people = {r["visitor_hash"] for r in same_rows}
    countries, cities = _geo_breakdown(same_rows)
    suggestions = _suggestions(sugg_rows, visitor_hash)

    # Meaning-group of the closest thought, if it is a near-identical wording
    # (seeds included): a new thought joins it right away instead of waiting
    # for the nightly regrouping.
    cluster_id = rows[0]["cluster_id"] if rows and rows[0]["distance"] < settings.cluster_distance else None

    return {
        "cluster_id": cluster_id,
        "same_count": len(same_people),
        "countries": countries,
        "cities": cities,
        "suggestions": suggestions,
    }

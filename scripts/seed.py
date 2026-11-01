"""Load starter thoughts so a new visitor sees related feelings, not an empty room.

Seeds are marked is_seed=true. They are never counted as people (not in the
"others felt this" number, the places line, or /api/stats); they only show up
as related suggestions. Safe to run repeatedly: existing seeds are skipped.

    PYTHONPATH=/opt/not-alone/app LANG=en_US.UTF-8 venv/bin/python scripts/seed.py
"""
from sqlalchemy import select

from app.db import SessionLocal
from app.embedding import embed
from app.models import Thought
from app.safety import has_crisis_keywords

SEEDS = [
    # loneliness
    "I feel lonely", "nobody really knows me", "I feel invisible",
    "I miss having someone to talk to", "I feel left out",
    # worry
    "I'm anxious all the time", "I can't stop overthinking",
    "I'm scared about the future", "I worry I'm not doing enough",
    "money stress is wearing me down",
    # tiredness / low mood
    "I'm so tired", "I feel numb", "I've lost interest in things I used to love",
    "I can't sleep", "everything feels heavy lately",
    # self-worth
    "I feel like I'm not good enough", "I feel like a fraud",
    "I compare myself to everyone", "I'm hard on myself",
    # relationships
    "I miss someone", "I'm going through a breakup",
    "I feel misunderstood by my family", "I've grown apart from my friends",
    "I'm grieving", "I wish I could say sorry",
    # work / school
    "I'm burned out", "I hate my job", "school is overwhelming",
    "I don't know what I want to do with my life",
    # change
    "I just moved and don't know anyone", "I feel stuck",
    "I'm starting over", "I feel homesick",
    # hope / good
    "I'm proud of myself today", "I'm grateful for small things",
    "I'm trying my best", "today was a good day", "I'm getting better",
]


def main():
    db = SessionLocal()
    try:
        existing = set(db.scalars(select(Thought.text).where(Thought.is_seed)).all())
        added = 0
        for text in SEEDS:
            if text in existing:
                continue
            assert not has_crisis_keywords(text), text
            db.add(Thought(text=text, embedding=embed(text), country="Unknown", city=None,
                           visitor_hash="seed", status="ok", is_seed=True))
            added += 1
        db.commit()
        print(f"seeds: {added} added, {len(existing)} already present")
    finally:
        db.close()


if __name__ == "__main__":
    main()

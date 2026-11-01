"""Emit a JSON snapshot of how the service currently treats recent thoughts, so
the improvement agent can judge whether each response was correct.

For each recent thought it reports the moderation status and the nearest other
thoughts (with distance and the same/related band the service would place them
in). Runs INSIDE the container (needs the model + DB).

    PYTHONPATH=/opt/not-alone/app venv/bin/python scripts/eval_snapshot.py [N]
"""
import json
import sys

from sqlalchemy import text as sql

from app.config import settings
from app.db import SessionLocal


def main(n: int = 40):
    db = SessionLocal()
    try:
        recent = db.execute(sql(
            "SELECT id, text, status, country, city FROM thoughts "
            "ORDER BY id DESC LIMIT :n"), {"n": n}).mappings().all()

        out = []
        for r in recent:
            nbrs = db.execute(sql(
                """
                SELECT text, status,
                       embedding <=> (SELECT embedding FROM thoughts WHERE id=:id) AS dist
                FROM thoughts
                WHERE id <> :id AND status <> 'blocked'
                ORDER BY dist LIMIT 6
                """), {"id": r["id"]}).mappings().all()
            neighbours = [
                {"text": n["text"], "dist": round(float(n["dist"]), 3),
                 "band": ("same" if n["dist"] < settings.same_distance
                          else "related" if n["dist"] < settings.suggest_distance
                          else "unrelated")}
                for n in nbrs
            ]
            same = sum(1 for n in neighbours if n["band"] == "same")
            out.append({
                "id": r["id"], "text": r["text"], "status": r["status"],
                "match_count": same, "neighbours": neighbours,
            })

        print(json.dumps({
            "thresholds": {"same": settings.same_distance, "related": settings.suggest_distance},
            "sample": out,
        }, ensure_ascii=False))
    finally:
        db.close()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)

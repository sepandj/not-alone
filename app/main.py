from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text as sql
from sqlalchemy.orm import Session
import os

from .config import settings
from .db import get_db, engine
from .models import Thought
from .embedding import embed
from .geo import locate, visitor_hash
from .safety import is_crisis, RESOURCES
from .matching import analyse

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="not-alone")
app.state.limiter = limiter

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.exception_handler(RateLimitExceeded)
def _rl(request: Request, exc):
    raise HTTPException(status_code=429, detail="Slow down a moment, then try again.")


def client_ip(request: Request) -> str:
    # Behind Cloudflare Tunnel / a reverse proxy, trust the forwarded client IP.
    xff = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


class ThoughtIn(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("empty")
        if len(v) > settings.max_thought_chars:
            v = v[: settings.max_thought_chars]
        return v


@app.post("/api/thoughts")
@limiter.limit("20/minute")
def submit(payload: ThoughtIn, request: Request, db: Session = Depends(get_db)):
    text_in = payload.text
    vec = embed(text_in)

    ip = client_ip(request)
    vhash = visitor_hash(ip)
    country, city = locate(ip)

    result = analyse(db, text_in, vec, vhash)

    crisis = is_crisis(text_in, vec)

    # Store the new thought (this person now counts toward future matches).
    db.add(Thought(text=text_in, embedding=vec, country=country, city=city, visitor_hash=vhash))
    db.commit()

    # The person who just submitted is included in the companionship count.
    others = result["same_count"]
    return {
        "you_said": text_in,
        "others_feeling_this": others,
        "total_including_you": others + 1,
        "countries": result["countries"],
        "cities": result["cities"],
        "suggestions": result["suggestions"],
        "crisis": RESOURCES if crisis else None,
    }


@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    total = db.execute(sql("SELECT COUNT(*) FROM thoughts")).scalar() or 0
    people = db.execute(sql("SELECT COUNT(DISTINCT visitor_hash) FROM thoughts")).scalar() or 0
    return {"thoughts": total, "people": people}


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

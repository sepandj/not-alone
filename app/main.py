from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text as sql
from sqlalchemy.orm import Session
import hmac
import os
import secrets
import time

from .config import settings
from .db import get_db, engine
from .models import Thought, Share
from .embedding import embed, Busy
from .geo import locate, visitor_hash, sign
from .safety import is_crisis, has_crisis_keywords, RESOURCES
from .matching import analyse, same_people_count
from . import share as sharing
from . import seo


def client_ip(request: Request) -> str:
    # Caddy (2.5+) discards any X-Forwarded-For sent by the client and sets it
    # to the real peer address, so the first entry is trustworthy here. Do not
    # trust CF-Connecting-IP: there is no Cloudflare in front, so anyone could
    # send it to fake their location or dodge rate limits.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


# Rate limits are per visitor. (Keying on the socket address would put every
# visitor in one bucket, because all traffic arrives from Caddy on 127.0.0.1.)
# Counts live in Redis so all uvicorn workers share them; with per-process
# memory, 4 workers let ~4x the intended rate through. If Redis is down, fall
# back to per-process memory rather than failing requests.
limiter = Limiter(key_func=client_ip, default_limits=["60/minute"],
                  storage_uri=settings.ratelimit_storage,
                  in_memory_fallback_enabled=True, swallow_errors=True)
app = FastAPI(title="not-alone")


@app.on_event("startup")
def _warm():
    # Load the model and crisis references before taking traffic, so the first
    # visitors after a restart don't wait several seconds.
    is_crisis("warm up", embed("warm up"))
app.state.limiter = limiter

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.exception_handler(RateLimitExceeded)
def _rl(request: Request, exc):
    raise HTTPException(status_code=429, detail="Slow down a moment, then try again.")


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
    try:
        vec = embed(text_in)
    except Busy:
        raise HTTPException(status_code=503, headers={"Retry-After": "5"},
                            detail="lots of people are here right now. try again in a moment.")

    ip = client_ip(request)
    vhash = visitor_hash(ip)
    country, city = locate(ip)

    result = analyse(db, text_in, vec, vhash)

    crisis = is_crisis(text_in, vec)

    # Store the new thought (this person now counts toward future matches).
    t = Thought(text=text_in, embedding=vec, country=country, city=city, visitor_hash=vhash)
    db.add(t)
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
        # Proof of authorship so this person (and only they) can share it.
        # Crisis thoughts are not shareable.
        "share": None if crisis else {"thought_id": t.id, "token": share_token(t.id)},
    }


def _vec(t: Thought) -> list[float]:
    # pgvector returns a numpy array; plain floats are needed to re-query.
    return [float(x) for x in t.embedding]


def share_token(thought_id: int) -> str:
    return sign(f"share:{thought_id}")


class ShareIn(BaseModel):
    thought_id: int
    token: str
    show_text: bool = False


def _own_thought(db: Session, p: ShareIn) -> Thought:
    if not hmac.compare_digest(p.token, share_token(p.thought_id)):
        raise HTTPException(status_code=403, detail="not yours to share")
    t = db.get(Thought, p.thought_id)
    if t is None or t.status == "blocked" or t.is_seed:
        raise HTTPException(status_code=404, detail="not found")
    if is_crisis(t.text, _vec(t)):
        raise HTTPException(status_code=403, detail="not shareable")
    return t


def _words_public(t: Thought, show_text: bool) -> bool:
    return show_text and t.status == "ok" and not has_crisis_keywords(t.text)


def _total(db: Session, t: Thought) -> int:
    return sharing.cached_count(t.id, lambda: max(1, same_people_count(db, _vec(t))))


@app.post("/api/share")
@limiter.limit("20/minute")
def create_share(payload: ShareIn, request: Request, db: Session = Depends(get_db)):
    t = _own_thought(db, payload)
    s = db.query(Share).filter_by(thought_id=t.id, show_text=payload.show_text).first()
    if s is None:
        s = Share(id=secrets.token_urlsafe(6), thought_id=t.id, show_text=payload.show_text)
        db.add(s)
        db.commit()
    return {
        "id": s.id,
        "url": f"{settings.public_url}/f/{s.id}",
        # 'pending': words will appear on the public link once moderation clears them.
        "words": ("shown" if _words_public(t, s.show_text)
                  else "pending" if s.show_text and t.status == "pending" else "hidden"),
    }


@app.post("/api/card")
@limiter.limit("30/minute")
def author_card(payload: ShareIn, request: Request, db: Session = Depends(get_db)):
    """Story-sized card for the author to save or post. Rendered from their own
    words and returned only to them, so it doesn't wait for moderation."""
    t = _own_thought(db, payload)
    jpg = sharing.render_card(t.text if payload.show_text else None, _total(db, t), "story")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


def _load_share(db: Session, share_id: str) -> tuple[Share, Thought]:
    s = db.get(Share, share_id)
    t = db.get(Thought, s.thought_id) if s else None
    if s is None or t is None or t.status == "blocked":
        raise HTTPException(status_code=404, detail="not found")
    return s, t


@app.get("/api/share/{share_id}")
def read_share(share_id: str, db: Session = Depends(get_db)):
    s, t = _load_share(db, share_id)
    return {"text": t.text if _words_public(t, s.show_text) else None, "people": _total(db, t)}


@app.get("/f/{share_id}.jpg")
def share_image(share_id: str, db: Session = Depends(get_db)):
    s, t = _load_share(db, share_id)
    words = t.text if _words_public(t, s.show_text) else None
    jpg = sharing.render_card(words, _total(db, t), "og")
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=300"})


@app.get("/f/{share_id}", response_class=HTMLResponse)
def share_page(share_id: str, db: Session = Depends(get_db)):
    s, t = _load_share(db, share_id)
    words = t.text if _words_public(t, s.show_text) else None
    return HTMLResponse(sharing.page(share_id, words, _total(db, t)),
                        headers={"Cache-Control": "public, max-age=60"})


_stats_cache: tuple[float, dict] | None = None


@app.get("/api/stats")
def stats(db: Session = Depends(get_db)):
    # Every home-page visit asks for this, so cache it briefly per worker.
    global _stats_cache
    now = time.monotonic()
    if _stats_cache and now - _stats_cache[0] < 30:
        return _stats_cache[1]
    total = db.execute(sql("SELECT COUNT(*) FROM thoughts WHERE NOT is_seed AND status <> 'blocked'")).scalar() or 0
    people = db.execute(sql("SELECT COUNT(DISTINCT visitor_hash) FROM thoughts WHERE NOT is_seed AND status <> 'blocked'")).scalar() or 0
    out = {"thoughts": total, "people": people}
    _stats_cache = (now, out)
    return out


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.include_router(seo.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

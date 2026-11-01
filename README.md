# not-alone

A small, kind web service. You type what's on your mind; it tells you how many
other people have shared the **same feeling** (by meaning, not matching words),
where in the world they are, and related thoughts others nearby also feel —
so you can see you're not alone.

> "I feel lonely", "I am alone", and "I have nobody" are understood as the
> *same* thought. "I am sad" / "I am depressed" show up as related suggestions.

## How it works

- **Meaning, not keywords.** Each thought is encoded by a multilingual
  sentence-embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) that runs
  entirely on CPU, offline. Thoughts are compared by cosine distance in
  PostgreSQL via **pgvector**.
  - distance `< 0.18` → treated as the *same* thought (counted together)
  - `0.18–0.42` → offered as a *related suggestion*
- **Privacy first.**
  - Raw text is stored only to seed suggestions and is **never shown verbatim**
    to other users — only aggregate counts and phrasings shared by many people.
  - IP addresses are **never stored**; they are turned into a salted hash used
    only to count *distinct people*.
  - A city appears in the stats only once **≥ 3 distinct people** from it share
    the thought (k-anonymity); smaller groups roll up into the country.
- **Care built in.** Thoughts expressing self-harm or suicidal intent surface
  crisis-helpline resources (detected across phrasings and languages).

## Architecture

- **FastAPI** + **uvicorn** web app (`app/`)
- **PostgreSQL 17** + **pgvector** for storage and similarity search
- **sentence-transformers** for embeddings (CPU)
- Runs as a **systemd** service inside an unprivileged LXC container
- Intended to sit behind **Cloudflare Tunnel** (no open inbound ports)

## Run locally (inside the server container)

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit DATABASE_URL and VISITOR_SALT
python scripts/init_db.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Location stats (optional)

City/country come from a free offline MaxMind **GeoLite2-City** database.
Register at maxmind.com, download `GeoLite2-City.mmdb`, place it at
`/opt/not-alone/GeoLite2-City.mmdb` (or set `GEOIP_DB_PATH`). Without it the
service still works; locations are recorded as *Unknown*.

## Note

This is a supportive, anonymous experiment — not a medical or crisis service.
If you or someone else is in danger, contact your local emergency number.

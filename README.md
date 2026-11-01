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
- Runs as a **systemd** service inside an unprivileged LXC container, as its
  own unprivileged `notalone` user with a read-only filesystem view
- **Caddy** in front for HTTPS (Let's Encrypt over port 443)

## Run locally (inside the server container)

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit DATABASE_URL and VISITOR_SALT
python scripts/init_db.py
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Location stats

City/country come from the free offline **DB-IP "IP to City Lite"** database
(CC BY 4.0, no account needed; attribution link is in the page footer).
`deploy/geoip-update.sh` downloads it to `/opt/not-alone/dbip-city-lite.mmdb`
and `not-alone-geoip.timer` refreshes it weekly. Without the file, locations
are recorded as *Unknown*.

## Nightly tuning (no LLM)

`scripts/autotune.py` runs at 03:30 (`not-alone-autotune.timer`):

- **Thresholds.** Looks at how far each real thought is from the nearest thought
  by someone else and finds the dip between "same feeling" and "unrelated".
  It moves `same_distance` toward that dip by at most 0.01 a night, within
  0.38-0.52, and only once 150+ people have posted. A change is kept only if
  the hand-labeled pairs in `scripts/tuning_pairs.json` still score as well as
  the calibrated defaults and typo'd thoughts still match their originals 85%
  of the time. Every run is logged in the `tuning` table with its reason.
- **Meaning-groups.** Thoughts that are rewordings of each other (every member
  within 0.30) are grouped so suggestions show one entry with a combined count.
  Each starter phrasing stays its own group. New posts join their nearest group
  right away.

Dry run: `PYTHONPATH=. venv/bin/python scripts/autotune.py --dry-run`

## Starter thoughts

`scripts/seed.py` loads a few dozen common feelings (`is_seed = true`) so new
visitors see related suggestions instead of an empty room. Seeds are never
counted as people: not in "others felt this", the places line, or `/api/stats`.

## Sharing

After posting, a person can share a story-sized card and a link
(`/f/<id>`) showing the live count of people who felt the same thing. Only the
author can create a share. Their words appear on the public link only if they
opt in and moderation has marked the thought `ok`; crisis thoughts can't be
shared.

## Moderation and review agents (run on the host)

- `agent/moderate.py` (every minute): new thoughts are first screened by a
  local model (Qwen2.5-3B via llama.cpp on the GPU, `agent/not-alone-llm.service`,
  127.0.0.1:8081). The local model may only **approve**, and only at
  `p(ok) >= 0.98` with no red flags (links, phone numbers, keyboard mash,
  instruction-like text). Everything else goes to Claude Haiku, which decides
  ok/block. If the local server is down, everything goes to Claude.
  Test set: `agent/tests/moderation_cases.json`.
- `agent/evaluate.py` (every 12 hours, `agent/not-alone-evaluate.timer`):
  reviews recent responses, re-moderates mistakes, and emails code suggestions.
  Skips when nothing new was posted. Its log quotes posts and is not committed.

## Note

This is a supportive, anonymous experiment — not a medical or crisis service.
If you or someone else is in danger, contact your local emergency number.

## License

MIT, see [LICENSE](LICENSE). City data from [DB-IP](https://db-ip.com) (CC BY 4.0).

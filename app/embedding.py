"""Turns a thought into a meaning-fingerprint (vector), fully offline on CPU.

The first call downloads the model (~120 MB) once; afterwards it is cached in
/opt/not-alone/models and no network is used.
"""
import os
import queue
import threading
from concurrent.futures import Future

os.environ.setdefault("HF_HOME", "/opt/not-alone/models")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import unicodedata

from sentence_transformers import SentenceTransformer

from .config import settings

_model: SentenceTransformer | None = None
_lock = threading.RLock()

_ws = re.compile(r"\s+")
_elong = re.compile(r"(.)\1{2,}")  # "sooo lonely" -> "soo lonely"


def normalize(text: str) -> str:
    """Light, language-agnostic cleanup so trivial variations don't look like
    different thoughts: Unicode-normalize, lowercase, squeeze repeats and spaces.
    It does not try to fix real misspellings — the embedding handles most of
    those via subword tokens — it just removes noise before encoding. Applied to
    every thought consistently, so stored and query vectors stay comparable."""
    t = unicodedata.normalize("NFKC", text).strip().lower()
    t = _elong.sub(r"\1\1", t)
    t = _ws.sub(" ", t)
    return t


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                # Threads per worker process. Workers x threads should not exceed
                # the CPU count, or the processes fight over cores and slow down.
                import torch
                torch.set_num_threads(int(os.environ.get("EMBED_THREADS", "2")))
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


# ---- micro-batching ----------------------------------------------------------
# Requests are handled on many threads at once. If each ran the model itself
# they would fight over the same few CPU cores (measured: 8.5 posts/s, p95 11s).
# Instead, one thread per process runs the model and encodes whatever has piled
# up as a single batch, which is ~2x more efficient per sentence on this CPU.
# Results are the same vectors as encoding one at a time.
MAX_BATCH = 32
# Load shedding: past this many waiting posts (~3-4 s of work for one worker),
# answer "busy, try again" right away instead of letting everyone time out.
MAX_QUEUE = 24


class Busy(Exception):
    pass

_q: "queue.Queue[tuple[str, Future]]" = queue.Queue()
_worker: threading.Thread | None = None


def _run():
    model = get_model()
    while True:
        batch = [_q.get()]
        while len(batch) < MAX_BATCH:
            try:
                batch.append(_q.get_nowait())
            except queue.Empty:
                break
        texts = [t for t, _ in batch]
        try:
            vecs = model.encode(texts, normalize_embeddings=True,  # unit vectors -> cosine
                                convert_to_numpy=True, batch_size=MAX_BATCH)
            for (_, fut), v in zip(batch, vecs):
                fut.set_result(v.tolist())
        except Exception as e:  # never leave a request hanging
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)


def _ensure_worker():
    global _worker
    if _worker is None:
        with _lock:
            if _worker is None:
                get_model()
                _worker = threading.Thread(target=_run, name="embedder", daemon=True)
                _worker.start()


def embed(text: str) -> list[float]:
    _ensure_worker()
    if _q.qsize() >= MAX_QUEUE:
        raise Busy()
    fut: Future = Future()
    _q.put((normalize(text), fut))
    return fut.result(timeout=60)

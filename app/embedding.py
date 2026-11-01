"""Turns a thought into a meaning-fingerprint (vector), fully offline on CPU.

The first call downloads the model (~120 MB) once; afterwards it is cached in
/opt/not-alone/models and no network is used.
"""
import os
import threading

os.environ.setdefault("HF_HOME", "/opt/not-alone/models")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import unicodedata

from sentence_transformers import SentenceTransformer

from .config import settings

_model: SentenceTransformer | None = None
_lock = threading.Lock()

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
                _model = SentenceTransformer(settings.embedding_model, device="cpu")
    return _model


def embed(text: str) -> list[float]:
    vec = get_model().encode(
        normalize(text),
        normalize_embeddings=True,  # unit vectors -> cosine distance is meaningful
        convert_to_numpy=True,
    )
    return vec.tolist()

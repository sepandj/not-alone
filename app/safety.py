"""Crisis detection.

If a thought expresses intent or desire to self-harm or die, we surface
helpline resources alongside the normal result. Detection is semantic (works
across phrasings and languages) with a keyword fast-path. This is a supportive
signpost, not a diagnosis, and deliberately errs toward showing help.
"""
import numpy as np

from .embedding import embed

_CRISIS_REFERENCES = [
    "I want to kill myself",
    "I want to die",
    "I am going to end my life",
    "I don't want to live anymore",
    "I want to hurt myself",
    "there is no reason to keep living",
    "everyone would be better off without me",
]

_KEYWORDS = (
    "kill myself", "end my life", "want to die", "suicide", "suicidal",
    "hurt myself", "self harm", "self-harm", "better off dead", "no reason to live",
)

_ref_vecs: np.ndarray | None = None

# A few widely-used lines. Expand per launch region.
RESOURCES = {
    "message": (
        "It sounds like you may be going through something very hard. "
        "You are not alone, and people want to help right now."
    ),
    "lines": [
        {"where": "International", "name": "Find a Helpline", "contact": "https://findahelpline.com"},
        {"where": "US", "name": "988 Suicide & Crisis Lifeline", "contact": "call or text 988"},
        {"where": "UK & ROI", "name": "Samaritans", "contact": "call 116 123"},
        {"where": "Iran", "name": "Social Emergency (Behzisti)", "contact": "call 123"},
    ],
}


def _refs() -> np.ndarray:
    global _ref_vecs
    if _ref_vecs is None:
        _ref_vecs = np.array([embed(t) for t in _CRISIS_REFERENCES], dtype=np.float32)
    return _ref_vecs


def has_crisis_keywords(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in _KEYWORDS)


def is_crisis(text: str, vec: list[float]) -> bool:
    if has_crisis_keywords(text):
        return True
    v = np.array(vec, dtype=np.float32)
    sims = _refs() @ v  # vectors are normalized -> dot product = cosine similarity
    return bool(sims.max() >= 0.62)

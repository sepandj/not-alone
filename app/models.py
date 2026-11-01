from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, DateTime, func, Index
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base
from .config import settings


class Thought(Base):
    __tablename__ = "thoughts"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Raw text is kept only to seed suggestions; it is never shown verbatim to
    # other users. Only aggregate counts and canonical/suggested phrasings leave
    # the server.
    text: Mapped[str] = mapped_column(String(settings.max_thought_chars))
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim))

    country: Mapped[str] = mapped_column(String(80), default="Unknown")
    city: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # Moderation lifecycle: 'pending' -> 'ok' | 'blocked'. Only 'ok' thoughts are
    # ever counted for other people or surfaced as suggestions. A newly submitted
    # thought starts 'pending' and the low-cost Claude agent clears it shortly
    # after. 'reason' stores the agent's short justification for blocks.
    status: Mapped[str] = mapped_column(String(10), default="pending", index=True)
    mod_reason: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Salted hash of the visitor's IP — lets us count *distinct people* per city
    # for k-anonymity without ever storing an IP address.
    visitor_hash: Mapped[str] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# IVFFlat/HNSW index for fast cosine search as the table grows.
Index(
    "ix_thoughts_embedding",
    Thought.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

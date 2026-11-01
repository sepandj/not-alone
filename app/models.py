from datetime import datetime

from pgvector.sqlalchemy import Vector
from datetime import date

from sqlalchemy import String, DateTime, Date, Integer, Boolean, ForeignKey, UniqueConstraint, func, Index
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

    # Starter thoughts written by the operator so a new visitor sees related
    # feelings instead of an empty room. They are NEVER counted as people (not
    # in "others felt this", geo, or stats); they only appear as suggestions.
    is_seed: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Share(Base):
    """A public link to one thought's live count. Only the person who wrote the
    thought can create one (see main.share_token). The words themselves are
    shown only if the author opted in AND moderation has marked them 'ok'."""
    __tablename__ = "shares"
    __table_args__ = (UniqueConstraint("thought_id", "show_text"),)

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    thought_id: Mapped[int] = mapped_column(ForeignKey("thoughts.id", ondelete="CASCADE"), index=True)
    show_text: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Visit(Base):
    """One row per visitor per day: powers the public view counter and the
    "where people are" panel. Like thoughts, it keeps only a salted hash of the
    IP (never the IP) plus the country/city it resolved to."""
    __tablename__ = "visits"

    day: Mapped[date] = mapped_column(Date, primary_key=True, server_default=func.current_date())
    visitor_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    country: Mapped[str] = mapped_column(String(80), default="Unknown")
    city: Mapped[str | None] = mapped_column(String(120), nullable=True)
    views: Mapped[int] = mapped_column(Integer, default=1, server_default="1")


# IVFFlat/HNSW index for fast cosine search as the table grows.
Index(
    "ix_thoughts_embedding",
    Thought.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

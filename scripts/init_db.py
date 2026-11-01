"""Create the pgvector extension and all tables. Safe to run repeatedly."""
from sqlalchemy import text

from app.db import engine, Base
from app import models  # noqa: F401  (registers tables)


def main():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    # Columns added after the first release (create_all won't alter tables).
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE thoughts ADD COLUMN IF NOT EXISTS is_seed boolean NOT NULL DEFAULT false"))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_thoughts_is_seed ON thoughts (is_seed)"))
    print("Database ready.")


if __name__ == "__main__":
    main()

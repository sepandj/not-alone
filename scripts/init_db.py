"""Create the pgvector extension and all tables. Safe to run repeatedly."""
from sqlalchemy import text

from app.db import engine, Base
from app import models  # noqa: F401  (registers tables)


def main():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    print("Database ready.")


if __name__ == "__main__":
    main()

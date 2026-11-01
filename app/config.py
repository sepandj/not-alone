from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://notalone:notalone@localhost:5432/notalone"

    # Sentence embedding model (multilingual, CPU-friendly, incl. Persian).
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    # Cosine-distance thresholds (pgvector <=> returns distance = 1 - cosine_sim).
    # Below same_distance  -> counted as the SAME thought.
    # Below suggest_distance (but above same) -> offered as a related suggestion.
    same_distance: float = 0.18
    suggest_distance: float = 0.42

    # Privacy: a city is only ever shown once at least this many distinct people
    # from it share the thought. Smaller groups roll up into the country total.
    city_k_anonymity: int = 3

    # Optional MaxMind GeoLite2-City.mmdb for city/country from IP.
    # Register free at maxmind.com; without it, location is recorded as Unknown.
    geoip_db_path: str = "/opt/not-alone/GeoLite2-City.mmdb"

    max_thought_chars: int = 280

    model_config_name: str = "not-alone"


settings = Settings()

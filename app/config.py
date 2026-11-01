from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://notalone:notalone@localhost:5432/notalone"

    # Sentence embedding model (multilingual, CPU-friendly, incl. Persian).
    # mpnet-base separates meaning noticeably better than the MiniLM sibling and
    # still runs comfortably on CPU. 768-dimensional embeddings.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768

    # Cosine-distance thresholds (pgvector <=> returns distance = 1 - cosine_sim).
    # Below same_distance  -> counted as the SAME thought.
    # Below suggest_distance (but above same) -> offered as a related suggestion.
    # Calibrated against the open STS-B dataset (1500 human-scored pairs) plus a
    # typo-robustness sweep (see scripts/calibrate.py). On real emotional
    # phrasings the model groups paraphrases tightly ("I feel lonely" ~ "I am
    # alone" ~ "I am sad" ~ 0.34; "anxious" ~ "worried" ~ 0.16) and rejects
    # unrelated text (~0.90). same=0.45 keeps those together and holds ~88% of
    # typo'd variants within range; wider feelings ("I have nobody" ~0.48)
    # surface as suggestions below suggest_distance.
    same_distance: float = 0.45
    suggest_distance: float = 0.65

    # Privacy: a city is only ever shown once at least this many distinct people
    # from it share the thought. Smaller groups roll up into the country total.
    city_k_anonymity: int = 3

    # Optional MaxMind GeoLite2-City.mmdb for city/country from IP.
    # Register free at maxmind.com; without it, location is recorded as Unknown.
    geoip_db_path: str = "/opt/not-alone/GeoLite2-City.mmdb"

    max_thought_chars: int = 280

    model_config_name: str = "not-alone"


settings = Settings()

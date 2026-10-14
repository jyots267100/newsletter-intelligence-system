"""Centralized application configuration, loaded from environment variables.

All tunable parameters (chunking, retrieval, model names, limits) live here
so that the rest of the codebase never hardcodes a magic number and every
value can be overridden via `.env` or real environment variables without
touching code.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # frozen=True gives Settings a stable __hash__, which lets it be used as
    # an argument to `@lru_cache`-wrapped dependency factories (see
    # app/api/dependencies.py) without pydantic's default "unhashable type"
    # behavior getting in the way.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", frozen=True
    )

    # --- Gemini ---------------------------------------------------------
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    llm_timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS")

    # --- Embeddings -------------------------------------------------------
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # --- Storage ----------------------------------------------------------
    chroma_persist_directory: str = Field(default="./chroma_data", alias="CHROMA_PERSIST_DIRECTORY")
    bm25_index_path: str = Field(default="./chroma_data/bm25_index.pkl", alias="BM25_INDEX_PATH")
    document_registry_path: str = Field(
        default="./chroma_data/documents.json", alias="DOCUMENT_REGISTRY_PATH"
    )

    # --- Chunking ---------------------------------------------------------
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    # --- Retrieval --------------------------------------------------------
    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_candidate_k: int = Field(default=20, alias="RETRIEVAL_CANDIDATE_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    # Analysis (summary/insights/actions) needs broader document coverage
    # than a single chat question, so it uses a larger top_k over the same
    # hybrid retrieval pipeline.
    analysis_top_k: int = Field(default=15, alias="ANALYSIS_TOP_K")

    # --- Uploads ------------------------------------------------------------
    max_upload_size_mb: int = Field(default=20, alias="MAX_UPLOAD_SIZE_MB")

    # --- Chat ---------------------------------------------------------------
    chat_history_turns: int = Field(default=6, alias="CHAT_HISTORY_TURNS")

    # --- App ------------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    app_env: str = Field(default="development", alias="APP_ENV")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_storage_dirs(self) -> None:
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.bm25_index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.document_registry_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_storage_dirs()
    return settings

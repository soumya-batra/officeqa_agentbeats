import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


@dataclass(frozen=True)
class SolverConfig:
    llm_provider: str
    openai_model: str
    anthropic_model: str
    anthropic_max_tokens: int
    gemini_model: str
    gemini_embedding_model: str
    reasoning_effort: str
    enable_web_search: bool
    corpus_dir: Path | None
    faiss_index_dir: Path | None
    parsed_json_dir: Path | None
    cpi_data_path: Path | None
    retrieval_top_k: int
    debug_output_dir: Path | None
    write_debug_artifacts: bool
    llm_cache_path: Path | None

    @property
    def embedding_provider(self) -> str:
        """Determine which provider to use for embeddings."""
        if self.llm_provider == "gemini" and os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        return "openai"

    @classmethod
    def from_env(cls) -> "SolverConfig":
        corpus_dir = _env_path("CORPUS_DIR")
        parsed_json_dir = _env_path("PARSED_JSON_DIR")
        if parsed_json_dir is None and corpus_dir is not None:
            candidate = corpus_dir.parent / "jsons" if corpus_dir.name == "transformed" else corpus_dir.parent / "jsons"
            if candidate.exists():
                parsed_json_dir = candidate
        return cls(
            llm_provider=os.environ.get("LLM_PROVIDER", "").lower(),
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5-20251101"),
            anthropic_max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000")),
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
            gemini_embedding_model=os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            reasoning_effort=os.environ.get("REASONING_EFFORT", ""),
            enable_web_search=_env_bool("ENABLE_WEB_SEARCH"),
            corpus_dir=corpus_dir,
            faiss_index_dir=_env_path("FAISS_INDEX_DIR"),
            parsed_json_dir=parsed_json_dir,
            cpi_data_path=_env_path("CPI_DATA_PATH"),
            retrieval_top_k=int(os.environ.get("RETRIEVAL_TOP_K", "25")),
            debug_output_dir=_env_path("DEBUG_OUTPUT_DIR"),
            write_debug_artifacts=_env_bool("WRITE_DEBUG_ARTIFACTS", True),
            llm_cache_path=_env_path("LLM_CACHE_PATH"),
        )

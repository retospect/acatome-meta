"""Unified configuration for all acatome packages.

Resolution order: env vars > ./acatome.toml > ~/.acatome/config.toml > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


ACATOME_HOME = Path.home() / ".acatome"


@dataclass
class StoreSection:
    path: str = "~/.acatome/store/"
    vector_backend: str = "chroma"
    graph_backend: str = "none"
    metadata_backend: str = "sqlite"
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "acatome"
    pg_schema: str = "acatome"
    pg_user: str = "acatome"
    pg_password: str = ""
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""


@dataclass
class EmbedProfile:
    model: str = "all-MiniLM-L6-v2"
    dim: int = 384
    provider: str = "chroma"
    index_dim: int | None = None
    matryoshka: bool = False


@dataclass
class EnrichSection:
    embed_profiles: list[str] = field(default_factory=lambda: ["default"])
    summarize: bool = True
    summarizer: str = "ollama/qwen3.5:9b"


@dataclass
class ExtractSection:
    header_method: str = "auto"
    verify: bool = True
    auto_ingest: bool = True
    enrich: EnrichSection = field(default_factory=EnrichSection)
    profiles: dict[str, EmbedProfile] = field(
        default_factory=lambda: {
            "default": EmbedProfile(),
            "accurate": EmbedProfile(
                model="Qwen/Qwen3-Embedding-4B",
                dim=2560,
                provider="sentence-transformers",
                index_dim=768,
                matryoshka=True,
            ),
        }
    )


@dataclass
class ApiSection:
    s2_api_key: str = ""
    crossref_mailto: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""


@dataclass
class AcatomeConfig:
    store: StoreSection = field(default_factory=StoreSection)
    extract: ExtractSection = field(default_factory=ExtractSection)
    api: ApiSection = field(default_factory=ApiSection)

    @property
    def store_path(self) -> Path:
        return Path(self.store.path).expanduser()


class BackendMissingError(ImportError):
    """Raised when a configured backend's dependencies are not installed."""


def load_config(*, validate: bool = True) -> AcatomeConfig:
    """Load configuration with resolution: env > ./acatome.toml > ~/.acatome/config.toml > defaults.

    Also respects ``$ACATOME_CONFIG`` env var pointing to a custom config file.
    """
    cfg = AcatomeConfig()

    # Load file-based config (lower priority first, then override)
    global_path = ACATOME_HOME / "config.toml"
    local_path = Path("acatome.toml")
    custom_path_str = os.environ.get("ACATOME_CONFIG")
    custom_path = Path(custom_path_str) if custom_path_str else None

    for path in [global_path, local_path, custom_path]:
        if path is not None and path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            _apply_toml(cfg, data)

    # Env vars override everything
    _apply_env(cfg)

    if validate:
        _validate_backends(cfg)

    return cfg


def _validate_backends(cfg: AcatomeConfig) -> None:
    """Check that required backend dependencies are installed."""
    if (
        cfg.store.vector_backend == "postgres"
        or cfg.store.metadata_backend == "postgres"
    ):
        try:
            import psycopg  # noqa: F401
        except ImportError:
            raise BackendMissingError(
                "acatome-store: backend 'postgres' requires extra dependencies.\n"
                'Install with: pip install "acatome-store[postgres]"'
            ) from None

    if cfg.store.graph_backend == "neo4j":
        try:
            import neo4j  # noqa: F401
        except ImportError:
            raise BackendMissingError(
                "acatome-store: backend 'neo4j' requires extra dependencies.\n"
                'Install with: pip install "acatome-store[neo4j]"'
            ) from None


def _apply_toml(cfg: AcatomeConfig, data: dict[str, Any]) -> None:
    """Apply parsed TOML data to config."""
    # [store]
    if "store" in data:
        s = data["store"]
        for key in ["path", "vector_backend", "graph_backend", "metadata_backend"]:
            if key in s:
                setattr(cfg.store, key, s[key])
        if "vector" in s and "backend" in s["vector"]:
            cfg.store.vector_backend = s["vector"]["backend"]
        if "graph" in s and "backend" in s["graph"]:
            cfg.store.graph_backend = s["graph"]["backend"]
        if "metadata" in s and "backend" in s["metadata"]:
            cfg.store.metadata_backend = s["metadata"]["backend"]
        if "postgres" in s:
            pg = s["postgres"]
            for key in ["host", "port", "database", "schema", "user", "password"]:
                if key in pg:
                    setattr(cfg.store, f"pg_{key}", pg[key])
        if "neo4j" in s:
            n = s["neo4j"]
            for key in ["url", "user", "password"]:
                if key in n:
                    setattr(cfg.store, f"neo4j_{key}", n[key])

    # [extract]
    if "extract" in data:
        e = data["extract"]
        for key in ["header_method", "verify", "auto_ingest"]:
            if key in e:
                setattr(cfg.extract, key, e[key])
        if "enrich" in e:
            en = e["enrich"]
            if "embed_profiles" in en:
                cfg.extract.enrich.embed_profiles = en["embed_profiles"]
            if "summarize" in en:
                cfg.extract.enrich.summarize = en["summarize"]
            if "summarizer" in en:
                cfg.extract.enrich.summarizer = en["summarizer"]
        if "profiles" in e:
            for name, pdata in e["profiles"].items():
                cfg.extract.profiles[name] = EmbedProfile(
                    model=pdata.get("model", "all-MiniLM-L6-v2"),
                    dim=pdata.get("dim", 384),
                    provider=pdata.get("provider", "chroma"),
                    index_dim=pdata.get("index_dim"),
                    matryoshka=pdata.get("matryoshka", False),
                )

    # [api]
    if "api" in data:
        a = data["api"]
        for key in [
            "s2_api_key",
            "crossref_mailto",
            "openai_api_key",
            "anthropic_api_key",
        ]:
            if key in a:
                setattr(cfg.api, key, a[key])


def _apply_env(cfg: AcatomeConfig) -> None:
    """Override config with environment variables.

    API keys use standard env var names (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    SEMANTIC_SCHOLAR_API_KEY) so they work with other tools too.
    Acatome-specific config keeps the ACATOME_ prefix.
    """
    env_map = {
        # Acatome-specific (ACATOME_ prefix)
        "ACATOME_PG_PASSWORD": ("store", "pg_password"),
        "ACATOME_PG_HOST": ("store", "pg_host"),
        "ACATOME_PG_PORT": ("store", "pg_port"),
        "ACATOME_PG_DATABASE": ("store", "pg_database"),
        "ACATOME_PG_USER": ("store", "pg_user"),
        "ACATOME_NEO4J_PASSWORD": ("store", "neo4j_password"),
        "ACATOME_CROSSREF_MAILTO": ("api", "crossref_mailto"),
        "ACATOME_STORE_PATH": ("store", "path"),
        "ACATOME_SUMMARIZER": ("extract_enrich", "summarizer"),
        # Standard API key env vars (no prefix)
        "OPENAI_API_KEY": ("api", "openai_api_key"),
        "ANTHROPIC_API_KEY": ("api", "anthropic_api_key"),
        "SEMANTIC_SCHOLAR_API_KEY": ("api", "s2_api_key"),
    }

    for env_key, (section, attr) in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if section == "store":
                target = cfg.store
            elif section == "api":
                target = cfg.api
            elif section == "extract_enrich":
                target = cfg.extract.enrich
            else:
                continue
            # Type coerce for port
            if attr == "pg_port":
                val = int(val)
            setattr(target, attr, val)

"""Unified configuration for all acatome packages.

Resolution order: env vars > ./acatome.toml > ~/.acatome/config.toml > defaults.
"""

from __future__ import annotations

import logging
import os
import warnings
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


log = logging.getLogger(__name__)


def _has_psycopg() -> bool:
    """Check if psycopg is importable."""
    try:
        import psycopg  # noqa: F401
        return True
    except ImportError:
        return False


def _has_neo4j() -> bool:
    """Check if neo4j driver is importable."""
    try:
        import neo4j  # noqa: F401
        return True
    except ImportError:
        return False


def load_config(*, validate: bool = True) -> AcatomeConfig:
    """Load configuration with resolution: env > ./acatome.toml > ~/.acatome/config.toml > defaults.

    Backend selection logic:
    - If a config file explicitly sets a backend, that backend is required.
      Missing dependencies cause a hard error with the source file named.
    - If no config file sets the backend, auto-detect from installed packages:
      psycopg installed → postgres, otherwise sqlite/chroma.

    Also respects ``$ACATOME_CONFIG`` env var pointing to a custom config file.
    """
    cfg = AcatomeConfig()

    # Track which config files explicitly set backend fields
    backend_sources: dict[str, str] = {}  # field -> file path

    # Load file-based config (lower priority first, then override)
    global_path = ACATOME_HOME / "config.toml"
    local_path = Path("acatome.toml")
    custom_path_str = os.environ.get("ACATOME_CONFIG")
    custom_path = Path(custom_path_str) if custom_path_str else None

    for path in [global_path, local_path, custom_path]:
        if path is not None and path.exists():
            with open(path, "rb") as f:
                data = tomllib.load(f)
            # Track backend sources before applying
            _track_backend_sources(data, str(path), backend_sources)
            _apply_toml(cfg, data)

    # Env vars override everything
    _apply_env(cfg)

    if validate:
        _resolve_backends(cfg, backend_sources)

    return cfg


def ensure_config() -> Path:
    """Write ~/.acatome/config.toml if it doesn't exist.

    Detects installed backends and writes a config that locks in the choice.
    Returns the path to the config file.
    """
    config_path = ACATOME_HOME / "config.toml"
    if config_path.exists():
        return config_path

    ACATOME_HOME.mkdir(parents=True, exist_ok=True)

    if _has_psycopg():
        backend_block = (
            '[store]\n'
            'metadata_backend = "postgres"\n'
            'vector_backend = "postgres"\n'
        )
    else:
        backend_block = (
            '[store]\n'
            'metadata_backend = "sqlite"\n'
            'vector_backend = "chroma"\n'
        )

    config_path.write_text(
        "# Auto-generated by acatome. Edit as needed.\n"
        "# Docs: https://github.com/retospect/acatome-chat\n\n"
        f"{backend_block}\n"
        '[extract.enrich]\n'
        'summarize = true\n'
        'summarizer = "ollama/qwen3.5:9b"\n'
    )
    log.info("Wrote default config to %s", config_path)
    return config_path


def _track_backend_sources(
    data: dict[str, Any], source: str, sources: dict[str, str]
) -> None:
    """Record which config file set backend fields."""
    store = data.get("store", {})
    if "metadata_backend" in store:
        sources["metadata_backend"] = source
    if "vector_backend" in store:
        sources["vector_backend"] = source
    if "graph_backend" in store:
        sources["graph_backend"] = source
    # Also check nested form: [store.vector] backend = ...
    if "vector" in store and "backend" in store["vector"]:
        sources["vector_backend"] = source
    if "metadata" in store and "backend" in store["metadata"]:
        sources["metadata_backend"] = source
    if "graph" in store and "backend" in store["graph"]:
        sources["graph_backend"] = source


def _resolve_backends(
    cfg: AcatomeConfig, sources: dict[str, str]
) -> None:
    """Auto-detect or validate backends based on installed packages.

    - No config set the backend → auto-detect from installed extras.
    - Config explicitly set it → validate dependency is available.
    """
    pg_fields = ["metadata_backend", "vector_backend"]
    pg_requested = (
        cfg.store.vector_backend == "postgres"
        or cfg.store.metadata_backend == "postgres"
    )
    pg_explicit = any(f in sources for f in pg_fields)

    if pg_explicit and pg_requested and not _has_psycopg():
        # Config file explicitly asked for postgres but it's not installed
        src_files = sorted(set(sources[f] for f in pg_fields if f in sources))
        raise BackendMissingError(
            f"Backend 'postgres' was set in: {', '.join(src_files)}\n"
            f"but psycopg is not installed.\n"
            f'Install with: pip install "acatome-store[postgres]"\n'
            f"Or change the backend to 'sqlite'/'chroma' in your config."
        )

    if not pg_explicit:
        # No config file set backend — auto-detect from installed packages
        if _has_psycopg():
            cfg.store.metadata_backend = "postgres"
            cfg.store.vector_backend = "postgres"
        else:
            cfg.store.metadata_backend = "sqlite"
            cfg.store.vector_backend = "chroma"

    # Neo4j validation
    if cfg.store.graph_backend == "neo4j":
        neo4j_explicit = "graph_backend" in sources
        if not _has_neo4j():
            if neo4j_explicit:
                src = sources["graph_backend"]
                raise BackendMissingError(
                    f"Backend 'neo4j' was set in: {src}\n"
                    f"but neo4j driver is not installed.\n"
                    f'Install with: pip install "acatome-store[neo4j]"'
                )
            else:
                cfg.store.graph_backend = "none"


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

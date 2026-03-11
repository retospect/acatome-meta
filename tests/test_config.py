"""Tests for config loading."""

from __future__ import annotations

import os
import pytest
from pathlib import Path


def _has_psycopg() -> bool:
    try:
        import psycopg  # noqa: F401
        return True
    except ImportError:
        return False

from acatome_meta.config import (
    AcatomeConfig,
    BackendMissingError,
    EmbedProfile,
    load_config,
    ensure_config,
    _apply_toml,
    _apply_env,
)


class TestDefaults:
    def test_default_store(self):
        cfg = AcatomeConfig()
        assert cfg.store.vector_backend == "chroma"
        assert cfg.store.metadata_backend == "sqlite"
        assert cfg.store.pg_database == "acatome"

    def test_default_profiles(self):
        cfg = AcatomeConfig()
        assert "default" in cfg.extract.profiles
        assert "accurate" in cfg.extract.profiles
        assert cfg.extract.profiles["default"].dim == 384
        assert cfg.extract.profiles["accurate"].dim == 2560
        assert cfg.extract.profiles["accurate"].matryoshka is True

    def test_default_enrich(self):
        cfg = AcatomeConfig()
        assert cfg.extract.enrich.embed_profiles == ["default"]
        assert cfg.extract.enrich.summarize is True

    def test_store_path(self):
        cfg = AcatomeConfig()
        assert cfg.store_path == Path.home() / ".acatome" / "store"


class TestApplyToml:
    def test_store_postgres(self):
        cfg = AcatomeConfig()
        data = {
            "store": {
                "vector": {"backend": "postgres"},
                "postgres": {
                    "host": "db.example.com",
                    "port": 5433,
                    "password": "secret",
                },
            }
        }
        _apply_toml(cfg, data)
        assert cfg.store.vector_backend == "postgres"
        assert cfg.store.pg_host == "db.example.com"
        assert cfg.store.pg_port == 5433
        assert cfg.store.pg_password == "secret"

    def test_extract_enrich(self):
        cfg = AcatomeConfig()
        data = {
            "extract": {
                "verify": False,
                "enrich": {
                    "embed_profiles": ["default", "accurate"],
                    "summarizer": "openai:gpt-4o-mini",
                },
            }
        }
        _apply_toml(cfg, data)
        assert cfg.extract.verify is False
        assert cfg.extract.enrich.embed_profiles == ["default", "accurate"]
        assert cfg.extract.enrich.summarizer == "openai:gpt-4o-mini"

    def test_custom_profile(self):
        cfg = AcatomeConfig()
        data = {
            "extract": {
                "profiles": {
                    "custom": {
                        "model": "my-model",
                        "dim": 512,
                        "provider": "sentence-transformers",
                    }
                }
            }
        }
        _apply_toml(cfg, data)
        assert "custom" in cfg.extract.profiles
        assert cfg.extract.profiles["custom"].dim == 512

    def test_api_keys(self):
        cfg = AcatomeConfig()
        data = {"api": {"s2_api_key": "abc123", "crossref_mailto": "me@test.com"}}
        _apply_toml(cfg, data)
        assert cfg.api.s2_api_key == "abc123"
        assert cfg.api.crossref_mailto == "me@test.com"

    def test_anthropic_api_key(self):
        cfg = AcatomeConfig()
        data = {"api": {"anthropic_api_key": "sk-ant-test"}}
        _apply_toml(cfg, data)
        assert cfg.api.anthropic_api_key == "sk-ant-test"

    def test_default_summarizer(self):
        cfg = AcatomeConfig()
        assert cfg.extract.enrich.summarizer == "ollama/qwen3.5:9b"


class TestApplyEnv:
    def test_pg_password(self, monkeypatch):
        cfg = AcatomeConfig()
        monkeypatch.setenv("ACATOME_PG_PASSWORD", "envpass")
        _apply_env(cfg)
        assert cfg.store.pg_password == "envpass"

    def test_pg_port(self, monkeypatch):
        cfg = AcatomeConfig()
        monkeypatch.setenv("ACATOME_PG_PORT", "5433")
        _apply_env(cfg)
        assert cfg.store.pg_port == 5433

    def test_s2_api_key(self, monkeypatch):
        cfg = AcatomeConfig()
        monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "s2key")
        _apply_env(cfg)
        assert cfg.api.s2_api_key == "s2key"

    def test_store_path(self, monkeypatch):
        cfg = AcatomeConfig()
        monkeypatch.setenv("ACATOME_STORE_PATH", "/tmp/custom_store")
        _apply_env(cfg)
        assert cfg.store.path == "/tmp/custom_store"

    def test_anthropic_api_key_env(self, monkeypatch):
        cfg = AcatomeConfig()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        _apply_env(cfg)
        assert cfg.api.anthropic_api_key == "sk-ant-env"

    def test_env_overrides_toml(self, monkeypatch):
        cfg = AcatomeConfig()
        _apply_toml(cfg, {"store": {"postgres": {"password": "toml_pw"}}})
        monkeypatch.setenv("ACATOME_PG_PASSWORD", "env_pw")
        _apply_env(cfg)
        assert cfg.store.pg_password == "env_pw"


class TestLoadConfig:
    def test_load_defaults_autodetect(self, tmp_path, monkeypatch):
        """With no config file, backends are auto-detected from installed packages."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("acatome_meta.config.ACATOME_HOME", tmp_path / ".acatome")
        monkeypatch.delenv("ACATOME_CONFIG", raising=False)
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        cfg = load_config()
        if _has_psycopg():
            assert cfg.store.vector_backend == "postgres"
            assert cfg.store.metadata_backend == "postgres"
        else:
            assert cfg.store.vector_backend == "chroma"
            assert cfg.store.metadata_backend == "sqlite"

    def test_load_local_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml_file = tmp_path / "acatome.toml"
        toml_file.write_text(
            '[store]\npath = "/tmp/test_store"\n'
            '[api]\ncrossref_mailto = "local@test.com"\n'
        )
        cfg = load_config()
        assert cfg.store.path == "/tmp/test_store"
        assert cfg.api.crossref_mailto == "local@test.com"

    def test_acatome_config_env_var(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        custom = tmp_path / "custom.toml"
        custom.write_text('[api]\ns2_api_key = "from_custom"\n')
        monkeypatch.setenv("ACATOME_CONFIG", str(custom))
        cfg = load_config()
        assert cfg.api.s2_api_key == "from_custom"

    def test_acatome_config_overrides_local(self, tmp_path, monkeypatch):
        """$ACATOME_CONFIG has higher priority than ./acatome.toml."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        local = tmp_path / "acatome.toml"
        local.write_text('[api]\ns2_api_key = "local"\n')
        custom = tmp_path / "custom.toml"
        custom.write_text('[api]\ns2_api_key = "custom"\n')
        monkeypatch.setenv("ACATOME_CONFIG", str(custom))
        cfg = load_config()
        assert cfg.api.s2_api_key == "custom"


class TestBackendValidation:
    @pytest.mark.skipif(
        not _has_psycopg(), reason="psycopg not installed"
    )
    def test_postgres_ok_when_installed(self, tmp_path, monkeypatch):
        """No error when psycopg is available."""
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "acatome.toml"
        toml.write_text('[store.vector]\nbackend = "postgres"\n')
        cfg = load_config()
        assert cfg.store.vector_backend == "postgres"

    def test_postgres_explicit_missing_shows_source(self, tmp_path, monkeypatch):
        """Config explicitly sets postgres but psycopg missing → error names the file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("acatome_meta.config.ACATOME_HOME", tmp_path / ".acatome")
        monkeypatch.delenv("ACATOME_CONFIG", raising=False)
        monkeypatch.setattr("acatome_meta.config._has_psycopg", lambda: False)
        toml = tmp_path / "acatome.toml"
        toml.write_text('[store]\nmetadata_backend = "postgres"\n')
        with pytest.raises(BackendMissingError, match="acatome.toml"):
            load_config()

    def test_neo4j_explicit_missing_shows_source(self, tmp_path, monkeypatch):
        """Config explicitly sets neo4j but driver missing → error names the file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("acatome_meta.config.ACATOME_HOME", tmp_path / ".acatome")
        monkeypatch.delenv("ACATOME_CONFIG", raising=False)
        toml = tmp_path / "acatome.toml"
        toml.write_text('[store.graph]\nbackend = "neo4j"\n')
        with pytest.raises(BackendMissingError, match="acatome.toml"):
            load_config()

    def test_validate_false_skips_check(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "acatome.toml"
        toml.write_text('[store.graph]\nbackend = "neo4j"\n')
        cfg = load_config(validate=False)  # Should not raise
        assert cfg.store.graph_backend == "neo4j"


class TestEnsureConfig:
    def test_creates_config_file(self, tmp_path, monkeypatch):
        """ensure_config() writes a config file if none exists."""
        fake_home = tmp_path / ".acatome"
        monkeypatch.setattr("acatome_meta.config.ACATOME_HOME", fake_home)
        path = ensure_config()
        assert path.exists()
        content = path.read_text()
        assert "[store]" in content
        if _has_psycopg():
            assert "postgres" in content
        else:
            assert "sqlite" in content
            assert "chroma" in content

    def test_does_not_overwrite(self, tmp_path, monkeypatch):
        """ensure_config() leaves existing config alone."""
        fake_home = tmp_path / ".acatome"
        fake_home.mkdir(parents=True)
        config = fake_home / "config.toml"
        config.write_text("# my custom config\n")
        monkeypatch.setattr("acatome_meta.config.ACATOME_HOME", fake_home)
        path = ensure_config()
        assert path.read_text() == "# my custom config\n"

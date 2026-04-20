# Changelog

All notable changes to **acatome-meta** will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.5] — 2026-04-20

### Changed

- **BREAKING (behavior)**: when `[store]` in a config file explicitly sets
  `metadata_backend = "postgres"` or `vector_backend = "postgres"` and
  `psycopg` is not installed, `load_config()` now raises
  `BackendMissingError` instead of warning and silently falling back to
  SQLite/Chroma. Silent fallback corrupted intent — writes landed in the
  wrong backend while callers believed they had succeeded. The error names
  the config file that selected postgres and the install command to fix it,
  mirroring the existing behavior for `graph_backend = "neo4j"`.

## [0.1.0] — 2026-03-11

### Added

- Initial release.

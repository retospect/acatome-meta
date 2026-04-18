"""Tests for shared literature helpers."""

from __future__ import annotations

import pytest

from acatome_meta.literature import (
    SKIP_EMBED_TYPES,
    EmbedderUnavailableError,
    build_embedder,
    first_author_key,
    first_author_surname,
    make_slug,
)


class TestSkipEmbedTypes:
    def test_is_frozenset(self):
        assert isinstance(SKIP_EMBED_TYPES, frozenset)

    def test_expected_members(self):
        assert SKIP_EMBED_TYPES == frozenset(
            {"section_header", "title", "author", "equation", "junk"}
        )


class TestFirstAuthorKey:
    def test_list_of_dicts_comma_first(self):
        assert first_author_key([{"name": "Smith, John"}]) == "Smith"

    def test_list_of_dicts_first_last(self):
        assert first_author_key([{"name": "Daniel S. Levine"}]) == "Daniel S. Levine"

    def test_semicolon_packed(self):
        authors = [{"name": "Daniel S. Levine; Nicholas Liesen; Lauren Chua"}]
        assert first_author_key(authors) == "Daniel S. Levine"

    def test_list_of_strings(self):
        assert first_author_key(["Zou, Jiawen"]) == "Zou"

    def test_json_string(self):
        assert first_author_key('[{"name": "Müller, Hans"}]') == "Müller"

    def test_empty_list(self):
        assert first_author_key([]) == ""

    def test_none(self):
        assert first_author_key(None) == ""

    def test_malformed_json(self):
        assert first_author_key("not-json") == ""

    def test_missing_name_key(self):
        assert first_author_key([{}]) == ""


class TestFirstAuthorSurname:
    def test_last_first(self):
        assert first_author_surname([{"name": "Smith, John"}]) == "Smith"

    def test_first_last(self):
        assert first_author_surname([{"name": "John Smith"}]) == "Smith"

    def test_first_middle_last(self):
        assert first_author_surname([{"name": "Daniel S. Levine"}]) == "Levine"

    def test_preserves_case(self):
        assert first_author_surname([{"name": "Müller, Hans"}]) == "Müller"

    def test_json_string_input(self):
        assert first_author_surname('[{"name": "Zou, Jiawen"}]') == "Zou"

    def test_empty(self):
        assert first_author_surname([]) == ""


class TestMakeSlug:
    """Mirrors the acatome_extract.ids test suite to keep behaviour stable."""

    def test_basic(self):
        assert (
            make_slug([{"name": "Smith, John"}], 2024, "Quantum Error Correction")
            == "smith2024quantum"
        )

    def test_skip_stopwords(self):
        assert (
            make_slug([{"name": "Jones"}], 2023, "A New Approach to Surface Codes")
            == "jones2023approach"
        )

    def test_no_author(self):
        assert make_slug([], 2020, "Thermal Decomposition") == "anon2020thermal"

    def test_no_year(self):
        assert make_slug([{"name": "Doe, Jane"}], None, "Some Title") == "doe0000some"

    def test_accented_name(self):
        assert (
            make_slug([{"name": "Müller, Hans"}], 2021, "Chicken Little")
            == "muller2021chicken"
        )

    def test_empty_title(self):
        assert make_slug([{"name": "Smith"}], 2024, "") == "smith2024untitled"

    def test_semicolon_separated_authors(self):
        authors = [{"name": "Daniel S. Levine; Nicholas Liesen; Lauren Chua"}]
        assert make_slug(authors, 2026, "Open Polymers Dataset") == "danielslevine2026open"

    def test_surname_length_cap(self):
        authors = [{"name": "Superlongauthornamethatgoesforeverandever"}]
        slug = make_slug(authors, 2024, "Test")
        assert len(slug) < 50

    def test_chinese_title_deterministic(self):
        a = make_slug([], 2023, "零间隙CO₂电解实验研究")
        b = make_slug([], 2023, "零间隙CO₂电解实验研究")
        assert a == b

    def test_non_latin_author_latin_title(self):
        assert (
            make_slug([{"name": "田中太郎"}], 2022, "Thermal Analysis")
            == "anon2022thermal"
        )


class TestBuildEmbedder:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            build_embedder("huggingface-hub")

    def test_sentence_transformers_requires_model(self):
        # Skip if the backend is unavailable — the ImportError path is tested separately.
        pytest.importorskip("sentence_transformers")
        with pytest.raises(ValueError, match="requires a model name"):
            build_embedder("sentence-transformers", model="")

    def test_error_is_import_error_subclass(self):
        # EmbedderUnavailableError should be catchable as ImportError.
        assert issubclass(EmbedderUnavailableError, ImportError)

"""Tests for lookup module — arxiv filename extraction and author parsing."""

from __future__ import annotations

from acatome_meta.lookup import _extract_arxiv_from_filename, _parse_author_string


class TestExtractArxivFromFilename:
    def test_standard_arxiv(self):
        assert _extract_arxiv_from_filename("/papers/2508.20254v1.pdf") == "2508.20254"

    def test_arxiv_with_version(self):
        assert _extract_arxiv_from_filename("/papers/2310.18288v3.pdf") == "2310.18288"

    def test_arxiv_no_version(self):
        assert _extract_arxiv_from_filename("/papers/2601.16955.pdf") == "2601.16955"

    def test_arxiv_with_timestamp_suffix(self):
        assert (
            _extract_arxiv_from_filename("/papers/2504.02767v1_20260402224204.pdf")
            == "2504.02767"
        )

    def test_non_arxiv_filename(self):
        assert _extract_arxiv_from_filename("/papers/smith2024catalyst.pdf") is None

    def test_doi_style_filename(self):
        assert _extract_arxiv_from_filename("/papers/s41557-025-01815-x.pdf") is None

    def test_ssrn_filename(self):
        assert _extract_arxiv_from_filename("/papers/ssrn-5409063.pdf") is None

    def test_five_digit_arxiv(self):
        assert _extract_arxiv_from_filename("/papers/2603.29152v1.pdf") == "2603.29152"

    def test_arxiv_with_page_suffix(self):
        """Filenames like 2603.29152v1-4.pdf (pages) — still extract the ID."""
        # The regex matches at the start, so this should work
        assert _extract_arxiv_from_filename("/papers/2603.29152v1-4.pdf") == "2603.29152"


class TestParseAuthorString:
    def test_empty(self):
        assert _parse_author_string("") == []

    def test_whitespace(self):
        assert _parse_author_string("   ") == []

    def test_single_author(self):
        assert _parse_author_string("Smith, John") == [{"name": "Smith, John"}]

    def test_semicolon_separated(self):
        result = _parse_author_string("Daniel S. Levine; Nicholas Liesen; Lauren Chua")
        assert len(result) == 3
        assert result[0] == {"name": "Daniel S. Levine"}
        assert result[2] == {"name": "Lauren Chua"}

    def test_and_separated(self):
        result = _parse_author_string("Smith, John and Doe, Jane")
        assert len(result) == 2
        assert result[0] == {"name": "Smith, John"}
        assert result[1] == {"name": "Doe, Jane"}

    def test_none_input(self):
        assert _parse_author_string(None) == []

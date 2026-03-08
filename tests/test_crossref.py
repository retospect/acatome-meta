"""Tests for CrossRef metadata lookup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from acatome_meta.crossref import _normalize, lookup_crossref


class TestCrossrefNormalize:
    def test_normalize_full(self, sample_crossref_response):
        msg = sample_crossref_response["message"]
        result = _normalize(msg, "10.1038/s41567-024-1234-5")
        assert result["title"] == "Quantum Error Correction in Practice"
        assert len(result["authors"]) == 2
        assert result["authors"][0]["name"] == "Smith, John"
        assert result["year"] == 2024
        assert result["journal"] == "Nature Physics"
        assert result["source"] == "crossref"

    def test_normalize_missing_authors(self):
        msg = {"title": ["A Paper"], "type": "article"}
        result = _normalize(msg, "10.1038/test")
        assert result["authors"] == []
        assert result["title"] == "A Paper"

    def test_normalize_empty_title(self):
        msg = {"author": [{"family": "Smith"}], "type": "article"}
        result = _normalize(msg, "10.1038/test")
        assert result["title"] == ""


class TestCrossrefLookup:
    @patch("acatome_meta.crossref.Crossref")
    def test_lookup_success(self, mock_cr_cls, sample_crossref_response):
        mock_cr = MagicMock()
        mock_cr.works.return_value = sample_crossref_response
        mock_cr_cls.return_value = mock_cr

        result = lookup_crossref("10.1038/s41567-024-1234-5")
        assert result is not None
        assert result["title"] == "Quantum Error Correction in Practice"

    @patch("acatome_meta.crossref.Crossref")
    def test_lookup_not_found(self, mock_cr_cls):
        mock_cr = MagicMock()
        mock_cr.works.return_value = None
        mock_cr_cls.return_value = mock_cr

        result = lookup_crossref("10.1038/nonexistent")
        assert result is None

    @patch("acatome_meta.crossref.Crossref")
    def test_lookup_exception(self, mock_cr_cls):
        mock_cr = MagicMock()
        mock_cr.works.side_effect = Exception("network error")
        mock_cr_cls.return_value = mock_cr

        result = lookup_crossref("10.1038/error")
        assert result is None

    @patch("acatome_meta.crossref.Crossref")
    def test_mailto_passed(self, mock_cr_cls):
        mock_cr = MagicMock()
        mock_cr.works.return_value = None
        mock_cr_cls.return_value = mock_cr

        lookup_crossref("10.1038/test", mailto="test@example.com")
        mock_cr_cls.assert_called_once_with(mailto="test@example.com")

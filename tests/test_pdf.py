"""Tests for PDF metadata extraction."""

from __future__ import annotations

from acatome_meta.pdf import _clean_doi, _extract_doi


class TestDOIExtraction:
    def test_doi_from_xmp(self):
        xmp = "<dc:identifier>doi:10.1038/s41567-024-1234-5</dc:identifier>"
        assert _extract_doi(xmp, "", {}) == "10.1038/s41567-024-1234-5"

    def test_doi_from_first_page(self):
        text = "Published: https://doi.org/10.1145/1234567.1234568\nAbstract..."
        assert _extract_doi("", text, {}) == "10.1145/1234567.1234568"

    def test_doi_from_info_dict(self):
        info = {"doi": "10.1103/PhysRevLett.123.456"}
        assert _extract_doi("", "", info) == "10.1103/PhysRevLett.123.456"

    def test_doi_cascade_priority(self):
        xmp = "<dc:identifier>doi:10.1038/xmp-doi</dc:identifier>"
        text = "doi:10.1038/text-doi"
        info = {"doi": "10.1038/info-doi"}
        assert _extract_doi(xmp, text, info) == "10.1038/xmp-doi"

    def test_no_doi_found(self):
        assert _extract_doi("", "no doi here", {}) is None

    def test_clean_doi_trailing_punct(self):
        assert _clean_doi("10.1038/s41567-024-1234-5.") == "10.1038/s41567-024-1234-5"
        assert _clean_doi("10.1038/s41567-024-1234-5,") == "10.1038/s41567-024-1234-5"
        assert _clean_doi("10.1038/s41567-024-1234-5") == "10.1038/s41567-024-1234-5"

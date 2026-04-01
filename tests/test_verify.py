"""Tests for metadata verification."""

from __future__ import annotations

from acatome_meta.verify import verify_metadata


class TestVerifyMetadata:
    def test_matching_title_and_author(self):
        header = {
            "title": "Quantum Error Correction",
            "authors": [{"name": "Smith, John"}],
        }
        text = "Quantum Error Correction\nJohn Smith\nDepartment of Physics"
        verified, warnings = verify_metadata(header, text)
        assert verified is True
        assert warnings == []

    def test_mismatched_title(self):
        header = {
            "title": "Completely Different Paper Title",
            "authors": [{"name": "Smith, John"}],
        }
        text = "Quantum Error Correction\nJohn Smith"
        verified, warnings = verify_metadata(header, text)
        assert verified is False
        assert any("Title mismatch" in w for w in warnings)

    def test_mismatched_author(self):
        header = {
            "title": "Quantum Error Correction",
            "authors": [{"name": "Zzzzynski, Xander"}],
        }
        text = "Quantum Error Correction\nJohn Smith"
        verified, warnings = verify_metadata(header, text)
        assert verified is False
        assert any("Author surname" in w for w in warnings)

    def test_empty_header(self):
        header = {"title": "", "authors": []}
        text = "Some text"
        verified, warnings = verify_metadata(header, text)
        assert verified is True
        assert warnings == []

    def test_custom_threshold(self):
        header = {
            "title": "Quantum Error Correction",
            "authors": [],
        }
        text = "quantum error corrections"
        verified, warnings = verify_metadata(header, text, threshold=95)
        assert verified is True  # partial_ratio is lenient

    def test_unicode_dash_in_title(self):
        """Title with Unicode hyphen (U+2010) should match ASCII hyphen in PDF."""
        header = {
            "title": "Heterogeneous single\u2010atom catalysis",
            "authors": [{"name": "Wang, Aiqin"}],
        }
        text = "Heterogeneous single-atom catalysis\nAiqin Wang"
        verified, warnings = verify_metadata(header, text)
        assert verified is True
        assert warnings == []

    def test_en_dash_in_title(self):
        """En-dash (U+2013) in crossref title vs hyphen in PDF."""
        header = {
            "title": "Metal\u2013organic frameworks for CO2 capture",
            "authors": [],
        }
        text = "Metal-organic frameworks for CO2 capture"
        verified, warnings = verify_metadata(header, text)
        assert verified is True

    def test_subtitle_fallback(self):
        """If full title with subtitle fails, main title alone should pass."""
        header = {
            "title": "Single-atom catalysis: a concise review of recent advances",
            "authors": [],
        }
        # PDF only shows the main title
        text = "Single-atom catalysis\nAuthors and affiliations..."
        verified, warnings = verify_metadata(header, text)
        assert verified is True

    def test_subtitle_with_dash_separator(self):
        """Subtitle separated by ' - ' should also try main title."""
        header = {
            "title": "Carbon capture and storage - current status and future directions",
            "authors": [],
        }
        text = "Carbon capture and storage\nA. Smith et al."
        verified, warnings = verify_metadata(header, text)
        assert verified is True

    def test_genuine_mismatch_still_fails(self):
        """Real mismatches should still be caught after normalization."""
        header = {
            "title": "Completely unrelated paper about quantum computing",
            "authors": [],
        }
        text = "Heterogeneous single-atom catalysis\nWang et al."
        verified, warnings = verify_metadata(header, text)
        assert verified is False

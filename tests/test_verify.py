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

"""Tests for PDF metadata extraction."""

from __future__ import annotations

from acatome_meta.pdf import (
    _clean_doi,
    _extract_doi,
    _pii_to_doi,
    is_garbage_title,
    is_pii,
)


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

    def test_doi_from_pii_in_title(self):
        info = {"title": "PII: S0009-2614(95)00905-J"}
        assert _extract_doi("", "", info) == "10.1016/S0009-2614(95)00905-J"

    def test_doi_from_pii_without_prefix(self):
        info = {"title": "0009-2614(80)80221-1"}
        assert _extract_doi("", "", info) == "10.1016/0009-2614(80)80221-1"

    def test_doi_from_pii_in_subject(self):
        info = {"subject": "S0926-3373(98)00040-X"}
        assert _extract_doi("", "", info) == "10.1016/S0926-3373(98)00040-X"

    def test_real_doi_beats_pii(self):
        """DOI in XMP should win over PII in title."""
        xmp = "<dc:identifier>doi:10.1038/real-doi</dc:identifier>"
        info = {"title": "PII: S0009-2614(95)00905-J"}
        assert _extract_doi(xmp, "", info) == "10.1038/real-doi"

    def test_clean_doi_trailing_punct(self):
        assert _clean_doi("10.1038/s41567-024-1234-5.") == "10.1038/s41567-024-1234-5"
        assert _clean_doi("10.1038/s41567-024-1234-5,") == "10.1038/s41567-024-1234-5"
        assert _clean_doi("10.1038/s41567-024-1234-5") == "10.1038/s41567-024-1234-5"


class TestPII:
    def test_pii_to_doi_with_prefix(self):
        assert _pii_to_doi("PII: S0009-2614(95)00905-J") == "10.1016/S0009-2614(95)00905-J"

    def test_pii_to_doi_without_s(self):
        assert _pii_to_doi("0009-2614(80)80221-1") == "10.1016/0009-2614(80)80221-1"

    def test_pii_to_doi_busca(self):
        assert _pii_to_doi("PII: S0926-3373(98)00040-X") == "10.1016/S0926-3373(98)00040-X"

    def test_pii_to_doi_no_match(self):
        assert _pii_to_doi("The Grotthuss mechanism") is None

    def test_pii_to_doi_empty(self):
        assert _pii_to_doi("") is None

    def test_is_pii_true(self):
        assert is_pii("PII: S0009-2614(95)00905-J") is True
        assert is_pii("0009-2614(80)80221-1") is True

    def test_is_pii_false(self):
        assert is_pii("The Grotthuss mechanism") is False
        assert is_pii("") is False
        assert is_pii("Direct Electrochemical Ammonia Synthesis") is False


class TestGarbageTitle:
    """Detection of embedded-metadata titles that should not be trusted."""

    def test_page_range_suffix(self):
        # ACS / InDesign page-range notation leaked into dc:title
        assert is_garbage_title("nl404795z 1..9") is True
        assert is_garbage_title("LQ8388 2..5") is True
        assert is_garbage_title("acs_nn_nn-2013-02954e 1..6") is True
        assert is_garbage_title("78868 651..703") is True

    def test_indesign_source_filename(self):
        assert is_garbage_title("nmat1849 Geim Progress Article.indd") is True
        assert is_garbage_title("paper_draft.doc") is True
        assert is_garbage_title("manuscript.docx") is True
        assert is_garbage_title("source.tex") is True

    def test_revtex_boilerplate(self):
        # APS revtex \title{USING STANDARD PRB STYLE...} template leakage
        assert is_garbage_title("USING STANDARD PRB S") is True
        assert is_garbage_title("Using Standard PRB Style") is True

    def test_latex_source_leakage(self):
        assert is_garbage_title(r"\documentclass{revtex4-2}") is True
        assert is_garbage_title(r"\begin{document} Some text") is True

    def test_empty(self):
        assert is_garbage_title("") is True
        assert is_garbage_title("   ") is True

    def test_real_titles_pass(self):
        # Genuine paper titles — must NOT be flagged as garbage
        real = [
            "High-κ dielectrics for advanced carbon-nanotube transistors and logic gates",
            "The rise of graphene",
            "Addition of nanoparticle dispersions to enhance flux pinning of the YBa2Cu3O7-x superconductor",
            "Graphene/MoS2 Hybrid Technology for Large-Scale Two-Dimensional Electronics",
            "Carbon Nanotubes as Schottky Barrier Transistors",
            "Flexible and Transparent MoS2 Field-Effect Transistors on Hexagonal Boron Nitride–Graphene Heterostructures",
            "Direct Electrochemical Ammonia Synthesis",
        ]
        for title in real:
            assert is_garbage_title(title) is False, f"false positive: {title!r}"

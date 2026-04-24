"""PDF metadata extraction via PyMuPDF (fitz)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import fitz

DOI_REGEX = re.compile(r"10\.\d{4,}/[^\s<\"}\)]+")

# Elsevier PII patterns — appear in title/subject fields of pre-2000 PDFs.
# Formatted: S0009-2614(95)00905-J or 0009-2614(80)80221-1
# The PII *is* the DOI suffix for Elsevier: DOI = 10.1016/{PII}
_PII_RE = re.compile(
    r"(?:PII[:\s]*)?"               # optional "PII:" prefix
    r"(S?\d{4}-\d{3}[\dX]"          # ISSN part: S0009-2614 or 0009-2614
    r"\(\d{2}\)"                     # (95)
    r"\d{4,5}-[A-Z\d])"             # 00905-J  (check digit: letter or digit)
)


def extract_pdf_meta(path: str | Path) -> dict[str, Any]:
    """Extract metadata, DOI, and content hash from a PDF.

    Args:
        path: Path to the PDF file.

    Returns:
        Dict with keys: info, xmp, doi, pdf_hash, first_pages_text, page_count.
    """
    path = Path(path)
    doc = fitz.open(str(path))

    info = doc.metadata or {}

    try:
        xmp_raw = doc.get_xml_metadata() or ""
    except Exception:
        xmp_raw = ""

    # Content hash
    pdf_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    # First 3 pages text (for verification + DOI extraction)
    first_pages_text = ""
    for i in range(min(3, doc.page_count)):
        try:
            first_pages_text += doc[i].get_text() + "\n"
        except Exception:
            pass

    # DOI extraction cascade: XMP → first-page text → info dict
    doi = _extract_doi(xmp_raw, first_pages_text, info)

    page_count = doc.page_count
    doc.close()

    return {
        "info": info,
        "xmp": xmp_raw,
        "doi": doi,
        "pdf_hash": pdf_hash,
        "first_pages_text": first_pages_text,
        "page_count": page_count,
    }


def _extract_doi(xmp: str, first_pages: str, info: dict[str, Any]) -> str | None:
    """Extract DOI using the three-source cascade."""
    # 1. XMP XML
    if xmp:
        match = DOI_REGEX.search(xmp)
        if match:
            return _clean_doi(match.group())

    # 2. First-page text
    if first_pages:
        match = DOI_REGEX.search(first_pages)
        if match:
            return _clean_doi(match.group())

    # 3. Info dict fields
    for key in ("doi", "subject", "keywords"):
        val = info.get(key, "")
        if val:
            match = DOI_REGEX.search(val)
            if match:
                return _clean_doi(match.group())

    # 4. PII in title or subject → Elsevier DOI
    title = info.get("title", "")
    subject = info.get("subject", "")
    for field in (title, subject):
        if field:
            pii_doi = _pii_to_doi(field)
            if pii_doi:
                return pii_doi

    return None


def _pii_to_doi(text: str) -> str | None:
    """Extract Elsevier DOI from a PII string.

    PII like 'S0009-2614(95)00905-J' → DOI '10.1016/S0009-2614(95)00905-J'.
    """
    m = _PII_RE.search(text)
    if m:
        return f"10.1016/{m.group(1)}"
    return None


def is_pii(text: str) -> bool:
    """Return True if *text* looks like an Elsevier PII string (not a real title)."""
    if not text:
        return False
    return bool(_PII_RE.search(text))


# Garbage title patterns commonly found in PDF embedded metadata.
# These are NOT real paper titles — they're filenames, manuscript tracking
# IDs, or typesetting-template boilerplate that leaked into dc:title.
# Passing them to a title-based search engine (S2, CrossRef title fuzz)
# poisons the lookup and returns an unrelated paper with high confidence.
_GARBAGE_TITLE_RES = [
    # Ends with "N..M" page-range notation (InDesign / Quark XPress page refs).
    # Examples: "nl404795z 1..9", "LQ8388 2..5", "acs_nn_nn-2013-02954e 1..6",
    # "78868 651..703"
    re.compile(r"\s\d+\.\.\d+\s*$"),
    # Ends with a document-source filename extension.
    # Examples: "nmat1849 Geim Progress Article.indd"
    re.compile(r"\.(?:indd|doc|docx|tex|pdf|qxp|qxd|ai|xml|eps)\s*$", re.IGNORECASE),
    # APS/AIP revtex template boilerplate that leaked into dc:title.
    # Example: "USING STANDARD PRB S"
    re.compile(r"^\s*USING\s+STANDARD\b", re.IGNORECASE),
    # Raw LaTeX source leakage.
    re.compile(r"\\(?:documentclass|usepackage|begin\{|end\{)"),
]


def is_garbage_title(text: str) -> bool:
    """Return True if *text* is a known-bad PDF embedded title pattern.

    Distinct from :func:`is_pii`, which detects Elsevier PII identifiers.
    Real paper titles never match these patterns; embedded ``dc:title``
    fields populated by typesetting pipelines frequently do.

    Used to gate S2 title-based fallback lookups so they don't poison
    results with random plausible-but-wrong papers.
    """
    if not text or not text.strip():
        return True
    return any(p.search(text) for p in _GARBAGE_TITLE_RES)


def _clean_doi(doi: str) -> str:
    """Strip trailing punctuation from extracted DOI."""
    return doi.rstrip(".,;:")

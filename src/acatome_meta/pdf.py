"""PDF metadata extraction via PyMuPDF (fitz)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import fitz

DOI_REGEX = re.compile(r"10\.\d{4,}/[^\s<\"}\)]+")


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

    return None


def _clean_doi(doi: str) -> str:
    """Strip trailing punctuation from extracted DOI."""
    return doi.rstrip(".,;:")

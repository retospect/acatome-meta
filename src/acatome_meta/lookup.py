"""Metadata lookup cascade: DOI → CrossRef → S2 → embedded fallback."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from acatome_meta.crossref import lookup_crossref
from acatome_meta.pdf import extract_pdf_meta
from acatome_meta.semantic_scholar import get_paper_by_id, lookup_s2

# arXiv filename patterns: 2508.20254v1.pdf, 2310.18288v3.pdf, etc.
_ARXIV_FILENAME_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def lookup(pdf_path: str) -> dict[str, Any]:
    """Full metadata lookup cascade for a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Merged metadata dict with 'source' indicating provenance.
    """
    pdf_meta = extract_pdf_meta(pdf_path)
    doi = pdf_meta.get("doi")

    mailto = os.environ.get("ACATOME_CROSSREF_MAILTO", "")
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

    # Try DOI → CrossRef
    if doi:
        result = lookup_doi(doi, mailto=mailto)
        if result:
            result["pdf_hash"] = pdf_meta["pdf_hash"]
            result["page_count"] = pdf_meta["page_count"]
            result["first_pages_text"] = pdf_meta["first_pages_text"]
            return result

    # Try arxiv ID from filename → S2
    arxiv_id = _extract_arxiv_from_filename(pdf_path)
    if arxiv_id:
        result = get_paper_by_id(f"ARXIV:{arxiv_id}", api_key=s2_key)
        if result:
            result["pdf_hash"] = pdf_meta["pdf_hash"]
            result["page_count"] = pdf_meta["page_count"]
            result["first_pages_text"] = pdf_meta["first_pages_text"]
            if not result.get("arxiv_id"):
                result["arxiv_id"] = arxiv_id
            return result

    # Try title → S2
    title = pdf_meta.get("info", {}).get("title", "")
    if title:
        result = lookup_title(title, s2_key=s2_key)
        if result:
            result["pdf_hash"] = pdf_meta["pdf_hash"]
            result["page_count"] = pdf_meta["page_count"]
            result["first_pages_text"] = pdf_meta["first_pages_text"]
            if doi and not result.get("doi"):
                result["doi"] = doi
            return result

    # Fallback: embedded PDF metadata
    info = pdf_meta.get("info", {})
    return {
        "title": info.get("title", ""),
        "authors": _parse_author_string(info.get("author", "")),
        "year": _parse_year(info.get("creationDate", "")),
        "doi": doi,
        "journal": "",
        "abstract": "",
        "entry_type": "article",
        "source": "embedded",
        "pdf_hash": pdf_meta["pdf_hash"],
        "page_count": pdf_meta["page_count"],
        "first_pages_text": pdf_meta["first_pages_text"],
    }


def lookup_doi(doi: str, mailto: str = "") -> dict[str, Any] | None:
    """Look up metadata by DOI via CrossRef."""
    return lookup_crossref(doi, mailto=mailto)


def lookup_title(title: str, s2_key: str = "") -> dict[str, Any] | None:
    """Look up metadata by title via Semantic Scholar."""
    return lookup_s2(title, api_key=s2_key)


def _parse_author_string(author: str) -> list[dict[str, str]]:
    """Split a raw PDF author string into individual author dicts.

    Handles semicolon-separated, ' and '-separated, and single-author strings.
    Returns empty list for empty/whitespace-only input.
    """
    if not author or not author.strip():
        return []
    # Semicolon-separated (most common in embedded metadata)
    if ";" in author:
        parts = [p.strip() for p in author.split(";") if p.strip()]
    # " and " separated
    elif " and " in author.lower():
        parts = [p.strip() for p in re.split(r"\s+and\s+", author, flags=re.IGNORECASE) if p.strip()]
    else:
        parts = [author.strip()]
    return [{"name": p} for p in parts]


def _extract_arxiv_from_filename(pdf_path: str) -> str | None:
    """Extract arXiv ID from a PDF filename like '2508.20254v1.pdf'."""
    stem = Path(pdf_path).stem
    # Strip trailing timestamp suffixes like _20260402224204
    stem = re.sub(r"_\d{14}$", "", stem)
    m = _ARXIV_FILENAME_RE.match(stem)
    if m:
        return m.group(1)  # e.g. '2508.20254' without version suffix
    return None


def _parse_year(date_str: str) -> int | None:
    """Extract year from PDF date string like 'D:20240115...'."""
    if not date_str:
        return None
    # PDF dates: D:YYYYMMDDHHmmSS or just YYYY...
    clean = date_str.replace("D:", "").strip()
    if len(clean) >= 4 and clean[:4].isdigit():
        year = int(clean[:4])
        if 1900 <= year <= 2100:
            return year
    return None

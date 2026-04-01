"""Metadata verification against PDF text."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# Unicode dashes / hyphens that should all be treated as ASCII hyphen-minus
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]")


def _normalize(text: str) -> str:
    """Normalize text for fuzzy comparison.

    - NFKC unicode normalization (folds ligatures, superscripts, etc.)
    - Fold all dash/hyphen variants to ASCII hyphen-minus
    - Collapse whitespace
    - Lowercase
    """
    text = unicodedata.normalize("NFKC", text)
    text = _DASH_RE.sub("-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _title_score(title: str, text: str) -> float:
    """Best partial_ratio of title against text, trying subtitle variants."""
    score = fuzz.partial_ratio(title, text)
    # If full title didn't match well, try the main title (before : or —)
    for sep in (":", " - ", " — ", " – "):
        if sep in title:
            main = title.split(sep, 1)[0].strip()
            if len(main) >= 10:
                score = max(score, fuzz.partial_ratio(main, text))
    return score


def verify_metadata(
    header: dict, first_pages_text: str, threshold: int = 80
) -> tuple[bool, list[str]]:
    """Verify looked-up metadata against PDF text.

    Args:
        header: Metadata dict with 'title' and 'authors'.
        first_pages_text: Text from first 1-3 pages of the PDF.
        threshold: Minimum fuzzy match score (0-100).

    Returns:
        Tuple of (verified: bool, warnings: list[str]).
    """
    warnings: list[str] = []
    norm_text = _normalize(first_pages_text[:2000])

    title = header.get("title", "")
    if title:
        score = _title_score(_normalize(title), norm_text)
        if score < threshold:
            warnings.append(
                f"Title mismatch: '{title[:60]}...' scored {score} < {threshold}"
            )

    authors = header.get("authors", [])
    if authors:
        norm_text_5k = _normalize(first_pages_text[:5000])
        for author in authors:
            name = author.get("name", "")
            surname = (
                name.split(",")[0].strip()
                if "," in name
                else name.split()[-1]
                if name.split()
                else ""
            )
            if surname:
                score = fuzz.partial_ratio(_normalize(surname), norm_text_5k)
                if score < threshold:
                    warnings.append(
                        f"Author surname '{surname}' scored {score} < {threshold}"
                    )

    verified = len(warnings) == 0
    return verified, warnings

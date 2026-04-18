"""Metadata verification against PDF text."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from acatome_meta.literature import surname_from_name

# Unicode dashes / hyphens that should all be treated as ASCII hyphen-minus
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]")
# HTML tags (e.g. <sub>, <sup>, <i>) sometimes present in S2 / CrossRef titles
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalize(text: str) -> str:
    """Normalize text for fuzzy comparison.

    - NFKC unicode normalization (folds ligatures, superscripts, etc.)
    - Fold all dash/hyphen variants to ASCII hyphen-minus
    - Collapse whitespace
    - Lowercase
    """
    text = _HTML_TAG_RE.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = _DASH_RE.sub("-", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Rejoin chemical formulas split by HTML tag removal (CO 2 → CO2, H 2 O → H2O)
    text = re.sub(r"([A-Za-z])\s(\d)", r"\1\2", text)
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
        author_pass = 0
        author_checked = 0
        author_warnings: list[str] = []
        for author in authors:
            surname = surname_from_name(author.get("name", ""))
            if surname:
                author_checked += 1
                norm_surname = _normalize(surname)
                score = fuzz.partial_ratio(norm_surname, norm_text_5k)
                # Short surnames (≤4 chars) are inherently noisy with partial_ratio
                effective_threshold = 60 if len(norm_surname) <= 4 else threshold
                if score >= effective_threshold:
                    author_pass += 1
                else:
                    author_warnings.append(
                        f"Author surname '{surname}' scored {score} < {effective_threshold}"
                    )
        # Fail only if we checked authors and NONE matched
        if author_checked > 0 and author_pass == 0:
            warnings.extend(author_warnings)

    verified = len(warnings) == 0
    return verified, warnings

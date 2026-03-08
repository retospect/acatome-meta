"""Metadata verification against PDF text."""

from __future__ import annotations

from rapidfuzz import fuzz


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

    title = header.get("title", "")
    if title:
        score = fuzz.partial_ratio(title.lower(), first_pages_text[:2000].lower())
        if score < threshold:
            warnings.append(
                f"Title mismatch: '{title[:60]}...' scored {score} < {threshold}"
            )

    authors = header.get("authors", [])
    if authors:
        for author in authors:
            name = author.get("name", "")
            surname = (
                name.split(",")[0].strip()
                if "," in name
                else name.split()[-1] if name.split() else ""
            )
            if surname:
                # Check first 3 pages
                score = fuzz.partial_ratio(
                    surname.lower(), first_pages_text[:5000].lower()
                )
                if score < threshold:
                    warnings.append(
                        f"Author surname '{surname}' scored {score} < {threshold}"
                    )

    verified = len(warnings) == 0
    return verified, warnings

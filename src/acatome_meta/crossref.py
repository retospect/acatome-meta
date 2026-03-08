"""CrossRef metadata lookup via habanero."""

from __future__ import annotations

from typing import Any

from habanero import Crossref


def lookup_crossref(doi: str, mailto: str = "") -> dict[str, Any] | None:
    """Fetch metadata from CrossRef for a given DOI.

    Args:
        doi: The DOI to look up.
        mailto: Email for CrossRef polite pool (recommended).

    Returns:
        Normalized metadata dict or None if not found.
    """
    cr = Crossref(mailto=mailto) if mailto else Crossref()
    try:
        result = cr.works(ids=doi)
    except Exception:
        return None

    if not result or "message" not in result:
        return None

    msg = result["message"]
    return _normalize(msg, doi)


def _normalize(msg: dict[str, Any], doi: str) -> dict[str, Any]:
    """Normalize CrossRef response to acatome header format."""
    authors = []
    for a in msg.get("author", []):
        name_parts = []
        if a.get("family"):
            name_parts.append(a["family"])
        if a.get("given"):
            name_parts.append(a["given"])
        authors.append({"name": ", ".join(name_parts)})

    year = None
    for date_field in ("published-print", "published-online", "created"):
        parts = msg.get(date_field, {}).get("date-parts", [[]])
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break

    title_list = msg.get("title", [])
    title = title_list[0] if title_list else ""

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "journal": (
            msg.get("container-title", [""])[0] if msg.get("container-title") else ""
        ),
        "abstract": msg.get("abstract", ""),
        "entry_type": msg.get("type", "article"),
        "source": "crossref",
    }

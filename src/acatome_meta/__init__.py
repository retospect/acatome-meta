"""acatome-meta: Lightweight metadata lookup for scientific papers."""

from importlib.metadata import version

from acatome_meta.citations import citations
from acatome_meta.lookup import lookup, lookup_doi, lookup_title

__all__ = ["citations", "lookup", "lookup_doi", "lookup_title"]
__version__ = version("acatome-meta")

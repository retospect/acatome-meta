"""acatome-meta: Lightweight metadata lookup for scientific papers."""

from acatome_meta.lookup import lookup, lookup_doi, lookup_title
from acatome_meta.citations import citations

__all__ = ["lookup", "lookup_doi", "lookup_title", "citations"]
__version__ = "0.2.2"

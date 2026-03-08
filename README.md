# acatome-meta

Lightweight metadata lookup and verification for scientific papers. Resolves PDFs to their DOI, title, authors, and journal using Crossref and Semantic Scholar.

## Features

- **PDF title extraction** — extracts candidate titles from PDF first pages
- **Crossref lookup** — DOI resolution and metadata retrieval
- **Semantic Scholar** — citation counts, abstracts, and supplementary metadata
- **Fuzzy verification** — confirms extracted metadata matches the PDF content
- **Citation parsing** — extracts structured author/year/title from reference strings
- **Zero-config** — works out of the box, optional API keys for higher rate limits

## Installation

```bash
uv pip install -e .
```

## Usage

```python
from acatome_meta import lookup

meta = lookup("/path/to/paper.pdf")
print(meta["doi"], meta["title"], meta["authors"])
```

## CLI

```bash
acatome-meta lookup paper.pdf
acatome-meta verify paper.pdf --doi 10.1234/example
```

## Configuration

Set `SEMANTIC_SCHOLAR_API_KEY` for higher rate limits:

```bash
export SEMANTIC_SCHOLAR_API_KEY=your_key_here
```

## Testing

```bash
uv run python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

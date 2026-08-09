# okf-graphrag

A consumer tool for [OKF (Open Knowledge Format)](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf) that ingests OKF bundles into a hybrid retrieval system — vector search (Qdrant) + graph traversal (Neo4j) — so agents can query OKF knowledge instead of just loading files flat into context.

## Why

OKF represents knowledge as markdown files with YAML frontmatter, cross-linked into a graph via normal markdown links. That graph structure is currently only realized visually, in the bundled viewer. `okf-graphrag` realizes it operationally: as a queryable graph combined with semantic search over concept bodies.

## Status

- ✅ **Parser** — extracts frontmatter, body, and links (including anchored links like `file.md#section`) from OKF concept and index files. Handles `index.md` files with no frontmatter as link-only nodes. Tested against the full `knowledge-catalog` repo (129 files) and the `ga4` sample bundle (14/14 parsed cleanly).
- ⬜ Neo4j loader — load parsed concepts as graph nodes, links as edges
- ⬜ Qdrant loader — embed concept bodies for semantic search
- ⬜ Retrieval layer — RRF fusion of vector + graph results via LangGraph

## Usage

Parse a single file:
```bash
python parser.py path/to/concept.md
```

Parse an entire bundle:
```bash
python parser.py --bundle path/to/bundle_dir
```

## Design notes

- `index.md` files without frontmatter are treated as link-only structural nodes (`is_index: true`) rather than skipped, since they carry real graph edges.
- Link extraction matches `.md` targets, including an optional trailing anchor (`file.md#section`) — the anchor is preserved in the returned target, not stripped.
- External links (non-`.md`) are intentionally excluded from the concept graph.

## License

Apache 2.0

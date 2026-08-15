# okf-graphrag

A consumer tool for [OKF (Open Knowledge Format)](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf) that ingests OKF bundles into a hybrid retrieval system — vector search (Qdrant) + graph traversal (Neo4j) — so agents can query OKF knowledge instead of just loading files flat into context.

## Why

OKF represents knowledge as markdown files with YAML frontmatter, cross-linked into a graph via normal markdown links. That graph structure is currently only realized visually, in the bundled viewer. `okf-graphrag` realizes it operationally: as a queryable graph combined with semantic search over concept bodies.

## Status

- ✅ **Parser** — extracts frontmatter, body, and links (including anchored links like `file.md#section`) from OKF concept and index files. Handles `index.md` files with no frontmatter as link-only nodes. Tested against the full `knowledge-catalog` repo (129 files) and the `ga4` sample bundle (14/14 parsed cleanly).
- ✅ **Neo4j loader** — loads parsed concepts as `:Concept` nodes and markdown links as `:LINKS_TO` edges. Idempotent (`MERGE`, not `CREATE`), so re-running updates in place. Verified against the `ga4` sample bundle: 14 nodes, 21 relationships.
- ✅ **Qdrant loader** — embeds concept bodies with `all-MiniLM-L6-v2` and upserts them into the `okf_concepts` collection. Idempotent (point ids are derived deterministically from the concept id). Verified against the `ga4` sample bundle: 14 points, keyed identically to the 14 Neo4j nodes.
- ⬜ Retrieval layer — RRF fusion of vector + graph results via LangGraph

## Setup

```bash
pip install pyyaml neo4j python-dotenv qdrant-client sentence-transformers
```

The Neo4j loader reads its connection details from environment variables, loaded
from a `.env` file in the project root (git-ignored):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

The Qdrant loader reads `QDRANT_URL`, defaulting to `http://localhost:6333` if
it isn't set. It needs no credentials. On first run it downloads the
`all-MiniLM-L6-v2` embedding model from the Hugging Face Hub and creates the
`okf_concepts` collection at that model's dimension.

## Usage

Parse a single file:
```bash
python parser.py path/to/concept.md
```

Parse an entire bundle:
```bash
python parser.py --bundle path/to/bundle_dir
```

Load a bundle into Neo4j:
```bash
python neo4j_loader.py path/to/bundle_dir
```

It prints how many nodes and relationships were merged, e.g.
`14 nodes merged, 21 relationships merged`. Paths are relative to the current
working directory — from the project root, the sample bundle is
`knowledge-catalog/okf/bundles/ga4`.

Re-running is safe: nodes are keyed on a bundle-relative id (`ga4/index.md`), so
a second load updates the same nodes rather than duplicating them — from any
working directory, relative path or absolute.

Inspect the result in Neo4j:
```cypher
MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) RETURN a.path, b.path
```

Load the same bundle into Qdrant:
```bash
python qdrant_loader.py path/to/bundle_dir
```

It prints how many concepts were embedded, e.g.
`14 concepts embedded and upserted into 'okf_concepts'`. Re-running is safe for
the same reason Neo4j is: each point id is derived from the concept id, so a
second load overwrites the same points.

Each point's `path` payload field is the same bundle-relative id the Neo4j nodes
are keyed on, so a vector hit joins straight onto the graph:
```cypher
MATCH (c:Concept {path: $path_from_qdrant})-[:LINKS_TO]->(n) RETURN n.path
```

## Design notes

- `index.md` files without frontmatter are treated as link-only structural nodes (`is_index: true`) rather than skipped, since they carry real graph edges.
- Link extraction matches `.md` targets, including an optional trailing anchor (`file.md#section`) — the anchor is preserved in the returned target, not stripped.
- External links (non-`.md`) are intentionally excluded from the concept graph.
- Each `:Concept` node is keyed on a **bundle-relative id** — the path relative to
  the bundle root, prefixed with the bundle directory name and normalized to
  forward slashes (`ga4/references/metrics/purchasers.md`). This id is
  independent of the working directory the loader ran from and of the host OS,
  so re-runs update nodes in place instead of duplicating them, and the same
  bundle keys identically on any machine. Downstream stores (Qdrant) should use
  this id to reference graph nodes.
- The on-disk location is kept as a separate `source_path` property. Nothing
  keys on it, so it can safely differ between machines.
- The loader stores every frontmatter field as an additional property. Values Neo4j can't store natively (nested
  dicts, mixed-type lists) are JSON-serialized; a frontmatter key named `path`
  is dropped so it can't overwrite the node id.
- Link targets are written relative to the linking file, so the loader resolves
  each one against its source directory and strips any `#anchor` before matching.
  Edges are only created when the target file is part of the same batch —
  dangling links are silently skipped rather than creating empty nodes.
- Bundle names come from the bundle directory's own name, so two bundles loaded
  into the same database stay distinct even though both contain `index.md`.
  Loading two *different* directories that share a basename would collide.
- Qdrant point ids must be an unsigned integer or a UUID, so they can't be the
  concept id itself. Each id is a UUIDv5 of the concept id under a fixed
  namespace — deterministic, so re-runs upsert rather than duplicate — and the
  readable id is kept in the payload as `path`.
- The embedded text is the frontmatter `description`, when present, followed by
  the body, so the one-line summary is weighted alongside the full content.
- Frontmatter is copied into the payload as-is, minus `path` and `source_path`
  so neither can overwrite the join key or the on-disk location. Dates are
  ISO-formatted and anything else non-JSON-serializable falls back to its string
  form.

## License

Apache 2.0

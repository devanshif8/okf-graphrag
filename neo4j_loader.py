"""Load parsed OKF bundles into Neo4j as :Concept nodes and :LINKS_TO edges."""

import argparse
import datetime
import json
import logging
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable

load_dotenv()

# The driver logs connection failures to stderr via logging's last-resort
# handler, duplicating the messages main() raises. Silence it unless the caller
# configures logging themselves.
logging.getLogger("neo4j").addHandler(logging.NullHandler())

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

PRIMITIVES = (str, int, float, bool, datetime.date, datetime.datetime)


def _property_value(value):
    """Coerce a frontmatter value into something the driver can store."""
    if value is None or isinstance(value, PRIMITIVES):
        return value
    if isinstance(value, (list, tuple)):
        items = [_property_value(v) for v in value]
        if all(isinstance(i, PRIMITIVES) for i in items):
            return items
    return json.dumps(value, ensure_ascii=False, default=str)


def _properties(frontmatter: dict) -> dict:
    """Flatten frontmatter into Neo4j-safe properties, minus a colliding path key."""
    return {
        str(k): _property_value(v)
        for k, v in (frontmatter or {}).items()
        if str(k) != "path"
    }


def _resolve_target(source_path: str, target: str) -> str:
    """Resolve a link target (relative, possibly anchored) against its source file."""
    target = target.split("#", 1)[0]
    return os.path.normpath(os.path.join(os.path.dirname(source_path), target))


def _bundle_name(bundle_root: str) -> str:
    """Name the bundle after its directory, e.g. '.../bundles/ga4' -> 'ga4'."""
    return os.path.basename(os.path.abspath(bundle_root).rstrip(os.sep)) or "bundle"


def concept_id(path: str, bundle_root: str) -> str:
    """Stable node id: '<bundle>/<path relative to bundle root>', forward slashes.

    Independent of the working directory the loader ran from and of the host OS,
    so the same file keys to the same node on every run and every machine.
    """
    relative = os.path.relpath(os.path.abspath(path), os.path.abspath(bundle_root))
    return f"{_bundle_name(bundle_root)}/{relative.replace(os.sep, '/')}"


def bundle_prefix(bundle_root: str) -> str:
    """The id prefix every concept in this bundle shares, e.g. 'ga4/'.

    Concept ids are '<bundle>/<relative path>', so this is what identifies the
    bundle's own nodes in a database that may hold several bundles.
    """
    return f"{_bundle_name(bundle_root)}/"


def bundle_edges(concepts: list[dict], bundle_root: str) -> list[tuple[str, str]]:
    """Every :LINKS_TO edge the bundle's links imply, as (source, target) ids.

    Targets are resolved on disk against the linking file, so a link only
    becomes an edge when it points at a file in the same batch: dangling links
    and self-links are dropped rather than creating empty nodes. Deduplicated
    and sorted, so the result is stable across runs.
    """
    # Filesystem path -> node id, so links can be resolved on disk then mapped.
    known = {
        os.path.normpath(os.path.abspath(c["path"])): concept_id(c["path"], bundle_root)
        for c in concepts
    }

    edges = set()
    for concept in concepts:
        source = concept_id(concept["path"], bundle_root)
        for link in concept.get("links") or []:
            resolved = _resolve_target(concept["path"], link.get("target", ""))
            target = known.get(os.path.normpath(os.path.abspath(resolved)))
            if target is not None and target != source:
                edges.add((source, target))
    return sorted(edges)


# Every node write MERGEs on :Concept(path) and every read matches on it, so
# it is the one property that has to be indexed. Without it Neo4j falls back to
# a full label scan per MERGE, making a load quadratic in the size of the
# bundle. A uniqueness constraint creates the backing index *and* enforces the
# invariant the whole design rests on: one node per concept id, which is what
# lets Qdrant payloads join onto the graph.
CONCEPT_PATH_CONSTRAINT = (
    "CREATE CONSTRAINT concept_path IF NOT EXISTS "
    "FOR (c:Concept) REQUIRE c.path IS UNIQUE"
)


def ensure_schema(session) -> None:
    """Create the :Concept(path) uniqueness constraint if it is missing.

    Idempotent (IF NOT EXISTS), so every load can call it unconditionally.
    """
    session.run(CONCEPT_PATH_CONSTRAINT)


def load_bundle_to_neo4j(concepts: list[dict], bundle_root: str) -> tuple[int, int]:
    """MERGE every concept as a :Concept node and link them with :LINKS_TO edges.

    Nodes are keyed on a bundle-relative id (see concept_id); the on-disk
    location is kept alongside it as a non-identifying source_path property.
    Ensures the :Concept(path) constraint first, so the MERGEs are index-backed
    rather than label scans.

    Returns (nodes_written, relationships_written).
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    nodes = 0
    relationships = 0

    try:
        with driver.session() as session:
            ensure_schema(session)

            for concept in concepts:
                props = _properties(concept.get("frontmatter"))
                props["source_path"] = os.path.abspath(concept["path"])
                session.run(
                    "MERGE (c:Concept {path: $path}) SET c += $props",
                    path=concept_id(concept["path"], bundle_root),
                    props=props,
                )
                nodes += 1

            for source, target in bundle_edges(concepts, bundle_root):
                session.run(
                    "MATCH (a:Concept {path: $source}) "
                    "MATCH (b:Concept {path: $target}) "
                    "MERGE (a)-[:LINKS_TO]->(b)",
                    source=source,
                    target=target,
                )
                relationships += 1
    finally:
        driver.close()

    return nodes, relationships


def prune_neo4j_bundle(concepts: list[dict], bundle_root: str) -> tuple[int, int]:
    """Delete this bundle's nodes and edges that its files no longer imply.

    The loaders only ever MERGE, so a concept deleted or renamed on disk leaves
    its node behind, and a link removed from a file leaves its edge behind;
    both then keep surfacing in retrieval. This reconciles the stored bundle
    with the parsed one. Scoped by id prefix, so other bundles in the same
    database are untouched.

    Stale edges go first: an edge touching a stale node is stale by definition
    (the surviving edge set only covers current concepts), so it is counted
    here rather than disappearing silently with the node.

    Returns (nodes_deleted, relationships_deleted).
    """
    if not concepts:
        raise ValueError(
            "refusing to prune from an empty parse - every node in the bundle "
            "would be deleted; check the bundle path"
        )

    prefix = bundle_prefix(bundle_root)
    keep_nodes = [concept_id(c["path"], bundle_root) for c in concepts]
    keep_edges = [list(edge) for edge in bundle_edges(concepts, bundle_root)]

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            edges = session.run(
                "MATCH (a:Concept)-[r:LINKS_TO]->(b:Concept) "
                "WHERE a.path STARTS WITH $prefix AND b.path STARTS WITH $prefix "
                "AND NOT [a.path, b.path] IN $keep "
                "DELETE r "
                "RETURN count(r) AS deleted",
                prefix=prefix,
                keep=keep_edges,
            ).single()["deleted"]

            nodes = session.run(
                "MATCH (c:Concept) "
                "WHERE c.path STARTS WITH $prefix AND NOT c.path IN $keep "
                "DETACH DELETE c "
                "RETURN count(c) AS deleted",
                prefix=prefix,
                keep=keep_nodes,
            ).single()["deleted"]
    finally:
        driver.close()

    return nodes, edges


def main() -> None:
    """Parse a bundle directory and push it into Neo4j."""
    from parser import parse_bundle

    ap = argparse.ArgumentParser(description="Load an OKF bundle into Neo4j.")
    ap.add_argument("bundle", help="path to the bundle directory")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="also delete this bundle's nodes and edges that are no longer on disk",
    )
    args = ap.parse_args()

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise SystemExit(
            "set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD (e.g. in a .env file)"
        )

    concepts = parse_bundle(args.bundle)
    try:
        nodes, relationships = load_bundle_to_neo4j(concepts, args.bundle)
    except ServiceUnavailable:
        raise SystemExit(
            f"could not reach Neo4j at {NEO4J_URI} - is the database running?"
        ) from None
    except AuthError:
        raise SystemExit(
            f"Neo4j rejected the credentials for user '{NEO4J_USER}' - "
            "check NEO4J_USER and NEO4J_PASSWORD"
        ) from None
    except ConfigurationError as exc:
        raise SystemExit(
            f"bad Neo4j configuration (NEO4J_URI={NEO4J_URI}): {exc}"
        ) from None
    print(f"{nodes} nodes merged, {relationships} relationships merged")

    if args.prune:
        try:
            gone_nodes, gone_edges = prune_neo4j_bundle(concepts, args.bundle)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(
            f"{gone_nodes} stale nodes deleted, "
            f"{gone_edges} stale relationships deleted"
        )


if __name__ == "__main__":
    main()

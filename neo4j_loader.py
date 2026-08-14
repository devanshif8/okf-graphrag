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


def load_bundle_to_neo4j(concepts: list[dict], bundle_root: str) -> tuple[int, int]:
    """MERGE every concept as a :Concept node and link them with :LINKS_TO edges.

    Nodes are keyed on a bundle-relative id (see concept_id); the on-disk
    location is kept alongside it as a non-identifying source_path property.

    Returns (nodes_written, relationships_written).
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    nodes = 0
    relationships = 0

    # Filesystem path -> node id, so links can be resolved on disk then mapped.
    known = {
        os.path.normpath(os.path.abspath(c["path"])): concept_id(c["path"], bundle_root)
        for c in concepts
    }

    try:
        with driver.session() as session:
            for concept in concepts:
                props = _properties(concept.get("frontmatter"))
                props["source_path"] = os.path.abspath(concept["path"])
                session.run(
                    "MERGE (c:Concept {path: $path}) SET c += $props",
                    path=concept_id(concept["path"], bundle_root),
                    props=props,
                )
                nodes += 1

            for concept in concepts:
                source = concept_id(concept["path"], bundle_root)
                targets = set()
                for link in concept.get("links") or []:
                    resolved = _resolve_target(concept["path"], link.get("target", ""))
                    target = known.get(os.path.normpath(os.path.abspath(resolved)))
                    if target is not None and target != source:
                        targets.add(target)

                for target in sorted(targets):
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


def main() -> None:
    """Parse a bundle directory and push it into Neo4j."""
    from parser import parse_bundle

    ap = argparse.ArgumentParser(description="Load an OKF bundle into Neo4j.")
    ap.add_argument("bundle", help="path to the bundle directory")
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


if __name__ == "__main__":
    main()

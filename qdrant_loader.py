"""Embed parsed OKF bundles and load them into Qdrant for semantic search."""

import argparse
import datetime
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from neo4j_loader import bundle_prefix, concept_id

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "okf_concepts"
MODEL_NAME = "all-MiniLM-L6-v2"

# Qdrant point ids must be an unsigned int or a UUID, but the concept id is a
# path like 'ga4/index.md'. Derive the point id from it so it stays stable
# across runs, and keep the readable id in the payload as 'path'.
ID_NAMESPACE = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def _point_id(concept_key: str) -> str:
    """Map a concept id onto the deterministic UUID Qdrant will store it under."""
    return str(uuid.uuid5(ID_NAMESPACE, concept_key))


def _payload_value(value):
    """Coerce a frontmatter value into something JSON-serializable for a payload."""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _payload_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_payload_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def embedding_text(concept: dict) -> str:
    """Text to embed: the frontmatter description, if any, plus the body."""
    description = (concept.get("frontmatter") or {}).get("description")
    body = (concept.get("body") or "").strip()
    if isinstance(description, str) and description.strip():
        return f"{description.strip()}\n\n{body}".strip()
    return body


def _embedding_dimension(model: SentenceTransformer) -> int:
    """The model's output width, under whichever name this version exposes it."""
    getter = getattr(model, "get_embedding_dimension", None)
    return getter() if getter else model.get_sentence_embedding_dimension()


def _ensure_collection(client: QdrantClient, size: int) -> None:
    """Create the collection at the model's output dimension, if it's missing."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=size, distance=Distance.COSINE),
        )


def load_bundle_to_qdrant(concepts: list[dict], bundle_root: str) -> int:
    """Embed every concept and upsert it into Qdrant, keyed to match Neo4j.

    Each point carries the concept's bundle-relative id (the same one
    neo4j_loader keys its :Concept nodes on), its on-disk source_path, and
    every frontmatter field, so a vector hit can be joined onto the graph.

    Returns the number of concepts upserted.
    """
    model = SentenceTransformer(MODEL_NAME)
    client = QdrantClient(url=QDRANT_URL)

    try:
        _ensure_collection(client, _embedding_dimension(model))

        texts = [embedding_text(c) for c in concepts]
        vectors = model.encode(texts, show_progress_bar=False)

        points = []
        for concept, vector in zip(concepts, vectors):
            key = concept_id(concept["path"], bundle_root)
            payload = {
                str(k): _payload_value(v)
                for k, v in (concept.get("frontmatter") or {}).items()
                if str(k) not in ("path", "source_path")
            }
            payload["path"] = key
            payload["source_path"] = os.path.abspath(concept["path"])
            points.append(
                PointStruct(id=_point_id(key), vector=vector.tolist(), payload=payload)
            )

        if points:
            client.upsert(collection_name=COLLECTION, points=points)
    finally:
        client.close()

    return len(points)


def _stale_point_ids(points, prefix: str, keep: set) -> list:
    """Of these points, the ids belonging to the bundle but not to `keep`.

    A point belongs to the bundle when its payload `path` — the concept id —
    starts with the bundle prefix; points from other bundles, and any without a
    readable path, are left alone.
    """
    stale = []
    for point in points:
        path = (point.payload or {}).get("path")
        if isinstance(path, str) and path.startswith(prefix) and path not in keep:
            stale.append(point.id)
    return stale


def prune_qdrant_bundle(concepts: list[dict], bundle_root: str) -> int:
    """Delete this bundle's points whose concept file is no longer on disk.

    The loader only upserts, so a concept deleted or renamed on disk leaves its
    vector behind and it keeps matching queries. Scoped by the id prefix in the
    payload, so other bundles in the same collection are untouched.

    Returns the number of points deleted.
    """
    if not concepts:
        raise ValueError(
            "refusing to prune from an empty parse - every point in the bundle "
            "would be deleted; check the bundle path"
        )

    prefix = bundle_prefix(bundle_root)
    keep = {concept_id(c["path"], bundle_root) for c in concepts}

    client = QdrantClient(url=QDRANT_URL)
    try:
        if not client.collection_exists(COLLECTION):
            return 0

        # Scrolled rather than filtered server-side: a prefix match on a payload
        # string needs a full-text index the collection doesn't declare, and the
        # ids alone are small enough to page through.
        stale, offset = [], None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            stale.extend(_stale_point_ids(points, prefix, keep))
            if offset is None:
                break

        if stale:
            client.delete(
                collection_name=COLLECTION,
                points_selector=PointIdsList(points=stale),
            )
    finally:
        client.close()

    return len(stale)


def main() -> None:
    """Parse a bundle directory, embed it, and push it into Qdrant."""
    from parser import parse_bundle

    ap = argparse.ArgumentParser(description="Load an OKF bundle into Qdrant.")
    ap.add_argument("bundle", help="path to the bundle directory")
    ap.add_argument(
        "--prune",
        action="store_true",
        help="also delete this bundle's points that are no longer on disk",
    )
    args = ap.parse_args()

    concepts = parse_bundle(args.bundle)
    try:
        count = load_bundle_to_qdrant(concepts, args.bundle)
    except (ResponseHandlingException, ConnectionError):
        raise SystemExit(
            f"could not reach Qdrant at {QDRANT_URL} - is it running?"
        ) from None
    print(f"{count} concepts embedded and upserted into '{COLLECTION}'")

    if args.prune:
        try:
            deleted = prune_qdrant_bundle(concepts, args.bundle)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(f"{deleted} stale points deleted from '{COLLECTION}'")


if __name__ == "__main__":
    main()

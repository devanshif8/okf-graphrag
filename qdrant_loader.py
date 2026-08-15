"""Embed parsed OKF bundles and load them into Qdrant for semantic search."""

import argparse
import datetime
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from neo4j_loader import concept_id

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


def main() -> None:
    """Parse a bundle directory, embed it, and push it into Qdrant."""
    from parser import parse_bundle

    ap = argparse.ArgumentParser(description="Load an OKF bundle into Qdrant.")
    ap.add_argument("bundle", help="path to the bundle directory")
    args = ap.parse_args()

    concepts = parse_bundle(args.bundle)
    try:
        count = load_bundle_to_qdrant(concepts, args.bundle)
    except (ResponseHandlingException, ConnectionError):
        raise SystemExit(
            f"could not reach Qdrant at {QDRANT_URL} - is it running?"
        ) from None
    print(f"{count} concepts embedded and upserted into '{COLLECTION}'")


if __name__ == "__main__":
    main()

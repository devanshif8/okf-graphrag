"""Embed parsed OKF bundles and load them into Qdrant for semantic search."""

import argparse
import datetime
import hashlib
import json
import os
import re
import threading
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

# Words per chunk. all-MiniLM-L6-v2 truncates its input at 256 word-pieces and
# English runs about 1.3 word-pieces per word, so ~195 words is where the model
# stops reading. 160 leaves headroom for text that tokenizes worse than average
# — code, names, numbers — because a chunk that overflows is not an error, it is
# silently truncated content, which is the exact failure chunking exists to fix.
CHUNK_WORDS = 160

# Words repeated from the end of the previous chunk into the start of the next.
# A hard boundary can land mid-argument and leave the sentences either side of
# it in two vectors that each hold half the meaning; the overlap means any span
# shorter than this appears whole in at least one chunk.
CHUNK_OVERLAP_WORDS = 32

# Paragraph, then sentence: the boundaries an author already put in the text.
# Splitting on them keeps a chunk about one thing far more often than slicing
# every N words regardless of where N lands.
PARAGRAPH_RE = re.compile(r"\n\s*\n")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Chunks embedded and upserted per round trip. The whole bundle used to be held
# in memory at once — every chunk's text, then every chunk's vector, then every
# PointStruct — which is fine for a sample bundle and fatal for a real corpus:
# a million chunks is well over a gigabyte of float32 before the payloads are
# counted. Working in batches makes peak memory a function of this number
# instead of the size of the corpus.
EMBED_BATCH_CHUNKS = 256

# Payload field holding the digest of everything that determines a concept's
# points. Re-loading a bundle re-embeds only the concepts whose digest changed.
DIGEST_FIELD = "content_digest"


# The embedding model, shared. Constructing a SentenceTransformer is not just
# slow (seconds of weight loading) — it also contacts huggingface.co to
# revalidate the cached files, so building one per call makes an otherwise
# offline operation depend on the network, and fail with it. This module owns
# MODEL_NAME, so it owns the instance; `retrieval` borrows it from here rather
# than keeping a second copy, which would double both costs.
_MODEL: SentenceTransformer | None = None
_MODEL_LOCK = threading.Lock()


def model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first use.

    Loading is guarded because a threaded server can reach this concurrently;
    the weights are immutable afterwards, so only construction races.
    """
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = SentenceTransformer(MODEL_NAME)
        return _MODEL


def _point_id(concept_key: str, chunk_index: int = 0) -> str:
    """Map a concept id and chunk number onto their deterministic Qdrant UUID.

    Chunk 0 hashes the bare concept id, so a concept short enough to be one
    chunk keeps the id it had before chunking existed. That makes the change a
    no-op for short documents — they upsert over themselves rather than
    orphaning a point under a new id — so no migration step is needed.
    """
    key = concept_key if chunk_index == 0 else f"{concept_key}#{chunk_index}"
    return str(uuid.uuid5(ID_NAMESPACE, key))


def _split_units(text: str, max_words: int, overlap: int) -> list[str]:
    """Break text into the largest pieces that each fit inside one chunk.

    Paragraphs first, falling back to sentences for a paragraph that is too
    long, and to a hard word slice for a sentence that is — a wall of text with
    no punctuation still has to be embedded, and a piece that cannot fit is
    worse than one split in an ugly place.

    Hard slices are cut to `max_words - overlap`, not `max_words`, so that a
    slice plus the overlap carried from the chunk before it still fits. Cutting
    them at the full width would force the packer to choose between exceeding
    the embedding window and dropping the overlap, and unpunctuated text is
    exactly where the overlap matters most — there is no sentence boundary to
    fall back on.
    """
    slab = max(1, max_words - overlap)

    units: list[str] = []
    for paragraph in PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph.split()) <= max_words:
            units.append(paragraph)
            continue

        for sentence in SENTENCE_RE.split(paragraph):
            words = sentence.split()
            if not words:
                continue
            if len(words) <= max_words:
                units.append(sentence.strip())
                continue
            for start in range(0, len(words), slab):
                units.append(" ".join(words[start : start + slab]))
    return units


def chunk_text(
    text: str,
    max_words: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping chunks that each fit the embedding window.

    Returns the chunks in document order. Empty text yields no chunks; callers
    that need every concept represented supply their own single empty chunk
    rather than having one invented here.
    """
    # An overlap at or above the chunk size would re-emit the whole previous
    # chunk each time and never advance through the document.
    overlap = max(0, min(overlap, max_words // 2))

    chunks: list[str] = []
    current: list[str] = []
    for unit in _split_units(text, max_words, overlap):
        words = unit.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            # Carry the overlap only when the next unit still fits behind it.
            # A natural boundary can produce a unit too wide to share a chunk
            # with the tail, and no chunk may exceed the embedding window —
            # that overflow is the silent truncation chunking exists to remove,
            # so the overlap is what gives way.
            tail = current[-overlap:] if overlap else []
            current = tail if len(tail) + len(words) <= max_words else []
        current.extend(words)

    if current:
        chunks.append(" ".join(current))
    return chunks


def concept_chunks(concept: dict) -> list[str]:
    """The chunks one concept is indexed as — always at least one.

    A concept with no text still gets an empty chunk so that every concept has
    a point to be joined onto from the graph. Dropping it instead would make a
    concept exist in Neo4j and not in Qdrant, breaking the invariant the two
    stores are keyed on.
    """
    return chunk_text(embedding_text(concept)) or [""]


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


def _concept_payload(concept: dict, key: str) -> dict:
    """The payload fields every chunk of one concept shares."""
    payload = {
        str(k): _payload_value(v)
        for k, v in (concept.get("frontmatter") or {}).items()
        if str(k) not in ("path", "source_path")
    }
    payload["path"] = key
    payload["source_path"] = os.path.abspath(concept["path"])
    return payload


def content_digest(chunks: list[str], payload: dict) -> str:
    """A stable digest of everything that decides a concept's stored points.

    Both halves matter. The chunk text decides the vectors, so a body edit has
    to invalidate them — and the payload decides what search returns alongside
    the vector, so a frontmatter edit that leaves the body untouched has to
    invalidate them too. Digesting only the text would let a corrected title or
    description sit stale in the index forever, because the load that should
    have rewritten it would skip the concept as unchanged.

    Sorted keys, so two runs over the same file agree regardless of dict order.
    """
    material = json.dumps(
        {"chunks": chunks, "payload": payload}, sort_keys=True, default=str
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _stored_digests(client: QdrantClient, prefix: str) -> dict:
    """Map each already-loaded concept in this bundle to its stored digest.

    Read from chunk 0, which every concept has. Scrolled rather than filtered
    server-side for the same reason the prune scrolls: a prefix match on a
    payload string would need a full-text index the collection does not declare.
    One pass over the collection is far cheaper than re-embedding it.
    """
    digests: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            path = payload.get("path")
            digest = payload.get(DIGEST_FIELD)
            if (
                isinstance(path, str)
                and path.startswith(prefix)
                and isinstance(digest, str)
                and payload.get("chunk_index", 0) == 0
            ):
                digests[path] = digest
        if offset is None:
            break
    return digests


def _flush(client: QdrantClient, encoder, texts: list[str], points: list) -> int:
    """Embed one batch of chunk texts and upsert their points. Returns the count."""
    if not texts:
        return 0

    vectors = encoder.encode(texts, show_progress_bar=False)
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(id=point_id, vector=vector.tolist(), payload=payload)
            for (point_id, payload), vector in zip(points, vectors)
        ],
    )
    return len(points)


def load_bundle_to_qdrant(
    concepts: list[dict],
    bundle_root: str,
    force: bool = False,
    batch_size: int = EMBED_BATCH_CHUNKS,
) -> dict:
    """Embed every concept and upsert it into Qdrant, keyed to match Neo4j.

    Each concept is split into chunks that fit the embedding window and stored
    as one point per chunk, so a long document is searchable throughout instead
    of by its opening paragraph. Every point carries the concept's
    bundle-relative id (the same one neo4j_loader keys its :Concept nodes on),
    its on-disk source_path, every frontmatter field, and its position in the
    document as `chunk_index` of `chunk_count` — so a vector hit still joins
    onto exactly one graph node, and retrieval can fold a concept's chunks back
    together.

    A concept whose digest matches what is already stored is skipped without
    being embedded, so re-loading a bundle after editing one file costs one
    document rather than all of them. `force` re-embeds everything, which is
    what to reach for after changing the model, the chunk width or the
    embedding text — none of which the digest can see. Embedding and upserting
    happen `batch_size` chunks at a time, so peak memory does not grow with the
    corpus.

    Returns {"concepts": n, "embedded": n, "skipped": n, "points": n}, where
    `points` counts only the points actually written.
    """
    encoder = model()
    client = QdrantClient(url=QDRANT_URL)

    try:
        _ensure_collection(client, _embedding_dimension(encoder))

        stored = {} if force else _stored_digests(client, bundle_prefix(bundle_root))

        texts: list[str] = []
        pending: list[tuple[str, dict]] = []
        embedded = skipped = written = 0

        for concept in concepts:
            key = concept_id(concept["path"], bundle_root)
            payload = _concept_payload(concept, key)
            chunks = concept_chunks(concept)
            digest = content_digest(chunks, payload)

            if stored.get(key) == digest:
                skipped += 1
                continue

            embedded += 1
            for position, chunk in enumerate(chunks):
                chunk_payload = dict(payload)
                chunk_payload["chunk_index"] = position
                chunk_payload["chunk_count"] = len(chunks)
                chunk_payload[DIGEST_FIELD] = digest
                texts.append(chunk)
                pending.append((_point_id(key, position), chunk_payload))

            # Flushed between concepts, never inside one, so a concept's chunks
            # are always written together: a crash mid-load can leave a concept
            # absent, but never half-indexed under a digest claiming it is whole.
            if len(texts) >= batch_size:
                written += _flush(client, encoder, texts, pending)
                texts, pending = [], []

        written += _flush(client, encoder, texts, pending)
    finally:
        client.close()

    return {
        "concepts": len(concepts),
        "embedded": embedded,
        "skipped": skipped,
        "points": written,
    }


def _stale_point_ids(points, prefix: str, keep: dict) -> list:
    """Of these points, the ids belonging to the bundle but no longer implied.

    `keep` maps each surviving concept id to the number of chunks it now has. A
    point belongs to the bundle when its payload `path` — the concept id —
    starts with the bundle prefix; points from other bundles, and any without a
    readable path, are left alone.

    Two things make a point stale: its concept is gone from disk, or its
    concept was edited shorter and its chunk no longer exists. The second case
    is why `keep` carries counts rather than being a set — an edit that drops a
    document from five chunks to three leaves chunks 3 and 4 behind, matching
    queries with text the document no longer contains.

    A point with no `chunk_index` predates chunking and is therefore chunk 0,
    which every surviving concept still has, so it is kept and overwritten in
    place by the next load.
    """
    stale = []
    for point in points:
        payload = point.payload or {}
        path = payload.get("path")
        if not isinstance(path, str) or not path.startswith(prefix):
            continue

        count = keep.get(path)
        if count is None:
            stale.append(point.id)
            continue

        index = payload.get("chunk_index")
        if isinstance(index, int) and index >= count:
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
    # Chunking here costs only text splitting, not embedding, so the prune can
    # know each surviving concept's current chunk count without a model.
    keep = {
        concept_id(c["path"], bundle_root): len(concept_chunks(c)) for c in concepts
    }

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
    ap.add_argument(
        "--force",
        action="store_true",
        help=(
            "re-embed every concept, including unchanged ones - needed after "
            "changing the model, the chunk width or the embedding text"
        ),
    )
    args = ap.parse_args()

    concepts = parse_bundle(args.bundle)
    try:
        result = load_bundle_to_qdrant(concepts, args.bundle, force=args.force)
    except (ResponseHandlingException, ConnectionError):
        raise SystemExit(
            f"could not reach Qdrant at {QDRANT_URL} - is it running?"
        ) from None
    print(
        f"{result['embedded']} of {result['concepts']} concepts embedded as "
        f"{result['points']} chunks and upserted into '{COLLECTION}'"
        + (f", {result['skipped']} unchanged" if result["skipped"] else "")
    )

    if args.prune:
        try:
            deleted = prune_qdrant_bundle(concepts, args.bundle)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(f"{deleted} stale points deleted from '{COLLECTION}'")


if __name__ == "__main__":
    main()

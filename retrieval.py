"""Hybrid retrieval over an OKF bundle: Qdrant vector search fused with Neo4j
graph traversal via Reciprocal Rank Fusion."""

import argparse
import atexit
import os
import threading

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sentence_transformers import SentenceTransformer

from neo4j_loader import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
import qdrant_loader
from qdrant_loader import COLLECTION, QDRANT_URL

# RRF's rank-smoothing constant. 60 is the value from the original paper
# (Cormack et al. 2009) and the usual default; raising it flattens the
# contribution of the top ranks relative to the tail.
RRF_K = 60

# How many of the vector hits seed the graph walk. Expanding from every hit
# would pull in most of a small bundle and flatten the graph ranking.
GRAPH_SEEDS = 5

# Index files are hubs: they carry the most :LINKS_TO edges, so they surface
# near the top of almost any expansion regardless of the query. Weighting the
# graph list below the vector list keeps that structural signal as a tie-break
# on semantic relevance instead of letting it outvote a strong vector hit.
VECTOR_WEIGHT = 1.0
GRAPH_WEIGHT = 0.5

# Concept ids are bundle-relative and always forward-slashed, so an index file
# is exactly one whose id ends in this. The parser's own is_index flag would be
# the more direct test, but it lives outside the frontmatter and so never
# reaches Neo4j as a node property.
INDEX_SUFFIX = "/index.md"

# Exponent on a node's degree when damping its graph score. A well-connected
# concept is reachable from more seeds whatever the query, so its rank partly
# reflects the shape of the bundle rather than relevance. 0.5 (divide by the
# square root of the degree) is the usual middle ground: it offsets the bias
# without burying genuinely central concepts the way dividing by degree
# outright would. 0 disables damping.
DEGREE_DAMPING = 0.5

# Qdrant stores one point per *chunk*, but everything downstream — the graph
# join, the fusion, the results — is per *concept*. So a search has to retrieve
# more points than it needs concepts, because several of them can belong to the
# same document. This multiplier is that headroom: with it, a query whose top
# hits are four chunks of one long document still comes back with other
# documents behind it instead of a single document repeated.
CHUNK_OVERSAMPLE = 4

# Vector candidates to retrieve regardless of how many results are asked for.
# Fusion needs a deeper pool than it returns: with top_k alone, asking for 2
# results would also shrink the candidate pool to 2, so a concept could vanish
# from the output for never having been retrieved rather than for ranking badly.
MIN_CANDIDATES = 10


def _seconds(name: str, default: float) -> float:
    """Read a timeout from the environment, falling back on anything unusable."""
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Timeouts. Without them a store that accepts a connection and then stops
# answering blocks the calling thread forever — and since the clients below are
# now long-lived and shared, that is a worker lost for the life of the process
# rather than for one request. Every one of these is a *bound*, not a target:
# the p90 query is ~0.3s, so these only fire when something is actually wrong.
#
# Neo4j splits its budget three ways: reaching the server, waiting for a free
# pooled connection, and retrying a transaction that failed transiently.
NEO4J_CONNECTION_TIMEOUT = _seconds("NEO4J_CONNECTION_TIMEOUT", 5.0)
NEO4J_ACQUISITION_TIMEOUT = _seconds("NEO4J_ACQUISITION_TIMEOUT", 10.0)
NEO4J_RETRY_TIME = _seconds("NEO4J_RETRY_TIME", 15.0)

# Qdrant takes a single per-request timeout covering connect and read.
QDRANT_TIMEOUT = _seconds("QDRANT_TIMEOUT", 10.0)


# Every client here is expensive to build and cheap to keep: the Neo4j driver
# is a connection *pool* meant to outlive individual queries, constructing a
# QdrantClient costs ~700ms (it probes the server and builds its HTTP stack),
# and loading all-MiniLM-L6-v2 takes seconds. Building any of them per query —
# as this module used to — put ~1s of pure setup in front of ~30ms of work.
#
# All three are safe to share: the driver and the Qdrant client are documented
# as thread-safe once constructed, and the model's weights are immutable after
# load. Only the construction races, so that is what the lock guards. Built
# lazily so importing this module stays cheap for callers that never search.
_DRIVER = None
_QDRANT: QdrantClient | None = None
_LOCK = threading.Lock()


def _model() -> SentenceTransformer:
    """Return the shared query embedding model, loading it on first use.

    Delegates to qdrant_loader, which owns MODEL_NAME: the query has to be
    embedded by the same model as the stored vectors or the two are not
    comparable, and sharing the instance makes that structural rather than a
    convention two modules have to keep agreeing on.
    """
    return qdrant_loader.model()


def _driver():
    """Return the shared Neo4j driver, opening it on first use.

    Configured from the same .env the loader reads. The driver connects lazily,
    so building it does no I/O; what it buys is a reused connection pool.
    """
    global _DRIVER
    with _LOCK:
        if _DRIVER is None:
            _DRIVER = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                connection_timeout=NEO4J_CONNECTION_TIMEOUT,
                connection_acquisition_timeout=NEO4J_ACQUISITION_TIMEOUT,
                max_transaction_retry_time=NEO4J_RETRY_TIME,
            )
        return _DRIVER


def _qdrant() -> QdrantClient:
    """Return the shared Qdrant client, connecting on first use."""
    global _QDRANT
    with _LOCK:
        if _QDRANT is None:
            _QDRANT = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT)
        return _QDRANT


def close_clients() -> None:
    """Close the shared stores and drop them, so the next call rebuilds.

    A long-lived server calls this on shutdown and a CLI run on exit; tests use
    it to reset between cases. Safe to call when nothing was ever opened.
    """
    global _DRIVER, _QDRANT
    with _LOCK:
        for client in (_DRIVER, _QDRANT):
            if client is not None:
                try:
                    client.close()
                except Exception:  # noqa: BLE001 - shutdown must not raise
                    pass
        _DRIVER = _QDRANT = None


atexit.register(close_clients)


def fold_chunks(hits: list[dict], top_k: int) -> list[dict]:
    """Collapse chunk hits into one entry per concept, best chunk winning.

    A concept scores as its single best-matching chunk rather than, say, the
    mean of its chunks. Max is the right summary because chunking exists to let
    one passage of a long document answer a narrow question: averaging would
    punish exactly that document for the many chunks that are about something
    else, which is the behaviour chunking was introduced to remove.

    `chunk_index` records which chunk matched, so a caller can show the passage
    that was actually responsible for the hit.
    """
    best: dict[str, dict] = {}
    for hit in hits:
        identifier = hit["id"]
        if identifier not in best or hit["score"] > best[identifier]["score"]:
            best[identifier] = hit

    ranked = sorted(best.values(), key=lambda h: (-h["score"], h["id"]))
    return ranked[:top_k]


def vector_search(query: str, top_k: int = 10) -> list[dict]:
    """Rank concepts by embedding similarity to the query, best first.

    Embeds the query with the same model qdrant_loader used to embed the
    bundle, and searches the collection it wrote. The collection holds one
    point per chunk, so the raw hits are folded back to one entry per concept
    before returning — see `fold_chunks`. Returns dicts of
    {"id": concept_id, "score": similarity_score, "chunk_index": int}.
    """
    client = _qdrant()

    if not client.collection_exists(COLLECTION):
        raise LookupError(
            f"collection '{COLLECTION}' does not exist at {QDRANT_URL} - "
            "load a bundle with qdrant_loader.py first"
        )
    vector = _model().encode(query).tolist()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k * CHUNK_OVERSAMPLE,
        with_payload=True,
    ).points

    # The payload 'path' is the bundle-relative id qdrant_loader keyed the point
    # on (via neo4j_loader.concept_id), so it doubles as the graph node id.
    chunks = [
        {
            "id": hit.payload["path"],
            "score": hit.score,
            "chunk_index": (hit.payload or {}).get("chunk_index", 0),
        }
        for hit in hits
        if (hit.payload or {}).get("path")
    ]
    return fold_chunks(chunks, top_k)


def graph_expand(
    concept_ids: list[str],
    hops: int = 1,
    exclude_index: bool = True,
    degree_damping: float = DEGREE_DAMPING,
) -> list[dict]:
    """Rank the graph neighbourhood of the given concepts, closest first.

    Walks :LINKS_TO edges in either direction — a concept that links *to* a
    strong hit is as relevant as one linked *from* it — out to `hops` steps.
    Returns dicts of {"id", "distance", "degree", "score"}, best first.

    Each neighbour scores `seeds_hit / (distance * degree ** damping)`: reached
    by more of the inputs is better, further away is worse, and dividing by
    degree offsets the fact that a well-connected concept is reachable from
    more seeds for reasons that have nothing to do with the query. With
    `degree_damping=0` the divisor vanishes and ranking is by proximity and
    seed count alone.

    With `exclude_index` (the default), index files are dropped from the
    *results*. The walk still routes *through* them, so two concepts that are
    only connected via an index remain neighbours — they just stop crowding
    out the concepts they link to. See the README on hub damping.
    """
    if hops < 1:
        raise ValueError("hops must be at least 1")
    if not concept_ids:
        return []

    # A variable-length pattern can't take its bound as a query parameter, so
    # the (validated) int is formatted in. Everything else stays parameterized.
    index_filter = (
        "AND NOT n.path ENDS WITH $index_suffix" if exclude_index else ""
    )
    query = f"""
    UNWIND range(0, size($seeds) - 1) AS i
    WITH $seeds[i] AS seed, i AS seed_rank
    MATCH p = (s:Concept {{path: seed}})-[:LINKS_TO*1..{hops}]-(n:Concept)
    WHERE n.path <> seed
    {index_filter}
    WITH n AS node, seed_rank, min(length(p)) AS distance
    WITH node,
         min(distance) AS distance,
         count(DISTINCT seed_rank) AS seeds_hit,
         min(seed_rank) AS best_seed
    WITH node.path AS id,
         distance,
         seeds_hit,
         best_seed,
         COUNT {{ (node)-[:LINKS_TO]-() }} AS degree
    RETURN id,
           distance,
           degree,
           toFloat(seeds_hit) / (distance * degree ^ $damping) AS score
    ORDER BY score DESC, distance ASC, best_seed ASC, id ASC
    """

    with _driver().session() as session:
        result = session.run(
            query,
            seeds=list(concept_ids),
            index_suffix=INDEX_SUFFIX,
            damping=float(degree_damping),
        )
        return [dict(record) for record in result]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = RRF_K,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Fuse named ranked lists of concept ids into one, best first.

    Each id scores sum(weight / (k + rank)) over the lists it appears in, with
    1-based ranks. An id missing from a list simply contributes nothing for it,
    so placing in both lists beats ranking highly in only one. `weights` scales
    each named list's contribution, defaulting to 1.0 for any list not named in
    it — with every weight equal this is plain RRF. Returns dicts of
    id/score/ranks, ties broken on the id so the order is stable.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for name, ids in rankings.items():
        weight = weights.get(name, 1.0)
        for position, concept in enumerate(ids, start=1):
            scores[concept] = scores.get(concept, 0.0) + weight / (k + position)
            ranks.setdefault(concept, {})[name] = position

    fused = [
        {"id": concept, "score": score, "ranks": ranks[concept]}
        for concept, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r["score"], r["id"]))
    return fused


def _annotate(results: list[dict]) -> list[dict]:
    """Attach each result's title, description and source_path from its node."""
    if not results:
        return results

    with _driver().session() as session:
        records = session.run(
            "MATCH (c:Concept) WHERE c.path IN $ids "
            "RETURN c.path AS id, c.title AS title, "
            "c.description AS description, c.source_path AS source_path",
            ids=[r["id"] for r in results],
        )
        properties = {r["id"]: dict(r) for r in records}

    for result in results:
        node = properties.get(result["id"]) or {}
        result["title"] = node.get("title")
        result["description"] = node.get("description")
        result["source_path"] = node.get("source_path")
    return results


def hybrid_search(
    query: str,
    top_k: int = 10,
    vector_weight: float = VECTOR_WEIGHT,
    graph_weight: float = GRAPH_WEIGHT,
    exclude_index: bool = True,
    degree_damping: float = DEGREE_DAMPING,
) -> list[dict]:
    """Search a loaded bundle with vector similarity and graph traversal fused.

    Runs a vector search for an initial ranked list, expands the graph
    neighbourhood of the top few of those hits for a second ranked list, and
    fuses the two with weighted RRF. The lists are correlated by construction,
    so fusion rewards concepts that are both semantically close to the query
    and structurally close to what matched. Index files are dropped from the
    graph list by default, and that list is weighted below the vector list, so
    hub structure can't outvote a strong vector hit; pass equal weights and
    exclude_index=False for plain RRF over the raw neighbourhood. Returns up to
    `top_k` results, best first, each annotated with title and description,
    and with the `chunk_index` that matched when the vector list found it.
    """
    hits = vector_search(query, top_k=max(top_k, MIN_CANDIDATES))
    if not hits:
        return []

    vector_ids = [hit["id"] for hit in hits]
    neighbours = graph_expand(
        vector_ids[:GRAPH_SEEDS],
        exclude_index=exclude_index,
        degree_damping=degree_damping,
    )
    graph_ids = [n["id"] for n in neighbours]

    fused = reciprocal_rank_fusion(
        {"vector": vector_ids, "graph": graph_ids},
        weights={"vector": vector_weight, "graph": graph_weight},
    )[:top_k]

    # Keep the chunk that put each vector hit in the list, so a caller can show
    # the passage responsible. Graph-only results have no such chunk: the walk
    # reached the whole concept, not a passage of it.
    matched = {hit["id"]: hit["chunk_index"] for hit in hits}
    for result in fused:
        if result["id"] in matched:
            result["chunk_index"] = matched[result["id"]]
    return _annotate(fused)


def main() -> None:
    """Run one hybrid query against the loaded bundle and print the ranking."""
    ap = argparse.ArgumentParser(
        description="Hybrid (vector + graph) search over a loaded OKF bundle."
    )
    ap.add_argument("query", help="the search query")
    ap.add_argument(
        "--top-k", type=int, default=10, help="results to return (default: 10)"
    )
    ap.add_argument(
        "--vector-weight",
        type=float,
        default=VECTOR_WEIGHT,
        help=f"weight of the vector list in the fusion (default: {VECTOR_WEIGHT})",
    )
    ap.add_argument(
        "--graph-weight",
        type=float,
        default=GRAPH_WEIGHT,
        help=f"weight of the graph list in the fusion (default: {GRAPH_WEIGHT})",
    )
    ap.add_argument(
        "--include-index",
        action="store_true",
        help="keep index.md hub nodes in the graph list (they are dropped by default)",
    )
    ap.add_argument(
        "--degree-damping",
        type=float,
        default=DEGREE_DAMPING,
        help=(
            "exponent damping well-connected nodes in the graph list "
            f"(default: {DEGREE_DAMPING}; 0 disables)"
        ),
    )
    args = ap.parse_args()

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise SystemExit(
            "set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD (e.g. in a .env file)"
        )

    try:
        results = hybrid_search(
            args.query,
            top_k=args.top_k,
            vector_weight=args.vector_weight,
            graph_weight=args.graph_weight,
            exclude_index=not args.include_index,
            degree_damping=args.degree_damping,
        )
    except LookupError as exc:
        raise SystemExit(str(exc)) from None
    except (ResponseHandlingException, ConnectionError):
        raise SystemExit(
            f"could not reach Qdrant at {QDRANT_URL} - is it running?"
        ) from None
    except UnexpectedResponse as exc:
        raise SystemExit(f"Qdrant rejected the query: {exc}") from None
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

    if not results:
        print("no results")
        return

    print(f"{len(results)} results for {args.query!r}\n")
    for position, result in enumerate(results, start=1):
        found_in = ", ".join(
            f"{name} #{rank}" for name, rank in sorted(result["ranks"].items())
        )
        heading = result.get("title") or result["id"]
        print(f"{position}. {heading}")
        if result.get("title"):
            print(f"   {result['id']}")
        print(f"   rrf {result['score']:.4f}  ({found_in})")
        if result.get("description"):
            print(f"   {result['description']}")
        print()


if __name__ == "__main__":
    main()

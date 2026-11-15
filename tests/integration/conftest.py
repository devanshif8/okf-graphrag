"""Fixtures for the integration suite: real Neo4j, real Qdrant, real embeddings.

Everything here is skipped unless both stores are actually reachable, so the
default `pytest` run stays the pure-logic suite that needs no services. Set
OKF_INTEGRATION=1 to turn a skip into a failure instead — CI wants to know that
its services came up, rather than passing green because nothing ran.

Each test gets its own bundle in a temp directory whose name is unique, which
makes its concept ids unique too: ids are '<bundle dir name>/<relative path>',
and every write and prune in the project is scoped by that prefix. So these
tests cannot see, corrupt, or delete a real bundle sharing the same database —
which matters, because a developer's local stores usually hold one.
"""

import os
import uuid

import pytest

import neo4j_loader
import qdrant_loader
import retrieval

REQUIRED = os.getenv("OKF_INTEGRATION") == "1"


def _unreachable() -> str | None:
    """Why the stores can't be used, or None if both answer."""
    if not all([neo4j_loader.NEO4J_URI, neo4j_loader.NEO4J_USER, neo4j_loader.NEO4J_PASSWORD]):
        return "NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD are not set"
    try:
        retrieval._driver().verify_connectivity()
    except Exception as exc:  # noqa: BLE001 - any failure is a reason to skip
        return f"Neo4j at {neo4j_loader.NEO4J_URI} is not reachable: {exc}"
    try:
        retrieval._qdrant().get_collections()
    except Exception as exc:  # noqa: BLE001
        return f"Qdrant at {qdrant_loader.QDRANT_URL} is not reachable: {exc}"
    return None


# The suite's own Qdrant collection. Unique ids keep tests from *touching* a
# real bundle, but not from *competing* with it: vector search ranks over the
# whole collection, so a developer's library of thousands of chunks crowds the
# few test concepts out of the top-k and ranking assertions fail for reasons
# that have nothing to do with the code. A separate collection removes the
# competition. Neo4j needs no equivalent: the graph walk starts from the vector
# hits, so once those are the test bundle's own it never leaves that bundle.
TEST_COLLECTION = "okf_concepts_itest"


def _drop_test_collection() -> None:
    client = retrieval._qdrant()
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


@pytest.fixture(scope="session")
def stores():
    """Skip the whole suite unless both stores answer — or fail, if required.

    Also points every Qdrant read and write at TEST_COLLECTION for the session,
    starting it empty and dropping it afterwards.
    """
    reason = _unreachable()
    if reason:
        if REQUIRED:
            pytest.fail(f"OKF_INTEGRATION=1 but {reason}")
        pytest.skip(reason, allow_module_level=True)

    # retrieval imported the name by value, so both modules need patching.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qdrant_loader, "COLLECTION", TEST_COLLECTION)
        patch.setattr(retrieval, "COLLECTION", TEST_COLLECTION)
        _drop_test_collection()  # a crashed earlier run may have left one
        try:
            yield
        finally:
            _drop_test_collection()
    retrieval.close_clients()


def write_bundle(root, files: dict[str, str]) -> str:
    """Write {relative path: contents} under root and return root as a string."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return str(root)


def concept(title: str, description: str, body: str, links: list[str] = ()) -> str:
    """An OKF concept file: frontmatter, body, and markdown links to `links`."""
    related = "".join(f"\n- [{t}]({t})" for t in links)
    return (
        f"---\ntitle: {title}\ndescription: {description}\n---\n\n{body}\n"
        + (f"\n## Related{related}\n" if links else "")
    )


@pytest.fixture
def bundle(stores, tmp_path):
    """A small, uniquely-named bundle on disk, cleaned out of both stores after.

    Four concepts and an index, wired so the graph walk has something to walk:
    `alpha` and `beta` both link to `shared`, and the index links to everything.
    """
    root = tmp_path / f"itest{uuid.uuid4().hex[:10]}"
    write_bundle(
        root,
        {
            "index.md": "# Test bundle\n\n- [alpha](alpha.md)\n- [beta](beta.md)\n"
            "- [shared](topics/shared.md)\n- [orphan](orphan.md)\n",
            "alpha.md": concept(
                "Alpha Concept",
                "The first test concept, about migrating birds.",
                "Swallows and swifts migrate thousands of miles each year.",
                ["topics/shared.md"],
            ),
            "beta.md": concept(
                "Beta Concept",
                "The second test concept, also about migrating birds.",
                "Arctic terns make the longest migration of any animal.",
                ["topics/shared.md"],
            ),
            "topics/shared.md": concept(
                "Shared Topic",
                "A topic both alpha and beta link to.",
                "Migration routes are shaped by wind and food availability.",
            ),
            "orphan.md": concept(
                "Orphan Concept",
                "Linked only from the index.",
                "Deep sea hydrothermal vents host chemosynthetic life.",
            ),
        },
    )

    yield str(root)

    _purge(str(root))


def _purge(bundle_root: str) -> None:
    """Remove every trace of a bundle from both stores.

    Scoped by the id prefix, so it can only ever touch what this test wrote.
    """
    prefix = neo4j_loader.bundle_prefix(bundle_root)

    with retrieval._driver().session() as session:
        session.run(
            "MATCH (c:Concept) WHERE c.path STARTS WITH $prefix DETACH DELETE c",
            prefix=prefix,
        )

    client = retrieval._qdrant()
    if client.collection_exists(qdrant_loader.COLLECTION):
        stale, offset = [], None
        while True:
            points, offset = client.scroll(
                collection_name=qdrant_loader.COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            stale += [
                p.id
                for p in points
                if str((p.payload or {}).get("path", "")).startswith(prefix)
            ]
            if offset is None:
                break
        if stale:
            client.delete(
                collection_name=qdrant_loader.COLLECTION,
                points_selector=qdrant_loader.PointIdsList(points=stale),
            )

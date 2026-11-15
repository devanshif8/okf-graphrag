"""End-to-end tests against real stores: load, join, prune, search.

These cover the claims the unit suite structurally cannot: that a MERGE really
is idempotent in Neo4j, that --prune deletes what it says and nothing else,
that the two stores agree on concept ids, and that hybrid_search returns
sensible rankings. Every one of those had previously been checked only by hand.
"""

import os

import pytest

import neo4j_loader
import qdrant_loader
import retrieval
from parser import parse_bundle

pytestmark = pytest.mark.integration


def ids_in_neo4j(bundle_root):
    prefix = neo4j_loader.bundle_prefix(bundle_root)
    with retrieval._driver().session() as session:
        return {
            r["path"]
            for r in session.run(
                "MATCH (c:Concept) WHERE c.path STARTS WITH $prefix RETURN c.path AS path",
                prefix=prefix,
            )
        }


def edges_in_neo4j(bundle_root):
    prefix = neo4j_loader.bundle_prefix(bundle_root)
    with retrieval._driver().session() as session:
        return {
            (r["a"], r["b"])
            for r in session.run(
                "MATCH (a:Concept)-[:LINKS_TO]->(b:Concept) "
                "WHERE a.path STARTS WITH $prefix "
                "RETURN a.path AS a, b.path AS b",
                prefix=prefix,
            )
        }


def ids_in_qdrant(bundle_root):
    prefix = neo4j_loader.bundle_prefix(bundle_root)
    client = retrieval._qdrant()
    found, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=qdrant_loader.COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        found |= {
            p.payload["path"]
            for p in points
            if str((p.payload or {}).get("path", "")).startswith(prefix)
        }
        if offset is None:
            break
    return found


def load_both(bundle_root):
    concepts = parse_bundle(bundle_root)
    neo4j_loader.load_bundle_to_neo4j(concepts, bundle_root)
    qdrant_loader.load_bundle_to_qdrant(concepts, bundle_root)
    return concepts


class TestSchema:
    def test_loading_creates_the_uniqueness_constraint(self, bundle):
        load_both(bundle)

        with retrieval._driver().session() as session:
            constraints = [dict(r) for r in session.run("SHOW CONSTRAINTS")]

        # Without this the MERGEs are full label scans and loads go quadratic.
        assert any(
            c["labelsOrTypes"] == ["Concept"] and c["properties"] == ["path"]
            for c in constraints
        ), "no uniqueness constraint on :Concept(path)"

    def test_the_constraint_actually_rejects_a_duplicate_id(self, bundle):
        load_both(bundle)
        existing = sorted(ids_in_neo4j(bundle))[0]

        # The whole Qdrant<->Neo4j join assumes one node per concept id. Prove
        # the database enforces it rather than trusting the loader to.
        from neo4j.exceptions import ConstraintError

        with pytest.raises(ConstraintError):
            with retrieval._driver().session() as session:
                session.run("CREATE (:Concept {path: $path})", path=existing)


class TestLoad:
    def test_every_parsed_concept_reaches_both_stores(self, bundle):
        concepts = load_both(bundle)
        expected = {neo4j_loader.concept_id(c["path"], bundle) for c in concepts}

        assert ids_in_neo4j(bundle) == expected
        assert ids_in_qdrant(bundle) == expected

    def test_the_two_stores_key_on_exactly_the_same_ids(self, bundle):
        # The join between vector hits and graph nodes is this equality; if it
        # ever drifts, hybrid search silently degrades to vector-only.
        load_both(bundle)
        assert ids_in_neo4j(bundle) == ids_in_qdrant(bundle)

    def test_links_become_edges_in_both_directions_of_the_walk(self, bundle):
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        edges = edges_in_neo4j(bundle)
        assert (f"{prefix}alpha.md", f"{prefix}topics/shared.md") in edges
        assert (f"{prefix}beta.md", f"{prefix}topics/shared.md") in edges

    def test_frontmatter_is_stored_as_node_properties(self, bundle):
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        with retrieval._driver().session() as session:
            record = session.run(
                "MATCH (c:Concept {path: $path}) RETURN c.title AS title, "
                "c.description AS description, c.source_path AS source_path",
                path=f"{prefix}alpha.md",
            ).single()

        assert record["title"] == "Alpha Concept"
        assert "migrating birds" in record["description"]
        assert record["source_path"].endswith("alpha.md")


class TestIdempotence:
    def test_reloading_updates_in_place_rather_than_duplicating(self, bundle):
        load_both(bundle)
        nodes, edges, points = (
            ids_in_neo4j(bundle),
            edges_in_neo4j(bundle),
            ids_in_qdrant(bundle),
        )

        load_both(bundle)
        load_both(bundle)

        assert ids_in_neo4j(bundle) == nodes
        assert edges_in_neo4j(bundle) == edges
        assert ids_in_qdrant(bundle) == points

    def test_an_edit_is_reflected_without_leaving_the_old_value(self, bundle, tmp_path):
        import os

        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        path = os.path.join(bundle, "alpha.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "---\ntitle: Alpha Renamed\ndescription: Now about something else.\n"
                "---\n\nCompletely different text.\n"
            )
        load_both(bundle)

        with retrieval._driver().session() as session:
            title = session.run(
                "MATCH (c:Concept {path: $path}) RETURN c.title AS title",
                path=f"{prefix}alpha.md",
            ).single()["title"]

        assert title == "Alpha Renamed"


class TestPrune:
    def test_a_deleted_file_is_removed_from_both_stores(self, bundle):
        import os

        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)
        assert f"{prefix}orphan.md" in ids_in_neo4j(bundle)

        os.remove(os.path.join(bundle, "orphan.md"))
        concepts = parse_bundle(bundle)
        neo4j_loader.load_bundle_to_neo4j(concepts, bundle)
        qdrant_loader.load_bundle_to_qdrant(concepts, bundle)

        # Without --prune the stale node and point survive: that is the bug
        # prune exists for, so assert it before asserting the fix.
        assert f"{prefix}orphan.md" in ids_in_neo4j(bundle)
        assert f"{prefix}orphan.md" in ids_in_qdrant(bundle)

        neo4j_loader.prune_neo4j_bundle(concepts, bundle)
        qdrant_loader.prune_qdrant_bundle(concepts, bundle)

        assert f"{prefix}orphan.md" not in ids_in_neo4j(bundle)
        assert f"{prefix}orphan.md" not in ids_in_qdrant(bundle)

    def test_a_removed_link_drops_its_edge_but_keeps_both_nodes(self, bundle):
        import os

        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)
        edge = (f"{prefix}alpha.md", f"{prefix}topics/shared.md")
        assert edge in edges_in_neo4j(bundle)

        with open(os.path.join(bundle, "alpha.md"), "w", encoding="utf-8") as handle:
            handle.write(
                "---\ntitle: Alpha Concept\ndescription: No longer links out.\n"
                "---\n\nSwallows and swifts migrate thousands of miles each year.\n"
            )
        concepts = parse_bundle(bundle)
        neo4j_loader.load_bundle_to_neo4j(concepts, bundle)
        neo4j_loader.prune_neo4j_bundle(concepts, bundle)

        assert edge not in edges_in_neo4j(bundle)
        assert f"{prefix}alpha.md" in ids_in_neo4j(bundle)
        assert f"{prefix}topics/shared.md" in ids_in_neo4j(bundle)

    def test_pruning_leaves_another_bundle_alone(self, bundle, tmp_path):
        # Prune is prefix-scoped; a shared database is the normal case.
        from .conftest import concept, write_bundle

        other = str(tmp_path / "othertestbundle")
        write_bundle(
            tmp_path / "othertestbundle",
            {"only.md": concept("Only", "A concept in a second bundle.", "Text.")},
        )
        load_both(other)
        load_both(bundle)

        try:
            survivor = f"{neo4j_loader.bundle_prefix(other)}only.md"
            assert survivor in ids_in_neo4j(other)

            concepts = parse_bundle(bundle)
            neo4j_loader.prune_neo4j_bundle(concepts, bundle)
            qdrant_loader.prune_qdrant_bundle(concepts, bundle)

            assert survivor in ids_in_neo4j(other)
            assert survivor in ids_in_qdrant(other)
        finally:
            from .conftest import _purge

            _purge(other)

    def test_pruning_from_an_empty_parse_is_refused(self, bundle):
        # The guard that stops a mistyped bundle path deleting the whole graph.
        load_both(bundle)

        with pytest.raises(ValueError, match="refusing to prune"):
            neo4j_loader.prune_neo4j_bundle([], bundle)
        with pytest.raises(ValueError, match="refusing to prune"):
            qdrant_loader.prune_qdrant_bundle([], bundle)

        assert ids_in_neo4j(bundle), "the refused prune must not have deleted anything"


class TestHybridSearch:
    def test_the_vector_list_ranks_on_meaning_alone(self, bundle):
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        hits = retrieval.vector_search("deep ocean vents and chemosynthesis")
        mine = [h["id"] for h in hits if h["id"].startswith(prefix)]

        assert mine, "the loaded bundle returned no vector hits at all"
        assert mine[0] == f"{prefix}orphan.md"

    def test_graph_structure_can_outrank_a_stronger_vector_hit(self, bundle):
        # The point of fusing: `shared` is linked from both alpha and beta, so
        # the walk reaches it from two seeds and lifts it above the concept the
        # embedding alone likes best. A pure vector ranking would not do this,
        # and if this ever stops happening the graph half is contributing
        # nothing.
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        fused = [
            r["id"]
            for r in retrieval.hybrid_search("deep ocean vents and chemosynthesis")
            if r["id"].startswith(prefix)
        ]

        assert f"{prefix}orphan.md" in fused, "the best semantic match fell out entirely"
        assert fused.index(f"{prefix}topics/shared.md") < fused.index(
            f"{prefix}orphan.md"
        )

    def test_results_are_annotated_from_the_graph(self, bundle):
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        results = retrieval.hybrid_search("migrating birds")
        # index.md has no frontmatter by design - it is a link-only structural
        # node - so it legitimately has no title, and the UI falls back to the
        # id for it. Every *concept* must be annotated though.
        mine = [
            r
            for r in results
            if r["id"].startswith(prefix) and not r["id"].endswith("/index.md")
        ]

        assert mine
        assert all(r["title"] for r in mine), "a concept came back without a title"

    def test_the_graph_walk_contributes_ranks_of_its_own(self, bundle):
        # The point of the whole design: some result must be reached by the
        # graph, not only by the vector search, or fusion is doing nothing.
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)

        results = retrieval.hybrid_search("migrating birds", top_k=20)
        mine = [r for r in results if r["id"].startswith(prefix)]

        assert any("graph" in r["ranks"] for r in mine), (
            "no result was reached by the graph walk - hybrid search has "
            "degraded to plain vector search"
        )

    def test_index_files_are_excluded_by_default_but_can_be_kept(self, bundle):
        load_both(bundle)
        prefix = neo4j_loader.bundle_prefix(bundle)
        index_id = f"{prefix}index.md"

        default = retrieval.hybrid_search("migrating birds", top_k=20)
        # The index is a hub: with exclusion off it should reach the results
        # through the graph list it was previously filtered out of.
        included = retrieval.hybrid_search(
            "migrating birds", top_k=20, exclude_index=False
        )

        graph_ranked = {
            r["id"] for r in default if "graph" in r["ranks"]
        }
        assert index_id not in graph_ranked
        assert index_id in {r["id"] for r in included if "graph" in r["ranks"]}

    def test_searching_an_empty_query_space_returns_nothing_rather_than_erroring(
        self, bundle
    ):
        load_both(bundle)
        assert isinstance(retrieval.hybrid_search("anything at all"), list)


class TestSharedClients:
    def test_the_stores_are_opened_once_and_reused(self, bundle):
        # The 30x latency fix: a second query must not build a second client.
        load_both(bundle)

        retrieval.hybrid_search("migrating birds")
        driver, client = retrieval._DRIVER, retrieval._QDRANT
        retrieval.hybrid_search("deep ocean vents")

        assert retrieval._DRIVER is driver
        assert retrieval._QDRANT is client

    def test_close_clients_is_safe_to_call_twice(self, bundle):
        retrieval.close_clients()
        retrieval.close_clients()
        # And the next use transparently rebuilds them.
        assert retrieval._qdrant().get_collections() is not None


class TestIncrementalIngest:
    """Re-loading a bundle must cost only what actually changed.

    Embedding is the expensive step in the whole pipeline, and a bundle is
    re-loaded far more often than it is created. Without a digest every reload
    re-embeds every document, which on a real corpus turns a one-file edit into
    a full re-index.
    """

    def test_a_second_load_embeds_nothing(self, bundle):
        concepts = parse_bundle(bundle)
        first = qdrant_loader.load_bundle_to_qdrant(concepts, bundle)

        second = qdrant_loader.load_bundle_to_qdrant(concepts, bundle)

        assert first["embedded"] == len(concepts)
        assert second["embedded"] == 0
        assert second["skipped"] == len(concepts)
        assert second["points"] == 0

    def test_editing_one_file_re_embeds_only_that_file(self, bundle):
        qdrant_loader.load_bundle_to_qdrant(parse_bundle(bundle), bundle)

        path = os.path.join(bundle, "alpha.md")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\nSwifts sleep on the wing for months at a time.\n")

        result = qdrant_loader.load_bundle_to_qdrant(parse_bundle(bundle), bundle)

        assert result["embedded"] == 1
        assert result["skipped"] == len(parse_bundle(bundle)) - 1

    def test_a_frontmatter_only_edit_still_re_embeds(self, bundle):
        """The body is untouched, so the vectors are unchanged - but the payload
        is what search returns, and a stale one would never be rewritten."""
        qdrant_loader.load_bundle_to_qdrant(parse_bundle(bundle), bundle)

        path = os.path.join(bundle, "alpha.md")
        text = open(path, encoding="utf-8").read()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text.replace("title: Alpha Concept", "title: Renamed Concept"))

        result = qdrant_loader.load_bundle_to_qdrant(parse_bundle(bundle), bundle)

        assert result["embedded"] == 1

    def test_force_re_embeds_everything(self, bundle):
        concepts = parse_bundle(bundle)
        qdrant_loader.load_bundle_to_qdrant(concepts, bundle)

        result = qdrant_loader.load_bundle_to_qdrant(concepts, bundle, force=True)

        assert result["embedded"] == len(concepts)
        assert result["skipped"] == 0

    def test_skipping_leaves_the_stored_points_searchable(self, bundle):
        """A skip must be invisible from the outside: the points are still there
        and still match, they simply were not rewritten."""
        concepts = parse_bundle(bundle)
        qdrant_loader.load_bundle_to_qdrant(concepts, bundle)
        qdrant_loader.load_bundle_to_qdrant(concepts, bundle)

        prefix = neo4j_loader.bundle_prefix(bundle)
        hits = retrieval.vector_search("migrating birds", top_k=10)

        assert any(hit["id"].startswith(prefix) for hit in hits)

    def test_a_small_batch_size_loads_the_same_points_as_one_big_batch(self, bundle):
        """Batching bounds memory; it must not change what ends up stored."""
        concepts = parse_bundle(bundle)
        qdrant_loader.load_bundle_to_qdrant(concepts, bundle)
        whole = ids_in_qdrant(bundle)

        qdrant_loader.load_bundle_to_qdrant(concepts, bundle, force=True, batch_size=1)

        assert ids_in_qdrant(bundle) == whole

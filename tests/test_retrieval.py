"""Unit tests for retrieval.py — the fusion maths and the pure guard clauses.

`reciprocal_rank_fusion` needs no services at all. `graph_expand` is only
exercised on the paths that return or raise before a driver is opened.
"""

import pytest

from retrieval import (
    CHUNK_OVERSAMPLE,
    DEGREE_DAMPING,
    GRAPH_SEEDS,
    GRAPH_WEIGHT,
    INDEX_SUFFIX,
    MIN_CANDIDATES,
    RRF_K,
    VECTOR_WEIGHT,
    fold_chunks,
    graph_expand,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    def test_scores_one_list_as_one_over_k_plus_rank(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "b"]})

        assert [r["id"] for r in fused] == ["a", "b"]
        assert fused[0]["score"] == pytest.approx(1 / (RRF_K + 1))
        assert fused[1]["score"] == pytest.approx(1 / (RRF_K + 2))

    def test_sums_the_contribution_of_every_list_an_id_appears_in(self):
        fused = reciprocal_rank_fusion({"vector": ["a"], "graph": ["a"]})

        assert fused[0]["score"] == pytest.approx(2 / (RRF_K + 1))

    def test_appearing_in_both_lists_beats_leading_only_one(self):
        fused = reciprocal_rank_fusion(
            {"vector": ["solo", "both"], "graph": ["both"]}
        )

        assert [r["id"] for r in fused] == ["both", "solo"]

    def test_records_the_one_based_rank_held_in_each_list(self):
        fused = reciprocal_rank_fusion(
            {"vector": ["a", "b"], "graph": ["b", "c"]}
        )
        ranks = {r["id"]: r["ranks"] for r in fused}

        assert ranks["a"] == {"vector": 1}
        assert ranks["b"] == {"vector": 2, "graph": 1}
        assert ranks["c"] == {"graph": 2}

    def test_a_missing_id_contributes_nothing_rather_than_a_penalty(self):
        with_gap = reciprocal_rank_fusion({"vector": ["a"], "graph": ["b"]})
        alone = reciprocal_rank_fusion({"vector": ["a"]})

        assert with_gap[0]["score"] == pytest.approx(alone[0]["score"])

    def test_weights_scale_each_named_list(self):
        fused = reciprocal_rank_fusion(
            {"vector": ["a"], "graph": ["b"]},
            weights={"vector": 1.0, "graph": 0.5},
        )
        scores = {r["id"]: r["score"] for r in fused}

        assert scores["a"] == pytest.approx(1.0 / (RRF_K + 1))
        assert scores["b"] == pytest.approx(0.5 / (RRF_K + 1))
        assert [r["id"] for r in fused] == ["a", "b"]

    def test_an_unnamed_list_defaults_to_weight_one(self):
        weighted = reciprocal_rank_fusion({"graph": ["a"]}, weights={"vector": 0.5})
        plain = reciprocal_rank_fusion({"graph": ["a"]})

        assert weighted[0]["score"] == pytest.approx(plain[0]["score"])

    def test_a_zero_weight_list_still_records_its_ranks(self):
        fused = reciprocal_rank_fusion(
            {"vector": ["a"], "graph": ["b"]}, weights={"graph": 0.0}
        )
        result = {r["id"]: r for r in fused}

        assert result["b"]["score"] == pytest.approx(0.0)
        assert result["b"]["ranks"] == {"graph": 1}

    def test_a_larger_k_flattens_the_gap_between_ranks(self):
        def spread(k):
            fused = reciprocal_rank_fusion({"vector": ["a", "b"]}, k=k)
            return fused[0]["score"] - fused[1]["score"]

        assert spread(1000) < spread(RRF_K) < spread(1)

    def test_ties_break_on_the_id_so_the_order_is_stable(self):
        fused = reciprocal_rank_fusion({"x": ["b"], "y": ["a"], "z": ["c"]})

        assert [r["id"] for r in fused] == ["a", "b", "c"]

    def test_empty_and_all_empty_inputs_fuse_to_nothing(self):
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"vector": [], "graph": []}) == []

    def test_a_repeated_id_within_one_list_keeps_its_best_rank(self):
        fused = reciprocal_rank_fusion({"vector": ["a", "a"]})

        assert len(fused) == 1
        assert fused[0]["score"] == pytest.approx(
            1 / (RRF_K + 1) + 1 / (RRF_K + 2)
        )


class TestGraphExpandGuards:
    def test_no_seeds_returns_no_neighbours_without_a_query(self):
        # Returns before a driver is opened, so this needs no database.
        assert graph_expand([]) == []

    @pytest.mark.parametrize("hops", [0, -1])
    def test_hops_below_one_is_rejected(self, hops):
        with pytest.raises(ValueError, match="hops must be at least 1"):
            graph_expand(["ga4/index.md"], hops=hops)

    def test_hops_is_validated_before_the_empty_seed_shortcut(self):
        with pytest.raises(ValueError):
            graph_expand([], hops=0)


class TestFoldChunks:
    def test_collapses_several_chunks_of_one_concept_into_one_result(self):
        hits = [
            {"id": "b/a.md", "score": 0.9, "chunk_index": 3},
            {"id": "b/a.md", "score": 0.7, "chunk_index": 0},
        ]

        assert fold_chunks(hits, 10) == [
            {"id": "b/a.md", "score": 0.9, "chunk_index": 3}
        ]

    def test_a_concept_scores_as_its_best_chunk_not_its_average(self):
        """One sharply relevant passage in a long document must outrank a
        document that is vaguely on topic throughout - that is what chunking
        was introduced to make possible."""
        hits = [
            {"id": "b/vague.md", "score": 0.60, "chunk_index": 0},
            {"id": "b/sharp.md", "score": 0.95, "chunk_index": 7},
            {"id": "b/sharp.md", "score": 0.10, "chunk_index": 0},
            {"id": "b/sharp.md", "score": 0.05, "chunk_index": 1},
        ]

        assert [hit["id"] for hit in fold_chunks(hits, 10)] == [
            "b/sharp.md",
            "b/vague.md",
        ]

    def test_keeps_the_index_of_the_chunk_that_matched(self):
        hits = [
            {"id": "b/a.md", "score": 0.2, "chunk_index": 0},
            {"id": "b/a.md", "score": 0.8, "chunk_index": 5},
        ]

        assert fold_chunks(hits, 10)[0]["chunk_index"] == 5

    def test_applies_top_k_after_folding_not_before(self):
        """Truncating before the fold would let one long document's chunks
        consume the whole result set."""
        hits = [
            {"id": "b/long.md", "score": 0.9, "chunk_index": 0},
            {"id": "b/long.md", "score": 0.8, "chunk_index": 1},
            {"id": "b/other.md", "score": 0.7, "chunk_index": 0},
        ]

        assert len(fold_chunks(hits, 2)) == 2

    def test_ties_break_on_the_id_so_the_order_is_stable(self):
        hits = [
            {"id": "b/z.md", "score": 0.5, "chunk_index": 0},
            {"id": "b/a.md", "score": 0.5, "chunk_index": 0},
        ]

        assert [hit["id"] for hit in fold_chunks(hits, 10)] == ["b/a.md", "b/z.md"]

    def test_no_hits_folds_to_nothing(self):
        assert fold_chunks([], 10) == []


class TestTuningDefaults:
    def test_the_chunk_oversample_leaves_room_for_repeated_concepts(self):
        assert CHUNK_OVERSAMPLE >= 2

    def test_the_graph_list_is_weighted_below_the_vector_list(self):
        assert 0 < GRAPH_WEIGHT < VECTOR_WEIGHT

    def test_the_candidate_pool_is_deeper_than_the_graph_seed_set(self):
        assert MIN_CANDIDATES > GRAPH_SEEDS > 0

    def test_degree_damping_sits_between_off_and_dividing_out_degree(self):
        assert 0 <= DEGREE_DAMPING <= 1

    def test_index_ids_are_matched_by_a_forward_slashed_suffix(self):
        assert INDEX_SUFFIX == "/index.md"
        assert "ga4/references/index.md".endswith(INDEX_SUFFIX)
        assert not "ga4/reindex.md".endswith(INDEX_SUFFIX)


class TestHybridSearchPassages:
    @pytest.fixture
    def stubbed(self, monkeypatch):
        """Stub both stores: a and b come from the vector list, c only from the walk."""
        import retrieval

        monkeypatch.setattr(
            retrieval,
            "vector_search",
            lambda query, top_k: [
                {"id": "a", "score": 0.9, "chunk_index": 3},
                {"id": "b", "score": 0.8, "chunk_index": 0},
            ],
        )
        monkeypatch.setattr(
            retrieval, "graph_expand", lambda ids, **kwargs: [{"id": "c"}, {"id": "a"}]
        )
        monkeypatch.setattr(retrieval, "_annotate", lambda results: results)
        return retrieval

    def test_a_vector_hit_keeps_the_chunk_that_matched(self, stubbed):
        results = {r["id"]: r for r in stubbed.hybrid_search("q")}
        assert results["a"]["chunk_index"] == 3
        assert results["b"]["chunk_index"] == 0

    def test_a_graph_only_hit_claims_no_chunk(self, stubbed):
        results = {r["id"]: r for r in stubbed.hybrid_search("q")}
        assert "chunk_index" not in results["c"]

    def test_a_vector_hit_carries_its_similarity_and_a_graph_only_hit_none(self, stubbed):
        results = {r["id"]: r for r in stubbed.hybrid_search("q")}
        assert results["a"]["similarity"] == 0.9
        assert results["b"]["similarity"] == 0.8
        assert "similarity" not in results["c"]

    def test_similarity_does_not_change_the_fused_order(self, stubbed):
        # a is in both lists, so it leads; the similarity is carried, not used.
        assert [r["id"] for r in stubbed.hybrid_search("q")][0] == "a"

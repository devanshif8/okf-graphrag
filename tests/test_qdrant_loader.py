"""Unit tests for qdrant_loader.py — id derivation, payload coercion, embed text.

Importing the module pulls in sentence_transformers but never instantiates a
model, so nothing here downloads weights or talks to Qdrant.
"""

import datetime
import uuid

import pytest

from qdrant_loader import (
    CHUNK_WORDS,
    ID_NAMESPACE,
    _payload_value,
    _point_id,
    _stale_point_ids,
    chunk_text,
    concept_chunks,
    content_digest,
    embedding_text,
    prune_qdrant_bundle,
)


class _Point:
    """The subset of a scrolled Qdrant record the prune reads."""

    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class TestPointId:
    def test_is_a_valid_uuid_string(self):
        assert str(uuid.UUID(_point_id("ga4/index.md"))) == _point_id("ga4/index.md")

    def test_is_deterministic_across_calls(self):
        assert _point_id("ga4/index.md") == _point_id("ga4/index.md")

    def test_is_pinned_to_the_uuid5_of_the_concept_id(self):
        # Pinned so a change of namespace or scheme, which would orphan every
        # point already stored, can't pass silently.
        assert _point_id("ga4/index.md") == str(
            uuid.uuid5(ID_NAMESPACE, "ga4/index.md")
        )
        assert _point_id("ga4/index.md") == "7b6d5af6-1aae-5481-9012-08c5ff84a9c7"

    def test_differs_between_concept_ids(self):
        assert _point_id("ga4/index.md") != _point_id("ga4/other.md")

    def test_differs_between_bundles_sharing_a_filename(self):
        assert _point_id("ga4/index.md") != _point_id("gsc/index.md")

    def test_is_a_version_5_uuid(self):
        assert uuid.UUID(_point_id("ga4/index.md")).version == 5


class TestPayloadValue:
    def test_passes_json_native_scalars_through(self):
        for value in ["text", 42, 3.5, True, None]:
            assert _payload_value(value) == value

    def test_iso_formats_dates_and_datetimes(self):
        assert _payload_value(datetime.date(2026, 1, 2)) == "2026-01-02"
        assert (
            _payload_value(datetime.datetime(2026, 1, 2, 3, 4, 5))
            == "2026-01-02T03:04:05"
        )

    def test_recurses_into_lists(self):
        assert _payload_value(["a", datetime.date(2026, 1, 2)]) == ["a", "2026-01-02"]

    def test_converts_tuples_to_lists(self):
        assert _payload_value(("a", "b")) == ["a", "b"]

    def test_recurses_into_dicts_and_stringifies_keys(self):
        value = {"owner": {"team": "analytics", 7: datetime.date(2026, 1, 2)}}

        assert _payload_value(value) == {
            "owner": {"team": "analytics", "7": "2026-01-02"}
        }

    def test_falls_back_to_the_string_form_of_anything_else(self):
        class Opaque:
            def __str__(self):
                return "opaque"

        assert _payload_value(Opaque()) == "opaque"
        assert _payload_value({"k": [Opaque()]}) == {"k": ["opaque"]}


class TestEmbeddingText:
    def test_prefixes_the_description_to_the_body(self):
        concept = {"frontmatter": {"description": "One line."}, "body": "Full body."}

        assert embedding_text(concept) == "One line.\n\nFull body."

    def test_falls_back_to_the_body_when_there_is_no_description(self):
        assert embedding_text({"frontmatter": {}, "body": "  Full body.  "}) == (
            "Full body."
        )

    def test_ignores_a_blank_or_non_string_description(self):
        assert embedding_text({"frontmatter": {"description": "   "},
                               "body": "Body."}) == "Body."
        assert embedding_text({"frontmatter": {"description": ["a"]},
                               "body": "Body."}) == "Body."

    def test_tolerates_missing_frontmatter_and_body(self):
        assert embedding_text({}) == ""
        assert embedding_text({"frontmatter": None, "body": None}) == ""

    def test_keeps_a_description_only_concept_non_empty(self):
        assert embedding_text({"frontmatter": {"description": "Only."},
                               "body": ""}) == "Only."


class TestStalePointIds:
    def test_returns_a_bundle_point_that_is_no_longer_on_disk(self):
        points = [_Point("p1", {"path": "ga4/gone.md", "chunk_index": 0})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md": 1}) == ["p1"]

    def test_keeps_a_point_still_on_disk(self):
        points = [_Point("p1", {"path": "ga4/kept.md", "chunk_index": 0})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md": 1}) == []

    def test_leaves_another_bundle_alone(self):
        # The collection is shared, so pruning ga4 must not touch gsc even
        # though no gsc id is in the keep set.
        points = [_Point("p1", {"path": "gsc/anything.md"})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md": 1}) == []

    def test_skips_a_point_with_no_readable_path(self):
        points = [_Point("p1", {}), _Point("p2", None), _Point("p3", {"path": 7})]

        assert _stale_point_ids(points, "ga4/", {}) == []

    def test_does_not_match_a_bundle_whose_name_merely_starts_the_same(self):
        # 'ga4_test/x.md' starts with 'ga4' but not with the prefix 'ga4/'.
        points = [_Point("p1", {"path": "ga4_test/x.md"})]

        assert _stale_point_ids(points, "ga4/", {}) == []

    def test_returns_chunks_past_the_end_of_a_shortened_concept(self):
        # The document was edited from three chunks down to two: chunk 2 is
        # still in the collection, still matching text the file no longer has.
        points = [
            _Point("p0", {"path": "ga4/a.md", "chunk_index": 0}),
            _Point("p1", {"path": "ga4/a.md", "chunk_index": 1}),
            _Point("p2", {"path": "ga4/a.md", "chunk_index": 2}),
        ]

        assert _stale_point_ids(points, "ga4/", {"ga4/a.md": 2}) == ["p2"]

    def test_keeps_a_point_written_before_chunking_existed(self):
        # No chunk_index means chunk 0, which every surviving concept still
        # has; the next load overwrites it in place.
        points = [_Point("p1", {"path": "ga4/kept.md"})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md": 3}) == []


class TestChunkText:
    def test_short_text_is_one_chunk_held_whole(self):
        assert chunk_text("a short concept body") == ["a short concept body"]

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_every_chunk_fits_inside_the_embedding_window(self):
        body = " ".join(f"word{n}" for n in range(1000))

        chunks = chunk_text(body)

        assert len(chunks) > 1
        assert all(len(chunk.split()) <= CHUNK_WORDS for chunk in chunks)

    def test_no_content_is_lost_between_chunks(self):
        body = " ".join(f"word{n}" for n in range(1000))

        joined = " ".join(chunk_text(body)).split()

        # Overlap repeats words, so compare the set, not the sequence.
        assert set(joined) == set(body.split())

    def test_consecutive_chunks_overlap(self):
        """A span shorter than the overlap must survive whole in some chunk."""
        body = " ".join(f"word{n}" for n in range(1000))

        chunks = chunk_text(body, max_words=100, overlap=20)
        first, second = chunks[0].split(), chunks[1].split()

        assert first[-20:] == second[:20]

    def test_overlap_at_or_above_the_chunk_size_still_terminates(self):
        """Without clamping, the carried tail would refill each chunk and the
        splitter would never advance through the document."""
        body = " ".join(f"word{n}" for n in range(500))

        chunks = chunk_text(body, max_words=50, overlap=500)

        assert len(chunks) < 100
        assert all(len(chunk.split()) <= 50 for chunk in chunks)

    def test_splits_on_paragraphs_before_sentences(self):
        body = "First para.\n\nSecond para.\n\nThird para."

        assert chunk_text(body, max_words=3, overlap=0) == [
            "First para.",
            "Second para.",
            "Third para.",
        ]

    def test_splits_an_overlong_paragraph_on_sentences(self):
        body = " ".join(f"Sentence number {n} here." for n in range(20))

        chunks = chunk_text(body, max_words=8, overlap=0)

        assert len(chunks) > 1
        assert all(chunk.endswith(".") for chunk in chunks)

    def test_hard_splits_a_sentence_with_no_punctuation_to_break_on(self):
        body = " ".join(f"word{n}" for n in range(60))

        chunks = chunk_text(body, max_words=10, overlap=0)

        assert all(len(chunk.split()) <= 10 for chunk in chunks)

    def test_is_deterministic(self):
        body = " ".join(f"word{n}" for n in range(400))

        assert chunk_text(body) == chunk_text(body)


class TestConceptChunks:
    def test_a_textless_concept_still_gets_one_chunk(self):
        """Every concept needs a point, or it exists in Neo4j and not Qdrant
        and the join the two stores rest on breaks."""
        assert concept_chunks({"frontmatter": {}, "body": ""}) == [""]

    def test_a_long_body_becomes_several_chunks(self):
        body = " ".join(f"word{n}" for n in range(1000))

        assert len(concept_chunks({"frontmatter": {}, "body": body})) > 1

    def test_the_description_leads_the_first_chunk(self):
        concept = {"frontmatter": {"description": "A summary."}, "body": "The body."}

        assert concept_chunks(concept)[0].startswith("A summary.")


class TestChunkPointIds:
    def test_chunk_zero_keeps_the_pre_chunking_id(self):
        """Short documents must upsert over themselves rather than orphaning
        their old point under a new id, so no migration is needed."""
        assert _point_id("ga4/index.md", 0) == _point_id("ga4/index.md")
        assert _point_id("ga4/index.md") == "7b6d5af6-1aae-5481-9012-08c5ff84a9c7"

    def test_each_chunk_of_a_concept_gets_its_own_id(self):
        ids = [_point_id("ga4/index.md", n) for n in range(4)]

        assert len(set(ids)) == 4

    def test_chunk_ids_are_deterministic_across_runs(self):
        assert _point_id("ga4/a.md", 3) == _point_id("ga4/a.md", 3)

    def test_chunk_ids_are_valid_uuids(self):
        assert uuid.UUID(_point_id("ga4/a.md", 2)).version == 5


class TestContentDigest:
    def test_is_stable_across_calls(self):
        assert content_digest(["a"], {"path": "b/a.md"}) == content_digest(
            ["a"], {"path": "b/a.md"}
        )

    def test_ignores_the_order_keys_were_inserted_in(self):
        first = content_digest(["a"], {"path": "b/a.md", "title": "A"})
        second = content_digest(["a"], {"title": "A", "path": "b/a.md"})

        assert first == second

    def test_changes_when_the_body_changes(self):
        assert content_digest(["a"], {}) != content_digest(["b"], {})

    def test_changes_when_the_chunk_boundaries_move(self):
        assert content_digest(["ab"], {}) != content_digest(["a", "b"], {})

    def test_changes_when_only_the_frontmatter_changes(self):
        """A corrected title with an untouched body must still invalidate the
        stored points, or the load that should rewrite the payload skips it."""
        assert content_digest(["a"], {"title": "Old"}) != content_digest(
            ["a"], {"title": "New"}
        )

    def test_survives_a_payload_value_json_cannot_encode(self):
        digest = content_digest(["a"], {"when": datetime.date(2026, 9, 10)})

        assert isinstance(digest, str) and len(digest) == 64


class TestPruneGuard:
    def test_refuses_to_prune_from_an_empty_parse(self, tmp_path):
        # A mistyped bundle path parses to nothing; without the guard that
        # would delete every point in the bundle. It must not reach a client.
        with pytest.raises(ValueError, match="empty parse"):
            prune_qdrant_bundle([], str(tmp_path / "ga4"))

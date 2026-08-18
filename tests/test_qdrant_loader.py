"""Unit tests for qdrant_loader.py — id derivation, payload coercion, embed text.

Importing the module pulls in sentence_transformers but never instantiates a
model, so nothing here downloads weights or talks to Qdrant.
"""

import datetime
import uuid

import pytest

from qdrant_loader import (
    ID_NAMESPACE,
    _payload_value,
    _point_id,
    _stale_point_ids,
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
        points = [_Point("p1", {"path": "ga4/gone.md"})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md"}) == ["p1"]

    def test_keeps_a_point_still_on_disk(self):
        points = [_Point("p1", {"path": "ga4/kept.md"})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md"}) == []

    def test_leaves_another_bundle_alone(self):
        # The collection is shared, so pruning ga4 must not touch gsc even
        # though no gsc id is in the keep set.
        points = [_Point("p1", {"path": "gsc/anything.md"})]

        assert _stale_point_ids(points, "ga4/", {"ga4/kept.md"}) == []

    def test_skips_a_point_with_no_readable_path(self):
        points = [_Point("p1", {}), _Point("p2", None), _Point("p3", {"path": 7})]

        assert _stale_point_ids(points, "ga4/", set()) == []

    def test_does_not_match_a_bundle_whose_name_merely_starts_the_same(self):
        # 'ga4_test/x.md' starts with 'ga4' but not with the prefix 'ga4/'.
        points = [_Point("p1", {"path": "ga4_test/x.md"})]

        assert _stale_point_ids(points, "ga4/", set()) == []


class TestPruneGuard:
    def test_refuses_to_prune_from_an_empty_parse(self, tmp_path):
        # A mistyped bundle path parses to nothing; without the guard that
        # would delete every point in the bundle. It must not reach a client.
        with pytest.raises(ValueError, match="empty parse"):
            prune_qdrant_bundle([], str(tmp_path / "ga4"))

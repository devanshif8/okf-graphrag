"""Unit tests for neo4j_loader.py — the id, path and property logic only.

Nothing here opens a driver or touches a database.
"""

import datetime
import json
import os

from neo4j_loader import (
    _bundle_name,
    _properties,
    _property_value,
    _resolve_target,
    concept_id,
)


class TestConceptId:
    def test_is_bundle_name_plus_relative_path(self, tmp_path):
        bundle = tmp_path / "ga4"
        path = bundle / "references" / "metrics" / "purchasers.md"

        assert concept_id(str(path), str(bundle)) == (
            "ga4/references/metrics/purchasers.md"
        )

    def test_uses_forward_slashes_whatever_the_host_separator(self, tmp_path):
        bundle = tmp_path / "ga4"
        path = bundle / "sub" / "deep" / "file.md"

        assert concept_id(str(path), str(bundle)) == "ga4/sub/deep/file.md"
        assert "\\" not in concept_id(str(path), str(bundle))

    def test_is_independent_of_the_working_directory(self, tmp_path, monkeypatch):
        bundle = tmp_path / "bundles" / "ga4"
        (bundle / "sub").mkdir(parents=True)
        path = bundle / "sub" / "file.md"
        path.write_text("x", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        from_root = concept_id(os.path.join("bundles", "ga4", "sub", "file.md"),
                               os.path.join("bundles", "ga4"))

        monkeypatch.chdir(bundle)
        from_inside = concept_id(os.path.join("sub", "file.md"), ".")
        absolute = concept_id(str(path), str(bundle))

        assert from_root == from_inside == absolute == "ga4/sub/file.md"

    def test_trailing_separator_on_the_root_does_not_change_it(self, tmp_path):
        bundle = tmp_path / "ga4"
        path = bundle / "index.md"

        assert concept_id(str(path), str(bundle) + os.sep) == concept_id(
            str(path), str(bundle)
        )

    def test_two_bundles_sharing_a_filename_get_distinct_ids(self, tmp_path):
        a = concept_id(str(tmp_path / "ga4" / "index.md"), str(tmp_path / "ga4"))
        b = concept_id(str(tmp_path / "gsc" / "index.md"), str(tmp_path / "gsc"))

        assert a != b


class TestBundleName:
    def test_is_the_directory_name(self, tmp_path):
        assert _bundle_name(str(tmp_path / "bundles" / "ga4")) == "ga4"

    def test_ignores_a_trailing_separator(self, tmp_path):
        assert _bundle_name(str(tmp_path / "ga4") + os.sep) == "ga4"

    def test_resolves_dot_against_the_working_directory(self, tmp_path, monkeypatch):
        bundle = tmp_path / "ga4"
        bundle.mkdir()
        monkeypatch.chdir(bundle)

        assert _bundle_name(".") == "ga4"


class TestResolveTarget:
    def test_resolves_relative_to_the_linking_file(self):
        source = os.path.join("bundle", "references", "a.md")

        assert _resolve_target(source, "b.md") == os.path.join(
            "bundle", "references", "b.md"
        )

    def test_strips_an_anchor(self):
        source = os.path.join("bundle", "a.md")

        assert _resolve_target(source, "b.md#section") == os.path.join("bundle", "b.md")

    def test_walks_up_with_dot_dot(self):
        source = os.path.join("bundle", "references", "metrics", "a.md")

        assert _resolve_target(source, "../../index.md") == os.path.join(
            "bundle", "index.md"
        )

    def test_handles_a_source_with_no_directory_part(self):
        assert _resolve_target("a.md", "b.md") == "b.md"

    def test_an_anchor_only_target_resolves_to_its_own_directory(self):
        assert _resolve_target(os.path.join("bundle", "a.md"), "#section") == "bundle"


class TestPropertyValue:
    def test_passes_primitives_through_unchanged(self):
        for value in ["text", 42, 3.5, True, None]:
            assert _property_value(value) == value

    def test_passes_dates_through_unchanged(self):
        day = datetime.date(2026, 1, 2)
        moment = datetime.datetime(2026, 1, 2, 3, 4, 5)

        assert _property_value(day) is day
        assert _property_value(moment) is moment

    def test_keeps_a_list_of_primitives_as_a_list(self):
        assert _property_value(["metric", "ecommerce"]) == ["metric", "ecommerce"]

    def test_keeps_a_mixed_primitive_list_as_a_list(self):
        assert _property_value([1, "two", 3.0, True]) == [1, "two", 3.0, True]

    def test_converts_a_tuple_to_a_list(self):
        assert _property_value(("a", "b")) == ["a", "b"]

    def test_json_serializes_a_nested_dict(self):
        value = {"owner": {"team": "analytics"}}

        assert json.loads(_property_value(value)) == value

    def test_json_serializes_a_list_of_lists(self):
        value = [[1, 2], [3, 4]]

        assert json.loads(_property_value(value)) == value

    def test_json_serializes_dicts_inside_a_list_elementwise(self):
        # Each dict becomes a JSON string, so the list itself stays storable.
        result = _property_value([{"a": 1}, "plain"])

        assert result == ['{"a": 1}', "plain"]

    def test_falls_back_to_a_string_for_unserializable_values(self):
        class Opaque:
            def __repr__(self):
                return "<opaque>"

        assert _property_value({"k": Opaque()}) == '{"k": "<opaque>"}'

    def test_keeps_non_ascii_readable(self):
        assert _property_value({"name": "café"}) == '{"name": "café"}'


class TestProperties:
    def test_flattens_frontmatter_and_stringifies_keys(self):
        props = _properties({"title": "One", 7: "seven", "tags": ["a", "b"]})

        assert props == {"title": "One", "7": "seven", "tags": ["a", "b"]}

    def test_drops_a_colliding_path_key(self):
        assert _properties({"path": "/somewhere/else.md", "title": "One"}) == {
            "title": "One"
        }

    def test_treats_none_and_empty_frontmatter_as_no_properties(self):
        assert _properties(None) == {}
        assert _properties({}) == {}

    def test_applies_property_value_coercion_to_each_field(self):
        props = _properties({"meta": {"nested": {"deep": 1}}})

        assert json.loads(props["meta"]) == {"nested": {"deep": 1}}

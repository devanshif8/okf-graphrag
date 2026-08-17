"""Unit tests for parser.py — pure filesystem/text logic, no services needed."""

import os

import pytest

from parser import LINK_RE, _md_files, parse_bundle, parse_okf_file


def write(path, text):
    """Write text to path, creating parent directories, and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


CONCEPT = """---
title: Purchasers
description: Users who bought something.
tags:
  - metric
  - ecommerce
---

Body text linking to [another](other.md) and [anchored](deep/thing.md#section).
Also an [external](https://example.com) link and a [doc](notes.txt).
"""


def test_parses_frontmatter_body_and_links(tmp_path):
    path = write(str(tmp_path / "purchasers.md"), CONCEPT)

    result = parse_okf_file(path)

    assert result["path"] == path
    assert result["frontmatter"] == {
        "title": "Purchasers",
        "description": "Users who bought something.",
        "tags": ["metric", "ecommerce"],
    }
    # The body keeps whatever followed the closing '---', blank line included.
    assert result["body"].lstrip("\n").startswith("Body text linking to")
    assert "title: Purchasers" not in result["body"]
    assert "is_index" not in result


def test_links_keep_anchors_and_drop_non_md_targets(tmp_path):
    path = write(str(tmp_path / "purchasers.md"), CONCEPT)

    links = parse_okf_file(path)["links"]

    assert links == [
        {"text": "another", "target": "other.md"},
        {"text": "anchored", "target": "deep/thing.md#section"},
    ]


def test_link_re_ignores_external_and_non_md_targets():
    text = "[a](x.md) [b](https://example.com) [c](notes.txt) [d](y.md#top)"

    assert LINK_RE.findall(text) == [("a", "x.md"), ("d", "y.md#top")]


def test_index_without_frontmatter_parses_as_link_only_node(tmp_path):
    body = "# Index\n\nSee [one](one.md) and [two](sub/two.md).\n"
    path = write(str(tmp_path / "index.md"), body)

    result = parse_okf_file(path)

    assert result["is_index"] is True
    assert result["frontmatter"] == {}
    assert result["body"] == body
    assert result["links"] == [
        {"text": "one", "target": "one.md"},
        {"text": "two", "target": "sub/two.md"},
    ]


def test_index_with_frontmatter_parses_as_an_ordinary_concept(tmp_path):
    path = write(str(tmp_path / "index.md"), "---\ntitle: Root\n---\n\n[a](a.md)\n")

    result = parse_okf_file(path)

    assert result["frontmatter"] == {"title": "Root"}
    assert "is_index" not in result


def test_non_index_without_frontmatter_is_skipped(tmp_path, capsys):
    path = write(str(tmp_path / "plain.md"), "# Just a heading\n\n[a](a.md)\n")

    assert parse_okf_file(path) is None
    assert "no frontmatter" in capsys.readouterr().out


def test_bad_yaml_is_skipped(tmp_path, capsys):
    path = write(str(tmp_path / "bad.md"), "---\ntitle: [unclosed\n---\n\nbody\n")

    assert parse_okf_file(path) is None
    assert "bad YAML" in capsys.readouterr().out


@pytest.mark.parametrize(
    "frontmatter", ["- one\n- two", "just a string", "42"], ids=["list", "str", "int"]
)
def test_non_mapping_frontmatter_is_skipped(tmp_path, capsys, frontmatter):
    path = write(str(tmp_path / "odd.md"), f"---\n{frontmatter}\n---\n\nbody\n")

    assert parse_okf_file(path) is None
    assert "not a mapping" in capsys.readouterr().out


def test_missing_file_is_reported_not_raised(tmp_path, capsys):
    assert parse_okf_file(str(tmp_path / "nope.md")) is None
    assert "could not read" in capsys.readouterr().out


def test_crlf_frontmatter_is_handled(tmp_path):
    path = str(tmp_path / "crlf.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"---\r\ntitle: Windows\r\n---\r\n\r\n[a](a.md)\r\n")

    result = parse_okf_file(path)

    assert result["frontmatter"] == {"title": "Windows"}
    assert result["links"] == [{"text": "a", "target": "a.md"}]


def test_md_files_recurses_and_ignores_other_extensions(tmp_path):
    write(str(tmp_path / "b.md"), "x")
    write(str(tmp_path / "a.md"), "x")
    write(str(tmp_path / "notes.txt"), "x")
    write(str(tmp_path / "sub" / "c.md"), "x")

    found = [os.path.relpath(p, str(tmp_path)) for p in _md_files(str(tmp_path))]

    assert sorted(found) == sorted(["a.md", "b.md", os.path.join("sub", "c.md")])
    # Files within a directory come back in sorted order.
    assert found.index("a.md") < found.index("b.md")


def test_parse_bundle_recurses_and_drops_failures(tmp_path):
    write(str(tmp_path / "index.md"), "[one](one.md)\n")
    write(str(tmp_path / "one.md"), "---\ntitle: One\n---\n\nbody\n")
    write(str(tmp_path / "sub" / "two.md"), "---\ntitle: Two\n---\n\nbody\n")
    write(str(tmp_path / "sub" / "broken.md"), "no frontmatter here\n")
    write(str(tmp_path / "sub" / "bad.md"), "---\ntitle: [unclosed\n---\n\nbody\n")
    write(str(tmp_path / "ignored.txt"), "not markdown\n")

    results = parse_bundle(str(tmp_path))

    titles = sorted(r["frontmatter"].get("title", "") for r in results)
    assert titles == ["", "One", "Two"]  # "" is the frontmatter-less index
    assert sum(1 for r in results if r.get("is_index")) == 1


def test_parse_bundle_on_empty_directory_returns_empty_list(tmp_path):
    assert parse_bundle(str(tmp_path)) == []

"""Unit tests for convert_to_okf.py — pure filesystem/text logic, no services needed.

The .txt and .html paths run everywhere. The .docx and .pdf paths need their
optional libraries, so those tests skip when the library is absent, matching how
convert_to_okf.py imports them lazily.
"""

import os

import pytest

from convert_to_okf import (
    ConversionError,
    _summarize,
    _titleize,
    build_frontmatter,
    convert_directory,
    convert_file,
    render_okf,
    slugify,
)
from parser import parse_okf_file


def write(path, text):
    """Write text to path, creating parent directories, and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


HTML = """<html><head><title>Revenue Recognition Policy</title>
<style>p { color: red }</style></head>
<body><h1>Scope</h1><p>Applies to all delivered orders.</p>
<ul><li>30-day return window</li></ul>
<script>tracker()</script></body></html>
"""


def test_txt_converts_to_parseable_okf(tmp_path):
    src = write(str(tmp_path / "quarterly_notes.txt"), "Q3 revenue grew 12%.\n")

    out = convert_file(src, str(tmp_path / "out"))

    assert os.path.basename(out) == "quarterly-notes.md"
    parsed = parse_okf_file(out)
    assert parsed is not None, "converted output must round-trip through parser.py"
    assert parsed["frontmatter"]["title"] == "Quarterly Notes"
    assert parsed["frontmatter"]["status"] == "draft"
    assert parsed["frontmatter"]["sources"][0]["resource"].endswith("quarterly_notes.txt")
    assert "Q3 revenue grew 12%." in parsed["body"]


def test_html_title_wins_and_markup_becomes_markdown(tmp_path):
    src = write(str(tmp_path / "policy.html"), HTML)

    out = convert_file(src, str(tmp_path / "out"))

    parsed = parse_okf_file(out)
    # The <title> tag beats the filename stem.
    assert parsed["frontmatter"]["title"] == "Revenue Recognition Policy"
    assert os.path.basename(out) == "revenue-recognition-policy.md"
    assert "# Scope" in parsed["body"]
    assert "- 30-day return window" in parsed["body"]
    # script/style contents are dropped, not treated as prose.
    assert "tracker()" not in parsed["body"]
    assert "color: red" not in parsed["body"]


def test_body_whitespace_is_normalized(tmp_path):
    src = write(str(tmp_path / "messy.txt"), "One\r\n\r\n\r\n\r\nTwo   \n\ttrailing \n")

    out = convert_file(src, str(tmp_path / "out"))

    body = parse_okf_file(out)["body"].strip()
    assert "\r" not in body
    assert "\n\n\n" not in body
    assert not any(line != line.rstrip() for line in body.split("\n"))


def test_unsupported_extension_raises(tmp_path):
    src = write(str(tmp_path / "notes.rst"), "text")

    with pytest.raises(ConversionError, match="unsupported file type"):
        convert_file(src, str(tmp_path / "out"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConversionError, match="not a file"):
        convert_file(str(tmp_path / "absent.txt"), str(tmp_path / "out"))


def test_empty_document_raises_instead_of_writing(tmp_path):
    src = write(str(tmp_path / "blank.txt"), "   \n\n\t\n")
    out_dir = str(tmp_path / "out")

    with pytest.raises(ConversionError, match="no extractable text"):
        convert_file(src, out_dir)
    assert not os.path.exists(os.path.join(out_dir, "blank.md"))


def test_repeat_conversions_do_not_overwrite(tmp_path):
    src = write(str(tmp_path / "report.txt"), "First body.\n")
    out_dir = str(tmp_path / "out")

    first = convert_file(src, out_dir)
    second = convert_file(src, out_dir)

    assert os.path.basename(first) == "report.md"
    assert os.path.basename(second) == "report-2.md"
    assert parse_okf_file(first)["body"].strip() == "First body."


def test_convert_directory_reports_each_outcome(tmp_path):
    write(str(tmp_path / "docs" / "good.txt"), "Readable prose.\n")
    write(str(tmp_path / "docs" / "sub" / "nested.html"), HTML)
    write(str(tmp_path / "docs" / "sub" / "empty.txt"), "  \n")
    write(str(tmp_path / "docs" / "notes.rst"), "unsupported")

    result = convert_directory(str(tmp_path / "docs"), str(tmp_path / "out"))

    assert len(result["converted"]) == 2, "recurses into subdirectories"
    assert [os.path.basename(s) for s, _ in result["failed"]] == ["empty.txt"]
    assert [os.path.basename(s) for s in result["skipped"]] == ["notes.rst"]
    # One bad document does not abort the batch.
    assert all(parse_okf_file(out) is not None for _src, out in result["converted"])


def test_convert_directory_rejects_a_file_path(tmp_path):
    src = write(str(tmp_path / "solo.txt"), "text")

    with pytest.raises(ConversionError, match="not a directory"):
        convert_directory(src, str(tmp_path / "out"))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Revenue Recognition Policy", "revenue-recognition-policy"),
        ("Q3 / FY26 — notes!", "q3-fy26-notes"),
        ("  spaced  out  ", "spaced-out"),
        ("!!!", "document"),
        ("", "document"),
    ],
)
def test_slugify(name, expected):
    assert slugify(name) == expected


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("q3_report", "Q3 Report"),
        ("margin-standard", "Margin Standard"),
        ("RFC_2119", "RFC 2119"),
    ],
)
def test_titleize(stem, expected):
    assert _titleize(stem) == expected


def test_summarize_skips_headings_and_truncates():
    body = "# Heading\n\n" + "word " * 200
    summary = _summarize(body)

    assert not summary.startswith("#")
    assert len(summary) <= 304 and summary.endswith("...")


def test_summarize_of_headings_only_body_is_empty():
    assert _summarize("# One\n\n## Two") == ""


def test_frontmatter_records_source_metadata():
    fm = build_frontmatter(
        "/docs/margin.docx", "Gross margin is revenue minus COGS.", {"author": "jsmith"}
    )

    assert fm["type"] == "Document"
    assert fm["tags"] == ["converted"]
    assert fm["generated"]["at"].endswith("Z")
    # Nothing machine-converted has been checked by a human yet.
    assert fm["status"] == "draft"
    assert "verified" not in fm
    source = fm["sources"][0]
    assert source["resource"].endswith("margin.docx")
    assert source["format"] == "docx"
    assert source["author"] == "jsmith"


def test_generated_by_follows_the_actor_convention():
    """SPEC §7: a tool actor is '<producer>/<version>', not a bare filename."""
    by = build_frontmatter("/docs/x.txt", "Body.", {})["generated"]["by"]

    producer, _, version = by.partition("/")
    assert producer and version, f"{by!r} is not '<producer>/<version>'"
    assert not by.endswith(".py")
    # The human: prefix drives trust tiers (SPEC §5.3) and must not be claimed.
    assert not by.startswith("human:")


def test_source_resource_is_followable(tmp_path):
    """SPEC §5.1: `resource` must be something a consumer can resolve."""
    src = write(str(tmp_path / "notes.txt"), "Body.")

    resource = build_frontmatter(src, "Body.", {})["sources"][0]["resource"]

    assert resource.startswith("file:///")
    assert resource.endswith("notes.txt")
    # A bare basename would not resolve from anywhere.
    assert resource != "notes.txt"


def test_empty_description_key_is_omitted_not_blank(tmp_path):
    """SPEC §4.1: description is a sentence; blank would break snippet fallback."""
    src = write(str(tmp_path / "headings_only.txt"), "# One\n\n## Two\n")

    out = convert_file(src, str(tmp_path / "out"))

    fm = parse_okf_file(out)["frontmatter"]
    assert "description" not in fm
    assert fm["title"] == "Headings Only"


def test_render_okf_shape():
    text = render_okf({"title": "T"}, "Body.")

    assert text.startswith("---\ntitle: T\n---\n\n")
    assert text.endswith("Body.\n")


def test_docx_headings_and_metadata(tmp_path):
    docx = pytest.importorskip("docx", reason="python-docx not installed")

    document = docx.Document()
    document.core_properties.title = "Margin Standard"
    document.core_properties.author = "human:jsmith@acme"
    document.add_heading("Definition", level=1)
    document.add_paragraph("Gross margin is revenue minus COGS.")
    document.add_paragraph("Excludes freight", style="List Bullet")
    src = str(tmp_path / "margin.docx")
    os.makedirs(str(tmp_path), exist_ok=True)
    document.save(src)

    out = convert_file(src, str(tmp_path / "out"))

    parsed = parse_okf_file(out)
    assert parsed["frontmatter"]["title"] == "Margin Standard"
    assert parsed["frontmatter"]["sources"][0]["author"] == "human:jsmith@acme"
    assert "# Definition" in parsed["body"]
    assert "- Excludes freight" in parsed["body"]


def test_pdf_text_and_page_count(tmp_path):
    pytest.importorskip("pypdf", reason="pypdf not installed")
    canvas = pytest.importorskip(
        "reportlab.pdfgen.canvas", reason="reportlab not installed"
    )

    src = str(tmp_path / "orders_spec.pdf")
    pdf = canvas.Canvas(src)
    pdf.setTitle("Orders Table Spec")
    pdf.drawString(72, 720, "One row per order.")
    pdf.showPage()
    pdf.drawString(72, 720, "Page two.")
    pdf.save()

    out = convert_file(src, str(tmp_path / "out"))

    parsed = parse_okf_file(out)
    assert parsed["frontmatter"]["title"] == "Orders Table Spec"
    assert parsed["frontmatter"]["sources"][0]["pages"] == 2
    assert "One row per order." in parsed["body"]
    assert "Page two." in parsed["body"]

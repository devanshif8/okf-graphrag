"""Convert common document formats (.pdf, .docx, .txt, .html) into OKF markdown.

Each input document becomes one OKF file: YAML frontmatter plus a markdown body,
the same shape parser.py already reads, so converted files can be loaded by
neo4j_loader.py and qdrant_loader.py without further work.

This is a standalone preprocessing step; it does not touch parser.py,
neo4j_loader.py, qdrant_loader.py, or retrieval.py.

Note: this converter does NOT extract cross-links between documents the way
native OKF files carry them. Source formats have no notion of an OKF concept
path, so no `[text](other.md)` links are emitted and the resulting bundle has no
:LINKS_TO edges. The graph will only contain link structure for converted
documents that are manually cross-referenced afterward — until then the loaders
see isolated nodes, and retrieval's degree scaling has nothing to contribute.
"""

import argparse
import datetime
import html
import logging
import os
import pathlib
import re
import sys

import yaml

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".html", ".htm")

# Written to `generated.by`, so it follows the OKF actor convention (SPEC §7):
# '<producer>/<version>' for a tool. Bump the version when the extraction or the
# frontmatter it emits changes meaningfully.
CONVERTER_ID = "convert_to_okf/0.1"

# Collapse runs of blank lines, and strip the whitespace extractors leave behind.
BLANK_RUN_RE = re.compile(r"\n{3,}")
TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)
SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")
HEADING_STYLE_RE = re.compile(r"Heading (\d)")


class ConversionError(Exception):
    """Raised when a document cannot be read, or its format has no extractor."""


def _now() -> str:
    """Current UTC time as an ISO-8601 'Z' timestamp, matching OKF frontmatter."""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _tidy(text: str) -> str:
    """Normalize extracted text: unify newlines, drop trailing space, cap blank runs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TRAILING_WS_RE.sub("", text)
    return BLANK_RUN_RE.sub("\n\n", text).strip()


def slugify(name: str) -> str:
    """Turn a title or filename into a lowercase, hyphenated OKF filename stem."""
    slug = SLUG_STRIP_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "document"


def _titleize(stem: str) -> str:
    """Readable title from a filename stem, e.g. 'q3_report' -> 'Q3 Report'."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", stem) if w]
    return " ".join(w if w.isupper() else w.capitalize() for w in words) or stem


def _extract_txt(path: str) -> tuple[str, dict]:
    """Read a plain-text file as-is; it carries no document metadata."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return _tidy(fh.read()), {}


def _extract_pdf(path: str) -> tuple[str, dict]:
    """Extract page text and document info from a PDF via pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConversionError(
            "reading .pdf files needs pypdf (pip install pypdf)"
        ) from exc

    # pypdf logs a line per malformed cross-reference entry ("Ignoring wrong
    # pointing object ..."). Real-world PDFs produce hundreds of these and they
    # drown the per-file warnings; the extraction itself still succeeds.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    reader = PdfReader(path)
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    meta = {"pages": len(reader.pages)}
    info = reader.metadata or {}
    for key, field in (("/Title", "title"), ("/Author", "author")):
        value = info.get(key)
        if value:
            meta[field] = str(value).strip()
    return _tidy("\n\n".join(p for p in pages if p)), meta


def _extract_docx(path: str) -> tuple[str, dict]:
    """Extract paragraphs from a .docx, mapping Word heading styles to markdown."""
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConversionError(
            "reading .docx files needs python-docx (pip install python-docx)"
        ) from exc

    document = docx.Document(path)
    lines = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "") if paragraph.style else ""
        heading = HEADING_STYLE_RE.match(style)
        if heading:
            lines.append("#" * min(int(heading.group(1)), 6) + f" {text}")
        elif style.startswith("List"):
            lines.append(f"- {text}")
        else:
            lines.append(text)

    meta = {}
    props = document.core_properties
    if props.title:
        meta["title"] = props.title.strip()
    if props.author:
        meta["author"] = props.author.strip()
    return _tidy("\n\n".join(lines)), meta


def _extract_html(path: str) -> tuple[str, dict]:
    """Extract readable text from HTML, keeping headings and list items as markdown."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ConversionError(
            "reading .html files needs beautifulsoup4 (pip install beautifulsoup4)"
        ) from exc

    with open(path, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    meta = {}
    if soup.title and soup.title.string:
        meta["title"] = html.unescape(soup.title.string).strip()

    lines = []
    body = soup.body or soup
    for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        if tag.name[0] == "h" and tag.name[1:].isdigit():
            lines.append("#" * int(tag.name[1]) + f" {text}")
        elif tag.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    if not lines:  # No structural tags: fall back to the document's whole text.
        lines = [body.get_text("\n", strip=True)]
    return _tidy("\n\n".join(lines)), meta


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
    ".html": _extract_html,
    ".htm": _extract_html,
}


def _summarize(body: str, limit: int = 300) -> str:
    """First real paragraph of the body, trimmed, as the frontmatter description."""
    for block in body.split("\n\n"):
        text = " ".join(block.split())
        if not text or text.startswith("#"):
            continue
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0] + "..."
    return ""


def build_frontmatter(source_path: str, body: str, meta: dict) -> dict:
    """Assemble the OKF frontmatter for one converted document.

    `status: draft` and the absent `verified` block are deliberate: nothing about
    a machine conversion has been checked by a human yet.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    title = meta.get("title") or _titleize(stem)
    source = {
        "id": slugify(stem),
        # SPEC §5.1 wants a resource a consumer can actually follow. The source
        # document sits outside the bundle, so a bare filename would not resolve
        # anywhere; a file:// URI of its absolute path does.
        "resource": pathlib.Path(source_path).resolve().as_uri(),
        "title": title,
        "format": os.path.splitext(source_path)[1].lower().lstrip("."),
    }
    if meta.get("author"):
        source["author"] = meta["author"]
    if meta.get("pages"):
        source["pages"] = meta["pages"]

    frontmatter = {
        "type": "Document",
        "title": title,
        "description": _summarize(body),
        "tags": ["converted"],
        "generated": {"by": CONVERTER_ID, "at": _now()},
        "status": "draft",
        "sources": [source],
    }
    # SPEC §4.1 describes `description` as a single sentence. A body with no
    # prose to summarize (headings only) has none, and an empty string would
    # make index generators and search snippets render blank rather than fall
    # back to the title, so drop the key entirely instead.
    if not frontmatter["description"]:
        del frontmatter["description"]
    return frontmatter


def render_okf(frontmatter: dict, body: str) -> str:
    """Serialize frontmatter and body into the '---' delimited OKF file text."""
    header = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{header}\n---\n\n{body}\n"


def _unique_path(output_dir: str, stem: str) -> str:
    """An unused '<stem>.md' path in output_dir, suffixing -2, -3, ... on collision."""
    candidate = os.path.join(output_dir, f"{stem}.md")
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(output_dir, f"{stem}-{counter}.md")
        counter += 1
    return candidate


def convert_file(input_path: str, output_dir: str) -> str:
    """Convert one document into an OKF markdown file, returning the path written.

    The extractor is chosen by extension; unsupported extensions, unreadable
    files, and documents with no extractable text raise ConversionError rather
    than writing a half-empty concept.
    """
    extension = os.path.splitext(input_path)[1].lower()
    extractor = EXTRACTORS.get(extension)
    if extractor is None:
        raise ConversionError(
            f"unsupported file type '{extension or '(none)'}': expected one of "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if not os.path.isfile(input_path):
        raise ConversionError(f"not a file: {input_path}")

    try:
        body, meta = extractor(input_path)
    except ConversionError:
        raise
    except Exception as exc:  # Extractor libraries raise their own error types.
        raise ConversionError(f"could not read {input_path}: {exc}") from exc

    if not body:
        raise ConversionError(f"no extractable text in {input_path}")

    frontmatter = build_frontmatter(input_path, body, meta)
    os.makedirs(output_dir, exist_ok=True)
    out_path = _unique_path(output_dir, slugify(frontmatter["title"]))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(render_okf(frontmatter, body))
    return out_path


def convert_directory(input_dir: str, output_dir: str) -> dict:
    """Convert every supported document under input_dir, recursively.

    Returns {"converted": [(source, output), ...], "failed": [(source, reason), ...],
    "skipped": [source, ...]}, so one unreadable document never aborts the batch.
    """
    if not os.path.isdir(input_dir):
        raise ConversionError(f"not a directory: {input_dir}")

    result = {"converted": [], "failed": [], "skipped": []}
    for root, _dirs, files in os.walk(input_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.splitext(name)[1].lower() not in EXTRACTORS:
                result["skipped"].append(path)
                continue
            try:
                result["converted"].append((path, convert_file(path, output_dir)))
            except ConversionError as exc:
                result["failed"].append((path, str(exc)))
    return result


def main() -> None:
    """Convert a document, or a directory of them, into OKF files."""
    ap = argparse.ArgumentParser(
        description="Convert .pdf/.docx/.txt/.html documents into OKF markdown."
    )
    ap.add_argument("path", help="a document to convert, or a directory of them")
    ap.add_argument(
        "--output-dir", required=True, help="directory to write the .md files into"
    )
    args = ap.parse_args()

    # Warnings quote source filenames, which routinely carry characters the
    # console codec cannot encode (a Windows cp1252 terminal chokes on the
    # narrow no-break space scanner apps put in filenames). Degrade those
    # characters instead of dying mid-report and losing the rest of the summary.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):  # not a reconfigurable text stream
            pass

    if os.path.isdir(args.path):
        result = convert_directory(args.path, args.output_dir)
        for source, reason in result["failed"]:
            print(f"warning: {source}: {reason}")
        print(
            f"{len(result['converted'])} converted, {len(result['failed'])} failed, "
            f"{len(result['skipped'])} skipped -> {args.output_dir}"
        )
        if not result["converted"]:
            raise SystemExit(1)
        return

    try:
        out_path = convert_file(args.path, args.output_dir)
    except ConversionError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"1 converted, 0 failed, 0 skipped -> {out_path}")


if __name__ == "__main__":
    main()

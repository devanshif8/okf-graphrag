"""Split full-text books into chapter-sized documents before conversion.

`qdrant_loader` embeds one vector per file, and `all-MiniLM-L6-v2` truncates its
input at 256 word-pieces — roughly 180 words. Converting a whole book into a
single OKF file therefore indexes its first page and silently discards the rest,
so a retrieval score measured that way would be measuring a missing chunking
layer rather than the retrieval design. Splitting first keeps each document
inside the model's window and turns a shelf of books into a corpus of thousands
of documents, which is the scale worth testing at anyway.

Chapters are the split point rather than a fixed word count because they are a
boundary the author already drew: a chapter is about one thing, which is what a
retrieval unit should be. A book with no detectable chapter headings falls back
to fixed-size chunks, since some text is better than none.

    python eval/split_books.py path/to/books --output-dir path/to/chapters

The output is plain `.txt`, one file per chapter, laid out as
`<book-slug>/<nn>-<chapter-slug>.txt` — the input `convert_to_okf.py` expects.
"""

import argparse
import os
import re

# Project Gutenberg wraps each book in a licence header and footer. They are
# identical across books, so leaving them in would give every document a large
# shared passage — the linker would find its keyphrases there and link the
# corpus into a clique on boilerplate.
GUTENBERG_START = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.M)
GUTENBERG_END = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.M)

# A chapter heading is a short line, alone on its line, opening with a chapter
# word and a numeral — arabic, roman, or spelled out. Requiring the line to be
# short is what keeps prose that merely mentions "chapter four" from splitting a
# book mid-paragraph.
CHAPTER_RE = re.compile(
    r"^[ \t]*("
    r"(?:CHAPTER|Chapter|BOOK|Book|PART|Part|LETTER|Letter|ACT|Act)"
    r"[ \t]+[IVXLCDM\d][IVXLCDM\d]*\.?"
    r"(?:[ \t]*[-—:.][ \t]*.{0,60})?"
    r")[ \t]*$",
    re.MULTILINE,
)

# Below this a "chapter" is almost always a table-of-contents entry or a stray
# heading rather than real text, and it would enter the corpus as noise.
MIN_CHAPTER_WORDS = 120

# Fallback chunk size, in words, for books with no detectable chapters. Well
# above the embedding window on purpose: the point of the fallback is to produce
# coherent passages, not to match the truncation limit exactly.
CHUNK_WORDS = 600

SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, limit: int = 60) -> str:
    """Reduce a title to a filesystem-safe slug."""
    slug = SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:limit].strip("-") or "untitled"


def strip_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg licence header and footer, if present."""
    start = GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text.strip()


def _chunk(text: str, words_per_chunk: int) -> list[tuple[str, str]]:
    """Split text into fixed-size word chunks as (title, body) pairs."""
    words = text.split()
    chunks = []
    for index in range(0, len(words), words_per_chunk):
        part = " ".join(words[index : index + words_per_chunk])
        if len(part.split()) >= MIN_CHAPTER_WORDS:
            chunks.append((f"Part {len(chunks) + 1}", part))
    return chunks


def split_chapters(text: str, chunk_words: int = CHUNK_WORDS) -> list[tuple[str, str]]:
    """Split one book's text into (title, body) pairs, one per chapter.

    Everything before the first chapter heading is dropped: it is the title
    page, dedication and table of contents, none of which is retrievable
    content. A book with fewer than two detected headings is chunked instead,
    on the assumption that the detection failed rather than that the book
    genuinely has one chapter.
    """
    matches = list(CHAPTER_RE.finditer(text))
    if len(matches) < 2:
        return _chunk(text, chunk_words)

    chapters = []
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if len(body.split()) >= MIN_CHAPTER_WORDS:
            chapters.append((match.group(1).strip(), body))

    # Detection can match a table of contents and nothing else, leaving every
    # candidate too short to keep. Chunking is better than returning nothing.
    return chapters or _chunk(text, chunk_words)


def split_book(path: str, output_dir: str, chunk_words: int = CHUNK_WORDS) -> int:
    """Split one book file into per-chapter .txt files. Returns the count."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = strip_boilerplate(fh.read())

    book_title = os.path.splitext(os.path.basename(path))[0].replace("_", " ").strip()
    book = slugify(book_title)
    chapters = split_chapters(text, chunk_words)
    if not chapters:
        return 0

    destination = os.path.join(output_dir, book)
    os.makedirs(destination, exist_ok=True)

    for number, (title, body) in enumerate(chapters, start=1):
        # The book slug leads the *filename*, not just the directory, because
        # convert_to_okf.py flattens the tree and names each output from the
        # basename alone. Without it every book's 'Chapter I' converts to
        # chapter-i.md, chapter-i-2.md, chapter-i-3.md — the converter's
        # collision suffix keeps them distinct but strips the one thing an
        # evaluation needs to read, which book a retrieved chapter came from.
        name = f"{book}-{number:03d}-{slugify(title, 40)}.txt"
        with open(os.path.join(destination, name), "w", encoding="utf-8") as fh:
            # The heading leads the file so the converter lifts a real title and
            # description instead of opening mid-sentence.
            fh.write(f"{book_title} — {title}\n\n{body}\n")
    return len(chapters)


def main() -> None:
    """Split every book under a directory and report what was written."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("books", help="directory of .txt books (or a single .txt)")
    ap.add_argument("--output-dir", required=True, help="where to write chapters")
    ap.add_argument(
        "--chunk-words", type=int, default=CHUNK_WORDS,
        help=f"chunk size for books with no chapters (default: {CHUNK_WORDS})",
    )
    ap.add_argument(
        "--limit", type=int, help="stop after this many books (for a quick trial)"
    )
    args = ap.parse_args()

    if os.path.isfile(args.books):
        paths = [args.books]
    else:
        paths = sorted(
            os.path.join(root, name)
            for root, _, names in os.walk(args.books)
            for name in names
            if name.lower().endswith(".txt")
        )
    if args.limit:
        paths = paths[: args.limit]

    if not paths:
        raise SystemExit(f"no .txt files found under {args.books}")

    total = 0
    for path in paths:
        count = split_book(path, args.output_dir, args.chunk_words)
        total += count
        status = f"{count} chapters" if count else "no usable text - skipped"
        print(f"  {os.path.basename(path)}: {status}")

    print(f"{len(paths)} books, {total} chapter files written to {args.output_dir}")


if __name__ == "__main__":
    main()

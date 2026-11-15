"""Download the evaluation library of Project Gutenberg books.

The library is several domains of public-domain books, listed with their domain
in `eval/library.csv`. The domain label is the ground truth for domain-level
evaluation, so the manifest — not the downloaded files — is the source of truth,
and every download is checked against it: a wrong Gutenberg id would silently
put a novel on the chemistry shelf and corrupt every score measured over it.

    python eval/fetch_library.py --output-dir library/books

Files are named after the manifest title, which `split_books.py` turns into the
book slug that prefixes every chapter file. Already-downloaded books are skipped,
so an interrupted run can simply be restarted.
"""

import argparse
import csv
import os
import re
import time
import urllib.request

MANIFEST = os.path.join(os.path.dirname(__file__), "library.csv")
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.M)

# Gutenberg asks automated clients to pace their requests.
DELAY_SECONDS = 2


def load_manifest(path: str = MANIFEST) -> list[dict]:
    """Read the manifest rows: gutenberg_id, domain, title."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def filename(title: str) -> str:
    """A filesystem-safe name for a book, kept readable for split_books.py."""
    return re.sub(r"[^A-Za-z0-9' -]+", "", title).strip() + ".txt"


def title_matches(expected: str, text: str) -> bool:
    """Whether the downloaded file's Title header agrees with the manifest.

    Compares the first significant word of the manifest title, which is loose
    enough to survive subtitles and edition notes but catches a wrong id.
    """
    found = TITLE_RE.search(text[:5000])
    if not found:
        return False
    words = [w for w in re.findall(r"[a-z]+", expected.lower()) if len(w) > 3]
    return all(w in found.group(1).lower() for w in words[:2])


def fetch(row: dict, output_dir: str) -> str:
    """Download one book if missing. Returns a one-word status."""
    path = os.path.join(output_dir, filename(row["title"]))
    if os.path.exists(path):
        return "cached"
    request = urllib.request.Request(
        URL.format(id=row["gutenberg_id"]), headers={"User-Agent": "okf-graphrag-eval"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        text = response.read().decode("utf-8", errors="replace")
    if not title_matches(row["title"], text):
        return "MISMATCH"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    time.sleep(DELAY_SECONDS)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", required=True, help="where to write the books")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    failures = 0
    for row in load_manifest():
        try:
            status = fetch(row, args.output_dir)
        except OSError as exc:
            status = f"FAILED ({exc})"
        failures += status not in ("cached", "downloaded")
        print(f"  [{row['domain']}] {row['title']}: {status}")

    if failures:
        raise SystemExit(f"{failures} book(s) failed; fix the manifest and re-run")
    print("all books present")


if __name__ == "__main__":
    main()

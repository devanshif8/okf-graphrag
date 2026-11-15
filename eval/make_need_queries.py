"""Turn the hand-written needs in `eval/needs.csv` into a scorable query set.

This is the accuracy test the whole library was built for. The other query sets
are passages lifted out of the corpus, which share their wording with the
document that should answer them; these are needs phrased the way a person asks
a librarian for something, so the retriever has to bridge the gap between how
the need is worded and how the book is written. That gap is where the graph
half can contribute something vector similarity cannot, and it is the only query
shape that can show it.

    python eval/make_need_queries.py library/shelf --output eval/need_queries.jsonl
    python eval/evaluate.py eval/need_queries.jsonl --group-by kind

A need is labelled with the domain or domains that should answer it, and every
chapter on those shelves counts as relevant. That is deliberately coarse: it
scores "did the right shelf come back", not "did the right chapter", so a hit on
an astronomy chapter about spectra counts as correct for a question about
eclipses. Read the result as a shelf-level measurement. Judging the chapters
themselves means reading the top k of every run by hand, which this cannot do
for you.

Two consequences of that coarseness are carried in the query set so the scorer
can allow for them. Each need gets a `base_rate` — the precision a random
ranking would score — because a cross-cutting need with two shelves has roughly
twice as many correct answers available as a single-shelf one, and would beat it
on precision while being no better answered. And each carries `kind`, so the
cross-cutting needs can be read as their own group: they are the ones whose
answer is spread across books that no single vector neighbourhood covers.

Unlike the sampled-passage sets there is no `exclude`: a hand-written need was
not lifted out of any book, so no book has to be held out of its ranking.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.fetch_library import load_manifest  # noqa: E402
from eval.make_domain_queries import book_domains, book_of  # noqa: E402
from neo4j_loader import concept_id  # noqa: E402
from parser import parse_bundle  # noqa: E402

NEEDS = os.path.join(os.path.dirname(__file__), "needs.csv")

# How a need names more than one shelf: "cookery|chemistry".
SEPARATOR = "|"

# A need this library cannot answer, labelled `none` in the domains column.
# Written as an explicit word rather than a blank cell, because a blank is what
# an unfinished row looks like and the two must not be confused.
ABSENT = "none"


def load_needs(path: str = NEEDS) -> list[dict]:
    """Read the needs file: a `domains` label and the `need` itself.

    Blank lines are skipped so the file stays comfortable to edit by hand, but
    a row missing either column is an error rather than a silent drop — a need
    that vanishes on the way into the query set takes its result with it.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    needs = []
    for number, row in enumerate(rows, start=2):  # header is line 1
        domains = (row.get("domains") or "").strip()
        need = (row.get("need") or "").strip()
        if not domains and not need:
            continue
        if not domains or not need:
            raise SystemExit(f"{path}:{number}: every row needs both 'domains' and 'need'")
        needs.append(
            {
                "need": need,
                "domains": [d.strip() for d in domains.split(SEPARATOR) if d.strip()],
            }
        )
    return needs


def chapters_by_domain(bundle_dir: str, manifest: list[dict]) -> dict:
    """Map each domain to the node ids of every chapter on its shelf."""
    domains = book_domains(manifest)
    slugs = list(domains)

    shelves: dict = defaultdict(set)
    for concept in parse_bundle(bundle_dir):
        if concept.get("is_index"):
            continue
        book = book_of(os.path.basename(concept["path"]), slugs)
        shelves[domains[book]].add(concept_id(concept["path"], bundle_dir))
    return shelves


def build_need_queries(bundle_dir: str, manifest: list[dict], needs: list[dict]) -> list[dict]:
    """Expand each need's domain label into the set of chapters that answer it.

    An unknown domain label stops the build. The labels are the entire ground
    truth here and they are typed by hand, so a typo would quietly produce a
    need with an empty relevant set that scores zero however well it was
    answered, and drag the headline down with it.
    """
    shelves = chapters_by_domain(bundle_dir, manifest)
    total = sum(len(ids) for ids in shelves.values())
    if not total:
        raise SystemExit(f"no chapters found in {bundle_dir}")

    queries = []
    for need in needs:
        if ABSENT in need["domains"] and need["domains"] != [ABSENT]:
            raise SystemExit(
                f"need {need['need']!r} is labelled both {ABSENT!r} and a shelf - "
                "a need either has an answer in this library or it does not"
            )
        if need["domains"] == [ABSENT]:
            queries.append(
                {
                    "query": need["need"],
                    "relevant": [],
                    "domain": ABSENT,
                    "kind": "probe",
                    "base_rate": 0.0,
                }
            )
            continue

        unknown = [d for d in need["domains"] if d not in shelves]
        if unknown:
            raise SystemExit(
                f"unknown domain {unknown[0]!r} in need {need['need']!r} - "
                f"known domains: {', '.join(sorted(shelves))}"
            )
        relevant: set = set().union(*(shelves[d] for d in need["domains"]))
        queries.append(
            {
                "query": need["need"],
                "relevant": sorted(relevant),
                "domain": SEPARATOR.join(need["domains"]),
                "kind": "cross-cutting" if len(need["domains"]) > 1 else "single",
                "base_rate": len(relevant) / total,
            }
        )
    return queries


def write_queries(queries: list[dict], path: str) -> None:
    """Write a query set as one JSON object per line."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for entry in queries:
            fh.write(json.dumps(entry) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="the loaded library bundle, e.g. library/shelf")
    ap.add_argument("--needs", default=NEEDS, help=f"the needs .csv (default: {NEEDS})")
    ap.add_argument("--output", required=True, help="where to write the .jsonl")
    ap.add_argument(
        "--probes",
        help=f"where to write the needs labelled '{ABSENT}', which have no answer here",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.bundle):
        raise SystemExit(f"no such bundle directory: {args.bundle}")

    try:
        queries = build_need_queries(args.bundle, load_manifest(), load_needs(args.needs))
    except FileNotFoundError as exc:
        raise SystemExit(f"could not read {exc.filename}") from None
    except LookupError as exc:
        raise SystemExit(
            f"{exc} - the bundle and eval/library.csv disagree; was it built "
            "from this manifest?"
        ) from None

    probes = [entry for entry in queries if entry["kind"] == "probe"]
    scored = [entry for entry in queries if entry["kind"] != "probe"]
    if probes and not args.probes:
        raise SystemExit(
            f"{len(probes)} needs are labelled '{ABSENT}' and cannot be scored for "
            "precision - pass --probes to write them to their own file"
        )

    write_queries(scored, args.output)
    counts: dict = defaultdict(int)
    for entry in scored:
        counts[entry["kind"]] += 1
    print(f"{len(scored)} needs written to {args.output}")
    for kind, count in sorted(counts.items()):
        print(f"  {kind}: {count}")

    if probes:
        write_queries(probes, args.probes)
        print(f"{len(probes)} probes written to {args.probes}")


if __name__ == "__main__":
    main()

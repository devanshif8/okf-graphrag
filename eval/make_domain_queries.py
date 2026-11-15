"""Generate a domain-level query set for the multi-domain evaluation library.

A floor, not the accuracy result. Each query is a passage sampled from one
chapter, and every chapter from the same domain counts as relevant — so the
question asked is "does a need from this shelf come back with books from this
shelf", scored with Precision@k. A high score means nothing is grossly broken;
it does not mean the system finds the *right* book, which needs hand-written
needs and hand-judged answers.

Two choices keep the floor from flattering the system:

- **The query's own book is excluded** from every ranking. A passage shares its
  vocabulary with the rest of its book far more than with its domain, so
  same-book hits would score as domain precision while measuring nothing but
  repetition. What remains is the librarian's question: other books on the shelf.
- **Queries are balanced across domains** and each carries the precision a
  random ranking would score, because the domains are not the same size and a
  large one wins precision for free.

    python eval/make_domain_queries.py library/shelf --output eval/domain_queries.jsonl
    python eval/evaluate.py eval/domain_queries.jsonl --group-by domain
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.fetch_library import filename, load_manifest  # noqa: E402
from eval.make_queries import sample_passages  # noqa: E402
from eval.split_books import slugify  # noqa: E402
from neo4j_loader import concept_id  # noqa: E402
from parser import parse_bundle  # noqa: E402

PER_DOMAIN = 25


def book_domains(manifest: list[dict]) -> dict:
    """Map each book slug, as split_books.py names it, to its domain."""
    return {
        slugify(os.path.splitext(filename(row["title"]))[0]): row["domain"]
        for row in manifest
    }


def book_of(name: str, slugs: list[str]) -> str:
    """The book a chapter file belongs to, by its slug prefix.

    Longest slug first, so a book whose slug is a prefix of another's cannot
    claim the other's chapters.
    """
    for slug in sorted(slugs, key=len, reverse=True):
        if name.startswith(slug + "-"):
            return slug
    raise LookupError(f"{name} matches no book in the manifest")


def build_domain_queries(
    bundle_dir: str,
    manifest: list[dict],
    per_domain: int = PER_DOMAIN,
    seed: int = 0,
) -> list[dict]:
    """Sample `per_domain` passage queries from each domain's chapters."""
    rng = random.Random(seed)
    domains = book_domains(manifest)
    slugs = list(domains)

    concepts = [c for c in parse_bundle(bundle_dir) if not c.get("is_index")]
    ids_by_book: dict = defaultdict(set)
    chapters_by_domain: dict = defaultdict(list)
    for concept in sorted(concepts, key=lambda c: c["path"]):
        book = book_of(os.path.basename(concept["path"]), slugs)
        ids_by_book[book].add(concept_id(concept["path"], bundle_dir))
        chapters_by_domain[domains[book]].append((book, concept))

    total = len(concepts)
    queries = []
    for domain in sorted(chapters_by_domain):
        chapters = chapters_by_domain[domain]
        in_domain = set().union(
            *(ids for book, ids in ids_by_book.items() if domains[book] == domain)
        )
        # Drawn without replacement until the domain runs out, then a chapter
        # that yields no usable passage is skipped rather than retried forever.
        for book, concept in rng.sample(chapters, len(chapters)):
            if sum(q["domain"] == domain for q in queries) >= per_domain:
                break
            passages = sample_passages(concept["body"], 1, "any", rng)
            if not passages:
                continue
            own_book = ids_by_book[book]
            relevant = in_domain - own_book
            queries.append(
                {
                    "query": passages[0][0],
                    "relevant": sorted(relevant),
                    "exclude": sorted(own_book),
                    "domain": domain,
                    "book": book,
                    "base_rate": len(relevant) / (total - len(own_book)),
                }
            )
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="the loaded library bundle, e.g. library/shelf")
    ap.add_argument("--output", required=True, help="where to write the .jsonl")
    ap.add_argument(
        "--per-domain", type=int, default=PER_DOMAIN,
        help=f"queries per domain (default: {PER_DOMAIN})",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = ap.parse_args()

    queries = build_domain_queries(
        args.bundle, load_manifest(), per_domain=args.per_domain, seed=args.seed
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for entry in queries:
            fh.write(json.dumps(entry) + "\n")

    counts: dict = defaultdict(int)
    for entry in queries:
        counts[entry["domain"]] += 1
    print(f"{len(queries)} queries written to {args.output}")
    for domain, count in sorted(counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()

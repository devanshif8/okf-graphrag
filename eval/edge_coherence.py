"""Measure how much of the linker's graph is signal, with no queries at all.

The cheapest diagnostic in the evaluation plan, and the one to run first. Every
book in the library carries a domain label in `eval/library.csv`, so each
`:LINKS_TO` edge can be asked a single question: does it stay on its own shelf?
If the answer is near chance, the graph is noise, fusion has nothing to harvest,
and hybrid search is vector search wearing a hat — which is worth knowing before
building a query set and hand-judging its answers.

    python eval/edge_coherence.py library/shelf
    python eval/edge_coherence.py library/shelf --output library/edge_coherence.json

Read the **cross-book** row, not the headline. Three quarters of the edges join
two chapters of the same book, where staying in-domain is guaranteed by
construction and measures nothing; only an edge that leaves its book is the
linker making a claim about two documents it had no a-priori reason to connect.

Each rate is printed against the rate a random graph would score. Chance is
computed per edge from its *source* — the source's domain resampled against
every other node it could have pointed at — rather than over all pairs, because
the domains are not the same size and the sources are not spread evenly across
them: a linker that fired mostly inside the largest domain would otherwise look
coherent for free.

Edges are read from the bundle on disk rather than from Neo4j. `bundle_edges` is
exactly what the loader writes, so the two agree, and the diagnostic then needs
no database running.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.fetch_library import load_manifest  # noqa: E402
from eval.make_domain_queries import book_domains, book_of  # noqa: E402
from neo4j_loader import bundle_edges, concept_id  # noqa: E402
from parser import parse_bundle  # noqa: E402

# How many leaking domain pairs to name in the report.
TOP_LEAKS = 10


def assign_nodes(bundle_dir: str, manifest: list[dict]) -> dict:
    """Map every chapter node id in the bundle to its (book, domain).

    Index hubs are dropped: they belong to no single book, and they link by
    construction rather than by similarity, so counting them would measure the
    bundle's table of contents rather than the linker.
    """
    domains = book_domains(manifest)
    slugs = list(domains)

    assignment = {}
    for concept in parse_bundle(bundle_dir):
        if concept.get("is_index"):
            continue
        book = book_of(os.path.basename(concept["path"]), slugs)
        assignment[concept_id(concept["path"], bundle_dir)] = (book, domains[book])
    return assignment


def measure(edges: list[tuple], assignment: dict) -> dict:
    """Score an edge list against a node -> (book, domain) assignment.

    Edges touching a node outside the assignment — an index hub, a document from
    another bundle — are skipped rather than counted as a miss, so the rates
    describe the edges actually judged.
    """
    total = len(assignment)
    if total < 2:
        raise ValueError("need at least two assigned nodes to measure coherence")

    book_sizes = Counter(book for book, _ in assignment.values())
    domain_sizes = Counter(domain for _, domain in assignment.values())

    judged = same_domain = same_book = 0
    cross_book = cross_book_same_domain = 0
    domain_chance = book_chance = cross_book_chance = 0.0
    leaks: Counter = Counter()
    by_domain: dict = defaultdict(lambda: {"edges": 0, "same_domain": 0, "chance": 0.0})

    for source, target in edges:
        if source not in assignment or target not in assignment:
            continue
        source_book, source_domain = assignment[source]
        target_book, target_domain = assignment[target]

        judged += 1
        # Chance for this edge: hold the source fixed, resample the target
        # uniformly over every other node.
        domain_chance += (domain_sizes[source_domain] - 1) / (total - 1)
        book_chance += (book_sizes[source_book] - 1) / (total - 1)

        if target_domain == source_domain:
            same_domain += 1
        if target_book == source_book:
            same_book += 1
            continue

        # Cross-book only past here: the source's own book is off the table, so
        # chance is resampled over what is left of the library.
        outside = total - book_sizes[source_book]
        cross_book += 1
        share = (domain_sizes[source_domain] - book_sizes[source_book]) / outside
        cross_book_chance += share
        stats = by_domain[source_domain]
        stats["edges"] += 1
        stats["chance"] += share
        if target_domain == source_domain:
            cross_book_same_domain += 1
            stats["same_domain"] += 1
        else:
            leaks[tuple(sorted((source_domain, target_domain)))] += 1

    def rate(count: int, out_of: int) -> float:
        return count / out_of if out_of else 0.0

    return {
        "nodes": total,
        "edges": judged,
        "skipped": len(edges) - judged,
        "same_domain": rate(same_domain, judged),
        "same_domain_chance": rate(domain_chance, judged),
        "same_book": rate(same_book, judged),
        "same_book_chance": rate(book_chance, judged),
        "cross_book": {
            "edges": cross_book,
            "same_domain": rate(cross_book_same_domain, cross_book),
            "chance": rate(cross_book_chance, cross_book),
        },
        "by_domain": {
            domain: {
                "edges": stats["edges"],
                "same_domain": rate(stats["same_domain"], stats["edges"]),
                "chance": rate(stats["chance"], stats["edges"]),
            }
            for domain, stats in by_domain.items()
        },
        "leaks": [
            {"domains": list(pair), "edges": count}
            for pair, count in leaks.most_common(TOP_LEAKS)
        ],
    }


def _report(result: dict) -> str:
    """The measurement as a plain-text report, each rate beside its chance."""
    cross = result["cross_book"]
    lines = [
        f"{result['nodes']} chapters, {result['edges']} edges judged"
        + (f" ({result['skipped']} skipped)" if result["skipped"] else ""),
        "",
        f"{'':<24}{'rate':>8}{'chance':>9}{'lift':>8}",
    ]

    def row(label: str, rate: float, chance: float) -> str:
        lift = f"{rate / chance:.1f}x" if chance else "-"
        return f"{label:<24}{rate:>8.3f}{chance:>9.3f}{lift:>8}"

    lines.append(row("same domain", result["same_domain"], result["same_domain_chance"]))
    lines.append(row("same book", result["same_book"], result["same_book_chance"]))
    lines.append(row("cross-book, same domain", cross["same_domain"], cross["chance"]))

    if result["by_domain"]:
        lines += ["", f"cross-book edges by source domain ({cross['edges']} edges)", ""]
        lines.append(f"{'':<24}{'edges':>8}{'rate':>8}{'chance':>9}")
        for domain, stats in sorted(result["by_domain"].items()):
            lines.append(
                f"{domain:<24}{stats['edges']:>8}"
                f"{stats['same_domain']:>8.3f}{stats['chance']:>9.3f}"
            )

    if result["leaks"]:
        lines += ["", "where cross-book edges leave their domain", ""]
        for leak in result["leaks"]:
            pair = " - ".join(leak["domains"])
            share = leak["edges"] / cross["edges"] if cross["edges"] else 0.0
            lines.append(f"  {pair:<40}{leak['edges']:>6}  {share:>6.1%} of cross-book")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="the linked library bundle, e.g. library/shelf")
    ap.add_argument("--output", help="also write the full measurement as JSON here")
    args = ap.parse_args()

    if not os.path.isdir(args.bundle):
        raise SystemExit(f"no such bundle directory: {args.bundle}")

    concepts = parse_bundle(args.bundle)
    if not concepts:
        raise SystemExit(f"no concepts found in {args.bundle}")

    try:
        assignment = assign_nodes(args.bundle, load_manifest())
    except LookupError as exc:
        raise SystemExit(
            f"{exc} - the bundle and eval/library.csv disagree; was it built "
            "from this manifest?"
        ) from None

    result = measure(bundle_edges(concepts, args.bundle), assignment)
    print(_report(result))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwritten to {args.output}")


if __name__ == "__main__":
    main()

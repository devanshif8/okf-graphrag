"""Build a hand-judging pool for the hand-written needs, and score what comes back.

Domain-level relevance — every chapter on the right shelf counts — is the cheap
ground truth, and on the needs it saturates: the three retrievers score within a
hair of each other because landing somewhere in astronomy is not hard. The only
way past that ceiling is to say which *chapters* actually answer each need, and
nothing but a person reading them can decide that.

    python eval/judge.py pool library/shelf eval/need_queries.jsonl \\
        --output library/judgements.csv          # writes the sheet to fill in
    python eval/judge.py build library/judgements.csv \\
        --queries eval/need_queries.jsonl --output eval/judged_queries.jsonl
    python eval/evaluate.py eval/judged_queries.jsonl --group-by kind

`pool` runs all three retrievers, pools the union of their top-k, and writes one
row per (need, chapter) with a snippet to read. `build` turns the filled-in
sheet back into a query set `evaluate.py` can score.

Three things keep the judgements honest:

- **The pool is blind.** A row does not say which retriever found the chapter,
  or at what rank, and the rows are shuffled within each need. A judge who can
  see that hybrid ranked something first will mark it relevant more often, and
  that bias would land squarely on the comparison the whole exercise exists to
  make.
- **Judge on the text, not the shelf.** The row carries the chapter's title,
  because a librarian sees that too, but not the domain label — the point of
  judging is to get past "astronomy came back for an astronomy question".
- **Judging is resumable.** Re-running `pool` over an existing sheet keeps every
  mark already made and only adds rows that are new, so the needs can be edited
  and the pool regrown without throwing away an afternoon's work.

What pooling costs, and it is worth stating in the write-up: only chapters some
retriever returned are ever judged, and everything unjudged is scored as
irrelevant. That is fair between the three runs here, since all three
contributed to the pool, but a fourth retriever scored against this file later
would be punished for finding good chapters these three missed.
"""

import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable  # noqa: E402
from qdrant_client.http.exceptions import (  # noqa: E402
    ResponseHandlingException,
    UnexpectedResponse,
)

from eval.evaluate import load_queries, run_query  # noqa: E402
from neo4j_loader import NEO4J_URI, NEO4J_USER, concept_id  # noqa: E402
from parser import parse_bundle  # noqa: E402
from qdrant_loader import QDRANT_URL  # noqa: E402
from retrieval import DEGREE_DAMPING, GRAPH_WEIGHT, VECTOR_WEIGHT, close_clients  # noqa: E402

# Judged per need, per retriever. Ten is what a person actually reads before
# deciding a search worked, and three runs at ten pool to well under thirty.
DEPTH = 10

# Enough of the chapter to judge it by without opening the file.
SNIPPET = 400

FIELDS = ["need", "chapter", "title", "relevant", "snippet"]

# What a judge writes in the `relevant` column.
YES = {"1", "y", "yes", "true"}
NO = {"0", "n", "no", "false"}


def chapter_text(bundle_dir: str) -> dict:
    """Map every chapter node id to its title and an opening snippet.

    The title is the chapter's own — "The Book Of Household Management 002
    Chapter Ii" — which names the book it came from as a librarian would see it,
    without labelling the shelf.
    """
    texts = {}
    for concept in parse_bundle(bundle_dir):
        if concept.get("is_index"):
            continue
        body = " ".join((concept.get("body") or "").split())
        frontmatter = concept.get("frontmatter") or {}
        texts[concept_id(concept["path"], bundle_dir)] = {
            "title": str(frontmatter.get("title") or os.path.basename(concept["path"])),
            "snippet": body[:SNIPPET],
        }
    return texts


def read_sheet(path: str) -> dict:
    """Read an existing judgement sheet as {(need, chapter): relevant-cell}.

    Missing file means nothing has been judged yet, which is the normal first
    run rather than an error.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return {
            (row["need"], row["chapter"]): (row.get("relevant") or "").strip()
            for row in csv.DictReader(fh)
        }


def pool_for(query: dict, depth: int, damping: float) -> list[str]:
    """The union of what every retriever returns for one need.

    Deduplicated across runs: a chapter all three found is one judgement, not
    three, and which run found it is deliberately not recorded.
    """
    rankings = run_query(
        query["query"],
        depth,
        [("hybrid", VECTOR_WEIGHT, GRAPH_WEIGHT)],
        damping,
        True,
        frozenset(query.get("exclude", ())),
    )
    pooled: dict = {}  # dict, not set: insertion order keeps the file stable
    for ranked in rankings.values():
        for identifier in ranked:
            pooled.setdefault(identifier, None)
    return list(pooled)


def build_pool(
    queries: list[dict],
    texts: dict,
    existing: dict,
    depth: int = DEPTH,
    damping: float = DEGREE_DAMPING,
    seed: int = 0,
) -> list[dict]:
    """Retrieve every need and lay the pooled chapters out as sheet rows."""
    rng = random.Random(seed)
    rows = []
    for number, query in enumerate(queries, start=1):
        need = query["query"]
        print(f"  [{number}/{len(queries)}] {need[:60]}", file=sys.stderr)
        pooled = pool_for(query, depth, damping)
        # Shuffled so rank order carries no hint about which run ranked what.
        rng.shuffle(pooled)
        for identifier in pooled:
            text = texts.get(identifier, {})
            rows.append(
                {
                    "need": need,
                    "chapter": identifier,
                    "title": text.get("title", ""),
                    "relevant": existing.get((need, identifier), ""),
                    "snippet": text.get("snippet", ""),
                }
            )
    return rows


def write_sheet(rows: list[dict], path: str) -> None:
    """Write the sheet, with the columns in a fixed order for a diffable file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # utf-8-sig: the BOM is what stops Excel reading the sheet as cp1252.
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_judgements(path: str) -> tuple[dict, int]:
    """Read the filled-in sheet as {need: {relevant chapter ids}}, plus a blank count.

    A cell that is neither a yes nor a no is an error, not a no: a typo silently
    read as 'irrelevant' would lower every score that chapter could have
    contributed to, and nothing downstream would show it happened.
    """
    judged: dict = {}
    blanks = 0
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for number, row in enumerate(csv.DictReader(fh), start=2):
            need = row["need"]
            judged.setdefault(need, set())
            cell = (row.get("relevant") or "").strip().lower()
            if not cell:
                blanks += 1
            elif cell in YES:
                judged[need].add(row["chapter"])
            elif cell not in NO:
                raise SystemExit(
                    f"{path}:{number}: '{cell}' is not a judgement - "
                    f"use one of {sorted(YES)} or {sorted(NO)}, or leave it blank"
                )
    return judged, blanks


def apply_judgements(queries: list[dict], judged: dict, total: int) -> list[dict]:
    """Replace each need's shelf-level relevant set with its judged chapters.

    A need nobody marked relevant anywhere is dropped rather than kept as a
    guaranteed zero: it measures the judge running out of afternoon, not the
    retriever, and averaging it in would understate every run equally but
    silently.
    """
    built = []
    for query in queries:
        relevant = sorted(judged.get(query["query"], ()))
        if not relevant:
            continue
        built.append(
            query | {
                "relevant": relevant,
                "base_rate": len(relevant) / total,
                "judged": True,
            }
        )
    return built


def _pool(args) -> None:
    if not os.path.isdir(args.bundle):
        raise SystemExit(f"no such bundle directory: {args.bundle}")
    # Probes have an empty relevant set on purpose, and pooling never reads it.
    queries = load_queries(args.queries, require_relevant=False)
    texts = chapter_text(args.bundle)
    existing = read_sheet(args.output)

    print(f"pooling {len(queries)} needs at depth {args.depth}...", file=sys.stderr)
    try:
        rows = build_pool(queries, texts, existing, args.depth, args.degree_damping)
    except (ResponseHandlingException, ConnectionError):
        raise SystemExit(f"could not reach Qdrant at {QDRANT_URL} - is it running?") from None
    except UnexpectedResponse as exc:
        raise SystemExit(f"Qdrant rejected the query: {exc}") from None
    except ServiceUnavailable:
        raise SystemExit(
            f"could not reach Neo4j at {NEO4J_URI} - is the database running?"
        ) from None
    except AuthError:
        raise SystemExit(f"Neo4j rejected the credentials for user '{NEO4J_USER}'") from None
    except ConfigurationError as exc:
        raise SystemExit(f"bad Neo4j configuration (NEO4J_URI={NEO4J_URI}): {exc}") from None
    finally:
        close_clients()

    write_sheet(rows, args.output)
    kept = sum(1 for row in rows if row["relevant"])
    print(f"\n{len(rows)} rows written to {args.output}")
    print(f"  {kept} already judged, {len(rows) - kept} to go")
    print(f"  fill the 'relevant' column with 1 or 0, then: {sys.argv[0]} build ...")


def _build(args) -> None:
    queries = load_queries(args.queries)
    judged, blanks = read_judgements(args.sheet)
    if blanks and not args.partial:
        raise SystemExit(
            f"{blanks} rows in {args.sheet} are still blank - finish them, or "
            "pass --partial to score the sheet as it stands (blanks count as "
            "irrelevant, which understates every run)"
        )

    total = len({identifier for query in queries for identifier in query["relevant"]})
    built = apply_judgements(queries, judged, total)
    if not built:
        raise SystemExit(f"nothing in {args.sheet} is marked relevant - is the column filled in?")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for entry in built:
            fh.write(json.dumps(entry) + "\n")

    dropped = len(queries) - len(built)
    marks = sum(len(entry["relevant"]) for entry in built)
    print(f"{len(built)} needs written to {args.output} ({marks} relevant chapters)")
    if dropped:
        print(f"  {dropped} needs dropped: nothing marked relevant for them")
    if blanks:
        print(f"  warning: {blanks} rows left blank, counted as irrelevant")


def probe_report(queries: list[dict], judged: dict) -> dict:
    """Score the probe set: how often a need with no answer got answered anyway.

    A probe is a question this library cannot answer. Every retriever still
    returns its k results — they always do — so the measurement is what a reader
    makes of them: a probe where nothing was marked relevant is the honest
    outcome, and one where something was is either a mislabelled probe or a
    chapter that genuinely fits, both of which are worth seeing by name.
    """
    rows = []
    for query in queries:
        hits = sorted(judged.get(query["query"], ()))
        rows.append({"need": query["query"], "hits": hits})
    answered = [row for row in rows if row["hits"]]
    return {
        "probes": len(rows),
        "answered": len(answered),
        "rate": len(answered) / len(rows) if rows else 0.0,
        "rows": rows,
    }


def _probes(args) -> None:
    queries = load_queries(args.queries, require_relevant=False)
    judged, blanks = read_judgements(args.sheet)
    report = probe_report(queries, judged)

    print(f"{report['probes']} probes - needs this library cannot answer\n")
    print(f"  {report['answered']} had something marked relevant ({report['rate']:.1%})")
    if blanks:
        print(f"  warning: {blanks} rows left blank, counted as not relevant")
    for row in report["rows"]:
        if row["hits"]:
            print(f"\n  {row['need']}")
            for identifier in row["hits"]:
                print(f"    {identifier}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    pool = sub.add_parser("pool", help="write the sheet of chapters to judge")
    pool.add_argument("bundle", help="the loaded library bundle, e.g. library/shelf")
    pool.add_argument("queries", help="the need query set from make_need_queries.py")
    pool.add_argument("--output", required=True, help="the judgement sheet .csv")
    pool.add_argument(
        "--depth", type=int, default=DEPTH,
        help=f"chapters pooled per retriever (default: {DEPTH})",
    )
    pool.add_argument(
        "--degree-damping", type=float, default=DEGREE_DAMPING,
        help=f"degree damping in the graph walk (default: {DEGREE_DAMPING})",
    )
    pool.set_defaults(func=_pool)

    build = sub.add_parser("build", help="turn the filled-in sheet into a query set")
    build.add_argument("sheet", help="the judgement sheet .csv")
    build.add_argument("--queries", required=True, help="the need query set it was pooled from")
    build.add_argument("--output", required=True, help="where to write the judged .jsonl")
    build.add_argument(
        "--partial", action="store_true",
        help="score a half-finished sheet, counting blanks as irrelevant",
    )
    build.set_defaults(func=_build)

    probes = sub.add_parser("probes", help="report on the needs with no answer here")
    probes.add_argument("sheet", help="the judgement sheet .csv")
    probes.add_argument("--queries", required=True, help="the probe set from make_need_queries.py")
    probes.set_defaults(func=_probes)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

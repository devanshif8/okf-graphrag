"""Score retrieval accuracy against a ground-truth query set.

Runs every query in a `.jsonl` built by `make_queries.py` and reports Recall@k,
MRR and nDCG@k for three retrievers over the *same* vector call:

  vector  the Qdrant ranking alone
  graph   the :LINKS_TO neighbourhood of the top vector hits, alone
  hybrid  the two fused with weighted RRF - what `hybrid_search` returns

The three-way split is the point. A single hybrid number says the system
returns something; the comparison says whether the graph half — the linker, the
edges, the fusion, the degree damping, all of it — earns its place, or whether
plain vector search would do as well. That is the project's actual claim, so it
is the thing an accuracy test should be pointed at.

    python eval/evaluate.py eval/queries.jsonl
    python eval/evaluate.py eval/queries.jsonl --sweep 0 0.25 0.5 1.0 2.0

One caveat to read the `graph` row with: it is not an independent retriever.
`graph_expand` needs seed ids, and the seeds are the top vector hits, so the
graph row is 'the neighbourhood of what vector search found' rather than 'the
graph on its own'. It can only score above zero on a query whose answer is
adjacent to a vector hit — which is exactly the contribution fusion is there to
harvest, but it is not a like-for-like baseline and should not be read as one.

A second caveat, for the `hybrid` row specifically: on a known-item query the
one correct document is usually the *seed itself*, and `graph_expand` excludes
each seed from its own neighbourhood. So the graph list can rarely rank the
answer it already has, and fusion has little room to improve on vector search
here. Known-item measures whether the corpus is retrievable at all; judging the
graph's contribution properly needs queries whose answer is a document the
vector list missed.

Query sets with a large relevant set — every document in a domain, as built by
`make_domain_queries.py` — are read through Precision@k instead: recall over
hundreds of relevant documents is near zero at any cutoff a person reads. Such a
query may also carry `exclude` (ids dropped from every ranking before scoring)
and `base_rate` (the precision a random ranking would score), and `--group-by`
breaks the tables down by any field, e.g. `domain`.
"""

import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable  # noqa: E402
from qdrant_client.http.exceptions import (  # noqa: E402
    ResponseHandlingException,
    UnexpectedResponse,
)

from qdrant_loader import QDRANT_URL  # noqa: E402
from neo4j_loader import NEO4J_URI, NEO4J_USER  # noqa: E402
from retrieval import (  # noqa: E402
    GRAPH_SEEDS,
    MIN_CANDIDATES,
    VECTOR_WEIGHT,
    GRAPH_WEIGHT,
    DEGREE_DAMPING,
    close_clients,
    graph_expand,
    reciprocal_rank_fusion,
    vector_search,
)

# Cut-offs reported for Recall and nDCG. 1 is 'did it put the answer first',
# 10 is a screenful, 5 sits between them.
CUTOFFS = (1, 5, 10)

# Bootstrap resamples behind every confidence interval. Two thousand is enough
# for a 95% interval to be stable to the third decimal, which is the precision
# the tables print.
RESAMPLES = 2000

CONFIDENCE = 0.95

# How deep to retrieve before scoring. Ranks past this count as a miss, so it
# bounds MRR from below: with depth 20 an answer at rank 25 scores 0 rather
# than 0.04, which is the right call - nobody reads to rank 25.
DEPTH = 20

# Most extra results fetched to make up for a query's `exclude` set. Excluded
# ids tend to crowd the top of the ranking (a passage's own book), so without
# over-fetching the cut-off would be scored over fewer than k survivors. Capped
# because a 350-chapter book would otherwise turn one query into a scan.
MAX_EXCLUDE_OVERFETCH = 100


def reciprocal_rank(ranked: list[str], relevant: set) -> float:
    """1/rank of the first relevant id, or 0 if none was retrieved."""
    for position, identifier in enumerate(ranked, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def recall_at(ranked: list[str], relevant: set, k: int) -> float:
    """Fraction of the relevant ids that appear in the top k."""
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at(ranked: list[str], relevant: set, k: int) -> float:
    """Fraction of the top k that is relevant.

    Divides by k even when fewer than k results came back: a slot the retriever
    could not fill is a slot the user did not get a useful document in.
    """
    return len(relevant & set(ranked[:k])) / k


def ndcg_at(ranked: list[str], relevant: set, k: int) -> float:
    """Normalized discounted cumulative gain over binary relevance."""
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, identifier in enumerate(ranked[:k], start=1)
        if identifier in relevant
    )
    ideal = sum(
        1.0 / math.log2(position + 1)
        for position in range(1, min(len(relevant), k) + 1)
    )
    return gain / ideal if ideal else 0.0


def bootstrap_ci(
    values: list[float],
    resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> tuple:
    """A percentile bootstrap interval for the mean of `values`.

    Resamples the *queries*, not the documents, because the query set is the
    sample: run the same system on another 40 needs and the mean moves, and the
    interval says by how much. Nothing here assumes the scores are normally
    distributed, which matters when a metric is mostly zeros and ones.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    count = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(rng.choice(values) for _ in range(count)) / count)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low = means[max(0, int(tail * resamples) - 1)]
    high = means[min(resamples - 1, int((1.0 - tail) * resamples))]
    return (low, high)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test on the queries where the two runs differed.

    Ties carry no information about direction and are excluded, which is the
    standard treatment and also the conservative one here: most queries tie, so
    counting them would manufacture agreement the comparison has not earned.
    Returns the probability of a split this lopsided from a fair coin.
    """
    trials = wins + losses
    if not trials:
        return 1.0
    fewer = min(wins, losses)
    tail = sum(math.comb(trials, i) for i in range(fewer + 1)) / (2 ** trials)
    return min(1.0, 2.0 * tail)


def compare_runs(
    treatment: list[float],
    baseline: list[float],
    resamples: int = RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> dict:
    """Paired comparison of two runs over the same queries.

    The bootstrap is paired — each resample draws whole queries and keeps both
    runs' scores for them — because the two runs saw identical queries and the
    question is whether the *difference* is real. Resampling them independently
    would throw that pairing away and widen the interval for no reason.
    """
    pairs = list(zip(treatment, baseline))
    wins = sum(1 for a, b in pairs if a > b + 1e-12)
    losses = sum(1 for a, b in pairs if a < b - 1e-12)
    differences = [a - b for a, b in pairs]
    low, high = bootstrap_ci(differences, resamples, confidence, seed)
    return {
        "difference": sum(differences) / len(differences) if differences else 0.0,
        "ci": [low, high],
        "p_value": sign_test(wins, losses),
        "wins": wins,
        "losses": losses,
        "ties": len(pairs) - wins - losses,
    }


def score_ranking(ranked: list[str], relevant: set) -> dict:
    """Every metric for one ranked list against one query's relevant set."""
    scores = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in CUTOFFS:
        scores[f"recall@{k}"] = recall_at(ranked, relevant, k)
        scores[f"precision@{k}"] = precision_at(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at(ranked, relevant, k)
    return scores


def run_query(
    query: str,
    depth: int,
    weights: list[tuple[str, float, float]],
    degree_damping: float,
    exclude_index: bool,
    exclude: frozenset = frozenset(),
) -> dict:
    """Retrieve one query every configured way, returning name -> ranked ids.

    The vector search and the graph walk run once and are shared across every
    fusion weighting, because the weights only change how the two existing
    lists are combined. Re-searching per weighting would multiply the cost of a
    sweep by the number of weights for identical results.

    `exclude` is applied to each finished ranking, not to the inputs: the
    retrievers and the fusion run exactly as a user's query would, and the
    excluded ids are only hidden from the scorer.
    """
    fetch = max(depth, MIN_CANDIDATES) + min(len(exclude), MAX_EXCLUDE_OVERFETCH)
    vector_ids = [hit["id"] for hit in vector_search(query, top_k=fetch)]
    if not vector_ids:
        return {name: [] for name, _, _ in weights} | {"vector": [], "graph": []}

    neighbours = graph_expand(
        vector_ids[:GRAPH_SEEDS],
        exclude_index=exclude_index,
        degree_damping=degree_damping,
    )
    graph_ids = [neighbour["id"] for neighbour in neighbours]

    def visible(ids: list[str]) -> list[str]:
        return [identifier for identifier in ids if identifier not in exclude][:depth]

    rankings = {"vector": visible(vector_ids), "graph": visible(graph_ids)}
    for name, vector_weight, graph_weight in weights:
        fused = reciprocal_rank_fusion(
            {"vector": vector_ids, "graph": graph_ids},
            weights={"vector": vector_weight, "graph": graph_weight},
        )
        rankings[name] = visible([result["id"] for result in fused])
    return rankings


def evaluate(
    queries: list[dict],
    depth: int = DEPTH,
    weights: list[tuple[str, float, float]] | None = None,
    degree_damping: float = DEGREE_DAMPING,
    exclude_index: bool = True,
    progress_every: int = 50,
    group_by: str = "region",
    metric: str | None = None,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> dict:
    """Score every query and aggregate, overall and per `group_by` value.

    Returns {"runs": {name: {metric: mean}}, "regions": {...}, "paired": {...},
    "queries": n}. Means are macro-averaged over queries, so every query counts
    the same regardless of how many documents are relevant to it. `regions` is
    keyed by the query's `group_by` field; the name predates the option.

    Queries carrying `base_rate` add a `chance` entry — the mean precision a
    random ranking would score — overall and per group, so a precision can be
    read against what the corpus's own class balance gives away for free.
    """
    weights = weights or [("hybrid", VECTOR_WEIGHT, GRAPH_WEIGHT)]
    names = ["vector", "graph"] + [name for name, _, _ in weights]

    totals: dict = {name: {} for name in names}
    regions: dict = {}
    per_query: dict = {name: [] for name in names}
    chance_total = 0.0
    started = time.time()

    for number, entry in enumerate(queries, start=1):
        relevant = set(entry["relevant"])
        rankings = run_query(
            entry["query"],
            depth,
            weights,
            degree_damping,
            exclude_index,
            frozenset(entry.get("exclude", ())),
        )
        region = str(entry.get(group_by, "any"))
        bucket = regions.setdefault(region, {"count": 0, "runs": {}, "chance": 0.0})
        bucket["count"] += 1
        bucket["chance"] += entry.get("base_rate", 0.0)
        chance_total += entry.get("base_rate", 0.0)

        for name in names:
            scores = score_ranking(rankings.get(name, []), relevant)
            per_query[name].append(scores)
            region_totals = bucket["runs"].setdefault(name, {})
            # Not `metric`: that is this function's parameter, and rebinding it
            # here would silently decide the headline comparison for us.
            for key, value in scores.items():
                totals[name][key] = totals[name].get(key, 0.0) + value
                region_totals[key] = region_totals.get(key, 0.0) + value

        if progress_every and number % progress_every == 0:
            rate = number / (time.time() - started)
            print(
                f"  {number}/{len(queries)} queries ({rate:.1f}/s)",
                file=sys.stderr,
                flush=True,
            )

    count = len(queries)
    runs = {
        name: {metric: total / count for metric, total in metrics.items()}
        for name, metrics in totals.items()
    }

    region_runs = {
        region: {
            "count": data["count"],
            "chance": data["chance"] / data["count"],
            "runs": {
                name: {
                    metric: total / data["count"] for metric, total in metrics.items()
                }
                for name, metrics in data["runs"].items()
            },
        }
        for region, data in regions.items()
    }

    # The metric the comparison is decided on. A query set carrying base rates
    # is a precision set — its relevant sets are far too large for recall to
    # mean anything — so the deepest precision cutoff is the headline; anything
    # else is a known-item set, where MRR is.
    headline = metric or (f"precision@{CUTOFFS[-1]}" if chance_total else "mrr")
    scores_for = {
        name: [query[headline] for query in scored] for name, scored in per_query.items()
    }

    # Paired comparison of the first fusion against plain vector search: how
    # often fusion actually moved the answer, in which direction, and whether a
    # difference that size survives the query set being a sample of 40-odd
    # needs rather than every question anyone might ask.
    primary = weights[0][0]
    paired = {"against": "vector", "run": primary, "metric": headline} | compare_runs(
        scores_for[primary], scores_for["vector"], resamples, seed=seed
    )

    return {
        "queries": count,
        "depth": depth,
        "chance": chance_total / count,
        "metric": headline,
        "runs": runs,
        "intervals": {
            name: list(bootstrap_ci(values, resamples, seed=seed))
            for name, values in scores_for.items()
        },
        "regions": region_runs,
        "paired": paired,
        "per_query": [
            {"query": entry["query"]}
            | {name: scores_for[name][number] for name in names}
            for number, entry in enumerate(queries)
        ],
    }


def _table(runs: dict, chance: float = 0.0) -> str:
    """Render a run -> metric mapping as a fixed-width table.

    A nonzero `chance` adds a row holding the random-ranking precision under
    each precision column, and nothing under the metrics it does not apply to.
    """
    metrics = (
        ["mrr"]
        + [f"recall@{k}" for k in CUTOFFS]
        + [f"precision@{k}" for k in CUTOFFS]
        + [f"ndcg@{k}" for k in CUTOFFS]
    )
    width = max(len(name) for name in runs) + 2
    header = "run".ljust(width) + "".join(metric.rjust(13) for metric in metrics)
    lines = [header, "-" * len(header)]
    for name, scores in runs.items():
        row = name.ljust(width)
        row += "".join(f"{scores.get(metric, 0.0):13.3f}" for metric in metrics)
        lines.append(row)
    if chance:
        row = "chance".ljust(width)
        row += "".join(
            f"{chance:13.3f}" if metric.startswith("precision") else "-".rjust(13)
            for metric in metrics
        )
        lines.append(row)
    return "\n".join(lines)


def _intervals(results: dict) -> str:
    """The headline metric with its confidence interval, and the paired verdict.

    Printed under the table rather than inside it: an interval per cell would be
    unreadable, and only one metric is ever the one the comparison turns on.
    """
    metric = results["metric"]
    paired = results["paired"]
    # Wide enough for the widest row label, not just the run names, so the
    # numbers stay in one column.
    labels = list(results["intervals"]) + [
        f"{paired['run']} - {paired['against']}", "sign test"
    ]
    width = max(len(label) for label in labels) + 2
    percent = int(CONFIDENCE * 100)

    lines = [
        f"{metric}, with {percent}% confidence intervals "
        f"({results['queries']} queries, {RESAMPLES} bootstrap resamples)",
        "",
    ]
    for name, (low, high) in results["intervals"].items():
        mean = results["runs"][name][metric]
        lines.append(f"  {name.ljust(width)}{mean:7.3f}   [{low:.3f}, {high:.3f}]")

    low, high = paired["ci"]
    lines += [
        "",
        f"  {(paired['run'] + ' - ' + paired['against']).ljust(width)}"
        f"{paired['difference']:+7.3f}   [{low:+.3f}, {high:+.3f}]",
        f"  {'sign test'.ljust(width)}p = {paired['p_value']:.3f}   "
        f"({paired['wins']} better, {paired['losses']} worse, {paired['ties']} tied)",
    ]

    # The interval is the finding when it straddles zero, so say so rather than
    # leaving a reader to infer it from two numbers with opposite signs.
    if low <= 0.0 <= high:
        lines.append(
            f"\n  The interval includes zero: this query set cannot show "
            f"{paired['run']} and {paired['against']} apart."
        )
    return "\n".join(lines)


def load_queries(path: str, require_relevant: bool = True) -> list[dict]:
    """Read a .jsonl query set, failing loudly on a malformed line.

    `require_relevant` is relaxed only by the probe set, whose needs have no
    answer anywhere in the library and so carry an empty relevant set on
    purpose. Scoring one for precision is meaningless, which is why it stays an
    error here by default rather than passing quietly into a results table.
    """
    queries = []
    with open(path, encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: not valid JSON ({exc})") from None
            if not entry.get("query"):
                raise SystemExit(f"{path}:{number}: needs 'query'")
            if require_relevant and not entry.get("relevant"):
                raise SystemExit(f"{path}:{number}: needs a non-empty 'relevant'")
            entry.setdefault("relevant", [])
            queries.append(entry)
    return queries


def main() -> None:
    """Score a query set against the loaded bundle and print the comparison."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("queries", help="ground-truth .jsonl from make_queries.py")
    ap.add_argument(
        "--depth", type=int, default=DEPTH,
        help=f"retrieval depth scored (default: {DEPTH})",
    )
    ap.add_argument(
        "--vector-weight", type=float, default=VECTOR_WEIGHT,
        help=f"vector weight in the fusion (default: {VECTOR_WEIGHT})",
    )
    ap.add_argument(
        "--graph-weight", type=float, default=GRAPH_WEIGHT,
        help=f"graph weight in the fusion (default: {GRAPH_WEIGHT})",
    )
    ap.add_argument(
        "--sweep", type=float, nargs="+", metavar="W",
        help="score one fusion per graph weight, holding the vector weight fixed",
    )
    ap.add_argument(
        "--degree-damping", type=float, default=DEGREE_DAMPING,
        help=f"degree damping in the graph walk (default: {DEGREE_DAMPING})",
    )
    ap.add_argument(
        "--include-index", action="store_true",
        help="keep index.md hub nodes in the graph list",
    )
    ap.add_argument(
        "--group-by", default="region",
        help="query field to break the results down by (default: region)",
    )
    ap.add_argument(
        "--metric",
        help="metric the comparison turns on (default: precision at the deepest "
             "cutoff for a precision set, mrr otherwise)",
    )
    ap.add_argument(
        "--resamples", type=int, default=RESAMPLES,
        help=f"bootstrap resamples behind each interval (default: {RESAMPLES})",
    )
    ap.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed (default: 0)")
    ap.add_argument("--limit", type=int, help="score only the first N queries")
    ap.add_argument("--output", help="also write the full results as JSON here")
    args = ap.parse_args()

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit(f"no queries in {args.queries}")

    if args.sweep:
        weights = [
            (f"hybrid g={weight:g}", args.vector_weight, weight)
            for weight in args.sweep
        ]
    else:
        weights = [("hybrid", args.vector_weight, args.graph_weight)]

    print(f"scoring {len(queries)} queries at depth {args.depth}...", file=sys.stderr)
    try:
        results = evaluate(
            queries,
            depth=args.depth,
            weights=weights,
            degree_damping=args.degree_damping,
            exclude_index=not args.include_index,
            group_by=args.group_by,
            metric=args.metric,
            resamples=args.resamples,
            seed=args.seed,
        )
    except LookupError as exc:
        raise SystemExit(str(exc)) from None
    except (ResponseHandlingException, ConnectionError):
        raise SystemExit(f"could not reach Qdrant at {QDRANT_URL} - is it running?") from None
    except UnexpectedResponse as exc:
        raise SystemExit(f"Qdrant rejected the query: {exc}") from None
    except ServiceUnavailable:
        raise SystemExit(
            f"could not reach Neo4j at {NEO4J_URI} - is the database running?"
        ) from None
    except AuthError:
        raise SystemExit(
            f"Neo4j rejected the credentials for user '{NEO4J_USER}'"
        ) from None
    except ConfigurationError as exc:
        raise SystemExit(f"bad Neo4j configuration (NEO4J_URI={NEO4J_URI}): {exc}") from None
    finally:
        close_clients()

    print(f"\n{results['queries']} queries, depth {results['depth']}\n")
    print(_table(results["runs"], results["chance"]))
    print()
    print(_intervals(results))

    if len(results["regions"]) > 1:
        for region, data in sorted(results["regions"].items()):
            print(f"\n{args.group_by} '{region}' ({data['count']} queries)\n")
            print(_table(data["runs"], data["chance"]))

    paired = results["paired"]
    print(
        f"\n{paired['run']} vs {paired['against']}, per query: "
        f"{paired['wins']} better, {paired['losses']} worse, {paired['ties']} unchanged"
    )

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"full results written to {args.output}")


if __name__ == "__main__":
    main()

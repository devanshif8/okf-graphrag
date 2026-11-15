"""Generate a known-item ground-truth query set from a loaded bundle.

Accuracy is only a number against ground truth: a set of queries paired with the
documents that should come back. Hand-labelling that is slow, so this builds it
the standard cheap way — sample a passage out of a document and use it as the
query, with that document as the one correct answer. It is a weaker signal than
human relevance judgements (a real user's phrasing is not lifted from the text),
but it is unbiased with respect to the retriever, needos no labelling, and scales
to as many queries as the corpus has documents.

    python eval/make_queries.py path/to/bundle --output eval/queries.jsonl

Each line is `{"query": ..., "relevant": ["<concept id>"], "region": ...}`.

The `region` field exists because of a specific limitation worth measuring
rather than hiding. `qdrant_loader` embeds one vector per file and the model
truncates at 256 word-pieces, so a passage from late in a long document was
never embedded at all and *cannot* be retrieved by vector search. Sampling only
from the head would flatter the system; sampling only from the tail would
condemn it. Sampling both and reporting the split shows how much of any error is
truncation rather than ranking — see `--region` and the per-region breakdown in
`evaluate.py`.
"""

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linker import strip_related  # noqa: E402
from neo4j_loader import concept_id  # noqa: E402
from parser import parse_bundle  # noqa: E402

# Words in a sampled query. Long enough to name something specific, short enough
# to resemble what a person would type.
PASSAGE_WORDS = 25

# Where the embedding window ends, in words. The model's limit is 256
# word-pieces; English averages around 1.3 word-pieces per word, so ~180 words
# is the practical boundary. Deliberately conservative on both sides: a passage
# is only 'head' if it is comfortably inside, only 'tail' if comfortably past.
HEAD_LIMIT = 150
TAIL_START = 250

# A passage that is mostly punctuation, numerals or single letters is a table or
# a page-number artefact, not a sentence anyone would search for.
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")
MIN_ALPHA_RATIO = 0.7


def _usable(words: list[str]) -> bool:
    """Whether a sampled passage reads as prose rather than an artefact."""
    if len(words) < PASSAGE_WORDS:
        return False
    alpha = sum(1 for word in words if WORD_RE.match(word))
    return alpha / len(words) >= MIN_ALPHA_RATIO


def sample_passages(
    body: str,
    count: int,
    region: str,
    rng: random.Random,
    passage_words: int = PASSAGE_WORDS,
) -> list[tuple[str, str]]:
    """Draw passages from one body as (passage, region) pairs.

    `region` selects where in the document to sample from: 'head' inside the
    embedding window, 'tail' past it, 'both' for an even split, 'any' for
    anywhere. A document too short to have a tail simply yields fewer passages
    rather than being padded with head samples labelled as tail.
    """
    words = strip_related(body).split()

    spans = {
        "head": (0, min(HEAD_LIMIT, len(words)) - passage_words),
        "tail": (TAIL_START, len(words) - passage_words),
        "any": (0, len(words) - passage_words),
    }

    if region == "both":
        wanted = ["head"] * ((count + 1) // 2) + ["tail"] * (count // 2)
    else:
        wanted = [region] * count

    passages = []
    for name in wanted:
        low, high = spans[name]
        if high < low:
            continue  # document too short for this region
        start = rng.randint(low, high)
        chunk = words[start : start + passage_words]
        if _usable(chunk):
            passages.append((" ".join(chunk), name))
    return passages


def build_queries(
    bundle_dir: str,
    per_document: int = 1,
    region: str = "both",
    seed: int = 0,
    limit: int | None = None,
) -> list[dict]:
    """Build the known-item query set for a bundle.

    Index files are excluded: they are structural hubs with no content of their
    own, so a passage sampled from one is not a question about a document.
    """
    rng = random.Random(seed)
    concepts = [c for c in parse_bundle(bundle_dir) if not c.get("is_index")]
    if limit:
        # Sampled, not truncated: the first N files alphabetically are usually
        # one or two books, which is not a sample of the corpus.
        concepts = rng.sample(concepts, min(limit, len(concepts)))

    queries = []
    for concept in sorted(concepts, key=lambda c: c["path"]):
        identifier = concept_id(concept["path"], bundle_dir)
        for passage, name in sample_passages(
            concept["body"], per_document, region, rng
        ):
            queries.append(
                {"query": passage, "relevant": [identifier], "region": name}
            )
    return queries


def main() -> None:
    """Write a ground-truth query set for a bundle and report the split."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="path to the converted, linked bundle")
    ap.add_argument("--output", required=True, help="where to write the .jsonl")
    ap.add_argument(
        "--per-document", type=int, default=1, help="queries per document (default: 1)"
    )
    ap.add_argument(
        "--region", choices=["head", "tail", "both", "any"], default="both",
        help="where in each document to sample from (default: both)",
    )
    ap.add_argument(
        "--limit", type=int, help="sample this many documents instead of all of them"
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = ap.parse_args()

    if not os.path.isdir(args.bundle):
        raise SystemExit(f"not a directory: {args.bundle}")

    queries = build_queries(
        args.bundle,
        per_document=args.per_document,
        region=args.region,
        seed=args.seed,
        limit=args.limit,
    )
    if not queries:
        raise SystemExit(
            f"no usable passages found in {args.bundle} - are the documents too short?"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for entry in queries:
            fh.write(json.dumps(entry) + "\n")

    regions = {}
    for entry in queries:
        regions[entry["region"]] = regions.get(entry["region"], 0) + 1
    breakdown = ", ".join(f"{count} {name}" for name, count in sorted(regions.items()))
    print(f"{len(queries)} queries ({breakdown}) written to {args.output}")


if __name__ == "__main__":
    main()

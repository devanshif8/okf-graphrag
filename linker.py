"""Derive `:LINKS_TO` structure for converted documents by shared keyphrases.

`convert_to_okf.py` turns a document into an OKF file, but a PDF has no notion
of an OKF concept path, so a converted bundle has no markdown links and
therefore no graph. Loaded as-is it gives Neo4j a pile of isolated nodes,
`graph_expand` finds nothing to walk, and `hybrid_search` silently degrades
into plain vector search — the fusion, the degree damping and the hub handling
all idle. This module supplies the missing edges.

The method is the KeyBERT one, reusing the embedding model the bundle is
already indexed with: enumerate noun-ish n-grams as candidate phrases, embed
them, and keep the ones closest to the document's own embedding, with MMR to
stop the top of the list being five rewordings of one idea. Two documents are
then linked when they share a keyphrase that is *discriminative* — one held by
at least two documents but not by most of them, since a phrase every document
carries separates nothing.

Why not link on document similarity directly: the vector list already ranks by
that, and an edge set derived from the same embedding distances would make the
graph ranking a restatement of the vector ranking, leaving RRF to fuse a list
with a copy of itself. Shared *phrases* are a different signal — two documents
can share one specific rare phrase while sitting far apart overall — so the two
ranked lists stay independent and the fusion has something to do.

Edges are written into the files as ordinary markdown links, under a `## Related`
heading, with the shared phrase as the link text:

    ## Related

    - [security training](security-training.md)

That keeps everything downstream untouched — `parser.py` already extracts links
with `LINK_RE`, `neo4j_loader.py` already resolves them into `:LINKS_TO` edges,
and the file stays readable, with each edge stating the reason it exists.
"""

import argparse
import heapq
import math
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from convert_to_okf import render_okf
from parser import parse_bundle
from qdrant_loader import MODEL_NAME, chunk_text

# Keyphrases per document. Ten is enough to cover a document's distinct topics
# without reaching so far down the ranking that noise starts matching noise.
TOP_N = 10

# Most related documents to link from any one document. Without a cap, a corpus
# on a single subject links every document to every other and the graph becomes
# a clique, which carries no more information than having no edges at all.
MAX_LINKS = 5

# A phrase held by more than this fraction of the bundle is treated as a theme
# rather than a topic and never creates an edge: in a corpus of HR documents,
# "the employee" is in all of them and separates none of them.
MAX_DF_RATIO = 0.5

# A hard ceiling on document frequency, applied alongside the ratio.
#
# The ratio alone is a proportion, so it grows with the corpus: at fifty
# thousand documents it still admits a phrase held by twenty-five thousand of
# them, and enumerating that phrase's pairs costs df² — 625 million for one
# phrase. A constant ceiling is what makes the whole pass affordable, and the
# arithmetic says how large it can be. With `top_n` phrases per document there
# are 10N phrase-holdings in total, so at a uniform df there are 10N/df phrases
# each costing df² pairs: 10·N·df overall. Cost is therefore *linear* in the
# corpus and in this ceiling, not quadratic — so the ceiling can be generous.
#
# 200 rather than something tighter because the cap is not free. A corpus on one
# subject — the case this project's linker is most often pointed at — has a small
# effective vocabulary and correspondingly high df, and a low ceiling silently
# deletes most of its edges: measured on 2000 documents sharing 200 phrases
# (average df 60), a ceiling of 50 yields 4590 edges where 200 yields 10000, and
# the difference is not noise being filtered but signal being dropped. 200 costs
# about two seconds there and keeps every edge.
MAX_DF_ABSOLUTE = 200

# MMR trade-off: 0 ranks purely by similarity to the document, 1 purely by
# difference from what's already selected. 0.3 keeps the list relevant while
# breaking up near-duplicate phrasings.
DIVERSITY = 0.3

# Candidate n-gram sizes. Single words are too ambiguous to link on alone, but they
# are kept because acronyms and product names are genuinely unigrams; 3 is the
# ceiling because longer spans almost never recur verbatim across documents.
NGRAM_RANGE = (1, 3)

MIN_PHRASE_CHARS = 3

# Candidates that reach the MMR pairwise matrix. See `_mmr`: the matrix is
# quadratic in this number and nothing below the top of the relevance ranking is
# ever selected, so capping it removes cost rather than quality.
MMR_POOL = 200

# Fewest documents a candidate phrase must appear in to be worth embedding.
#
# This is the corpus-scale optimisation. Embedding every candidate n-gram of
# every document is the most expensive thing the linker does — roughly a
# thousand short strings per document, so ten million for a ten thousand
# document corpus. But an edge requires a *shared* phrase, so a phrase that
# occurs in exactly one document cannot produce one however good it is, and in
# real prose most n-grams occur exactly once. Counting document frequency over
# the raw candidate strings costs no model at all, and discarding the singletons
# before embedding removes the large majority of the work.
#
# Only the lower bound is safe to apply here. The upper bound in `derive_links`
# is over *selected* keyphrases, which is a subset of candidates — a phrase can
# be a candidate everywhere and selected in three documents, where it still
# forms a legitimate edge — so filtering candidates by an upper bound would
# delete edges the original would have drawn. The lower bound has no such
# asymmetry: selected implies candidate, so selected in two documents implies a
# candidate document frequency of at least two.
MIN_CANDIDATE_DF = 2

# Texts per `encode` call. The model batches internally; this bounds the size of
# the array it hands back, so peak memory does not scale with the corpus.
ENCODE_BATCH = 512

# Section appended to each file. Rewritten in place on every run, so re-linking
# a bundle updates the edges instead of stacking a second copy of them.
RELATED_HEADING = "## Related"
RELATED_RE = re.compile(
    rf"\n*{re.escape(RELATED_HEADING)}\n.*?(?=\n## |\Z)", re.DOTALL
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")

# Function words only — no domain nouns, so the list stays neutral across
# corpora. Modals and auxiliaries earn their place because they are what glue
# verb fragments like 'must complete security' into candidate position.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because
    been before being below between both but by can cannot could did do does doing
    done down during each few for from further had has have having he her here hers
    him his how i if in into is it its itself just may me might more most must my
    no nor not now of off on once only or other our ours out over own same shall
    she should so some such than that the their theirs them then there these they
    this those through to too under until up upon very was we were what when where
    which while who whom why will with within would you your yours
    """.split()
)


def _sentences(text: str) -> list[str]:
    """Split a body into spans that a phrase may not cross.

    Candidates are built inside these spans only, so a phrase can't be stitched
    together across a sentence boundary, a line break or a heading.
    """
    without_headings = re.sub(r"^#{1,6} .*$", "\n", text, flags=re.MULTILINE)
    return re.split(r"[.!?;:,()\[\]{}\n]+", without_headings)


def candidate_phrases(body: str, ngram_range: tuple = NGRAM_RANGE) -> list[str]:
    """Enumerate deduplicated candidate keyphrases from a document body.

    A candidate is an n-gram containing no function word at all — the cheap
    approximation of a noun phrase that avoids a POS-tagger dependency. Testing
    only the first and last word is not enough: it admits verb fragments like
    'training covers password', which recur across documents far too rarely to
    link on and which crowd real topics out of the ranking.
    """
    low, high = ngram_range
    seen = {}
    for sentence in _sentences(body):
        words = [w.lower() for w in WORD_RE.findall(sentence)]
        for size in range(low, high + 1):
            for start in range(len(words) - size + 1):
                gram = words[start : start + size]
                if any(word in STOPWORDS for word in gram):
                    continue
                if any(len(w) < 2 for w in gram) and size == 1:
                    continue
                phrase = " ".join(gram)
                if len(phrase) < MIN_PHRASE_CHARS or phrase.isdigit():
                    continue
                seen.setdefault(phrase, None)
    return list(seen)


def _mmr(doc_vector, candidate_vectors, phrases, top_n, diversity, pool=MMR_POOL):
    """Select top_n phrases balancing closeness to the document with variety.

    Only the `pool` most relevant candidates take part. The redundancy term
    needs every pairwise similarity between the candidates it considers, and
    that matrix is quadratic: a thousand candidates is four megabytes, a whole
    unchunked book is around a hundred and fifty thousand, which is ninety
    gigabytes and simply does not run. Since at most `top_n` phrases are ever
    selected, and they come off the top of the relevance ranking, the
    similarity between the 800th and 900th candidate is never consulted —
    computing it is pure waste rather than lost quality.
    """
    relevance = candidate_vectors @ doc_vector

    if len(phrases) > pool:
        # Stable sort so an exact tie in relevance keeps candidate order, and
        # the same document always yields the same phrases.
        keep = np.argsort(-relevance, kind="stable")[:pool]
        candidate_vectors = candidate_vectors[keep]
        phrases = [phrases[i] for i in keep]
        relevance = relevance[keep]

    between = candidate_vectors @ candidate_vectors.T

    selected = [int(np.argmax(relevance))]
    remaining = set(range(len(phrases))) - set(selected)

    while len(selected) < min(top_n, len(phrases)) and remaining:
        index = list(remaining)
        redundancy = between[np.ix_(index, selected)].max(axis=1)
        score = (1 - diversity) * relevance[index] - diversity * redundancy
        selected.append(index[int(np.argmax(score))])
        remaining.discard(selected[-1])

    return [phrases[i] for i in selected]


def strip_related(body: str) -> str:
    """Return a body without the linker's own `## Related` section.

    Everything that reads a body has to go through this. The section is written
    back into the file, so on a second run it would otherwise be prose like any
    other: the link text would be re-extracted as a keyphrase and the edges from
    one run would decide the edges of the next, drifting a little further each
    time. Stripping it first makes linking a function of the source text alone.
    """
    return RELATED_RE.sub("", body).strip()


def _drop_contained(phrases: list[str]) -> list[str]:
    """Drop a phrase already contained in a better-ranked one.

    'training' adds nothing once 'security training' has been selected, and
    keeping both would let one topic occupy two of the ten slots.
    """
    kept: list[str] = []
    for phrase in phrases:
        if not any(phrase in better for better in kept):
            kept.append(phrase)
    return kept


def _encode(model: SentenceTransformer, texts: list[str]) -> list:
    """Embed texts in bounded batches, normalized, as a list of vectors.

    normalize_embeddings makes the dot product a cosine, which is what both the
    MMR relevance and the redundancy terms assume.
    """
    vectors = []
    for start in range(0, len(texts), ENCODE_BATCH):
        batch = texts[start : start + ENCODE_BATCH]
        vectors.extend(model.encode(batch, normalize_embeddings=True))
    return vectors


def document_vector(body: str, model: SentenceTransformer):
    """Embed a whole document rather than as much of it as the model will read.

    `model.encode(body)` truncates at 256 word-pieces — about 180 words — so on
    anything longer this vector described the opening paragraph and MMR ranked
    every keyphrase for relevance to page one. Averaging the document's chunk
    embeddings covers the whole text instead, reusing the same splitter the
    Qdrant loader indexes with so both stores mean the same thing by "this
    document".
    """
    chunks = chunk_text(body) or [body]
    vectors = np.atleast_2d(np.asarray(_encode(model, chunks)))
    mean = vectors.mean(axis=0)

    # The mean of unit vectors is not itself a unit vector, and the MMR
    # relevance term is a dot product standing in for a cosine.
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def _rank(doc_vector, phrases: list[str], vector_of: dict, top_n, diversity):
    """Rank one document's candidate phrases, given vectors already computed."""
    if not phrases:
        return []
    vectors = np.asarray([vector_of[phrase] for phrase in phrases])
    ranked = _mmr(doc_vector, vectors, phrases, top_n * 2, diversity)
    return _drop_contained(ranked)[:top_n]


def keyphrases(
    body: str,
    model: SentenceTransformer,
    top_n: int = TOP_N,
    diversity: float = DIVERSITY,
) -> list[str]:
    """Rank a single document's most characteristic phrases, best first.

    Embeds this document's candidates on their own. `bundle_keyphrases` is the
    path that scales — it shares one vocabulary across the corpus and discards
    phrases that cannot form an edge — but a single document has no corpus to
    compare against, so nothing here can be skipped.
    """
    body = strip_related(body)
    candidates = candidate_phrases(body)
    if not candidates:
        return []

    vector_of = dict(zip(candidates, _encode(model, candidates)))
    return _rank(
        document_vector(body, model), candidates, vector_of, top_n, diversity
    )


def candidate_document_frequency(candidates_by_path: dict) -> dict:
    """How many documents each candidate phrase appears in."""
    frequency: dict = {}
    for candidates in candidates_by_path.values():
        for phrase in set(candidates):
            frequency[phrase] = frequency.get(phrase, 0) + 1
    return frequency


def bundle_keyphrases(
    concepts: list[dict],
    model: SentenceTransformer,
    top_n: int = TOP_N,
    diversity: float = DIVERSITY,
    min_df: int = MIN_CANDIDATE_DF,
) -> dict:
    """Map each non-index concept's file path to its ranked keyphrases.

    Two passes. The first enumerates every document's candidate phrases with no
    model at all and counts how many documents each appears in; candidates below
    `min_df` are dropped, because a phrase no other document contains cannot
    produce an edge and embedding it is the largest avoidable cost here. The
    second embeds what survives — once per *distinct* phrase across the whole
    corpus, not once per document that holds it — and ranks each document
    against that shared vocabulary.

    Both halves matter. On a real corpus most n-grams occur exactly once, so the
    filter removes the large majority of candidates; of what remains, a phrase
    held by forty documents was previously embedded forty times.

    This does change which phrases are selected, and deliberately: with
    singletons gone, more of each document's ten slots go to phrases that can
    actually link something, which is what the keyphrases are for. A document's
    unique-in-the-corpus phrase was never able to earn an edge with it.
    """
    bodies = {
        concept["path"]: strip_related(concept["body"])
        for concept in concepts
        if not concept.get("is_index")
    }

    candidates_by_path = {
        path: candidate_phrases(body) for path, body in bodies.items()
    }
    frequency = candidate_document_frequency(candidates_by_path)

    eligible = {
        path: [phrase for phrase in candidates if frequency[phrase] >= min_df]
        for path, candidates in candidates_by_path.items()
    }

    # Sorted so the vocabulary — and therefore every batch boundary in the
    # encoder — is identical between runs over the same corpus.
    vocabulary = sorted({phrase for phrases in eligible.values() for phrase in phrases})
    vector_of = dict(zip(vocabulary, _encode(model, vocabulary)))

    return {
        path: _rank(
            document_vector(bodies[path], model),
            eligible[path],
            vector_of,
            top_n,
            diversity,
        )
        for path in bodies
    }


# Decimal places a pair's score is rounded to before partners are ordered.
#
# Two partners sharing phrases of equal weight have mathematically equal scores,
# but the sums are accumulated in whatever order the phrase dictionary happens
# to be iterated in, and floating-point addition is not associative: the same
# tie can come out one unit-in-the-last-place apart, which silently decides the
# ordering. Rounding first collapses that noise so a real tie falls through to
# the path, which is stable. Twelve places is far finer than any score
# difference the log weights can produce and far coarser than the noise.
SCORE_PRECISION = 12


def _partner_order(partner: tuple) -> tuple:
    """Sort key for one candidate edge: strongest first, ties on the target id."""
    score, _phrase, other = partner
    return (-round(score, SCORE_PRECISION), other)


def derive_links(
    keyphrase_map: dict,
    max_links: int = MAX_LINKS,
    max_df_ratio: float = MAX_DF_RATIO,
    max_df_absolute: int = MAX_DF_ABSOLUTE,
) -> dict:
    """Pair up documents that share discriminative keyphrases.

    Returns {path: [(phrase, target_path), ...]}, each document's list ordered
    by how strongly it relates to the target and capped at `max_links`. A pair
    is scored by summing log(N / df) over the phrases it shares, so agreeing on
    a phrase only two documents use counts for far more than agreeing on one
    half the bundle carries.

    The cost is the number of document pairs that actually share a phrase, not
    the number of pairs that exist. That distinction is the difference between
    running and not running: this used to walk every document against every
    other to collect each one's partners, which is fine for the fourteen files
    of a sample bundle and is a hundred million dictionary lookups at ten
    thousand. Co-occurrence is sparse — most documents share nothing with most
    others — so iterating what was found is smaller than N² by orders of
    magnitude, and never larger.
    """
    total = len(keyphrase_map)
    if total < 2:
        return {path: [] for path in keyphrase_map}

    holders: dict = {}
    for path, phrases in keyphrase_map.items():
        for phrase in phrases:
            holders.setdefault(phrase, set()).add(path)

    max_df = max(2, min(math.floor(max_df_ratio * total), max_df_absolute))

    # (path, other) -> [summed weight, best (weight, phrase) seen]. Accumulated
    # as it goes rather than kept as a list of every shared phrase: only the sum
    # and the strongest phrase are ever read, and at corpus scale the discarded
    # middle of that list is most of the memory the function would hold.
    shared: dict = {}
    for phrase, paths in holders.items():
        df = len(paths)
        if df < 2 or df > max_df:
            continue

        weight = math.log(total / df)
        ordered = sorted(paths)
        for index, path in enumerate(ordered):
            for other in ordered[index + 1 :]:
                # One entry per unordered pair, mirrored: the edge is
                # symmetric, so computing it twice only costs twice.
                for key in ((path, other), (other, path)):
                    entry = shared.get(key)
                    if entry is None:
                        shared[key] = [weight, (weight, phrase)]
                    else:
                        entry[0] += weight
                        entry[1] = max(entry[1], (weight, phrase))

    # Group by source in one pass over what was found. Documents that share
    # nothing never appear here, which is why they cost nothing.
    partners: dict = {}
    for (path, other), (score, best) in shared.items():
        # The strongest shared phrase names the edge; the total decides which
        # partners survive the cap.
        partners.setdefault(path, []).append((score, best[1], other))

    links: dict = {path: [] for path in keyphrase_map}
    for path, found in partners.items():
        # nsmallest on the sort key, rather than sorting the whole list: a hub
        # document in a large corpus can have thousands of partners and only
        # `max_links` of them are ever used.
        best = heapq.nsmallest(max_links, found, key=_partner_order)
        links[path] = [(phrase, other) for _, phrase, other in best]
    return links


def render_related(path: str, related: list) -> str:
    """Render the '## Related' section body for one document."""
    lines = [RELATED_HEADING, ""]
    for phrase, target in related:
        relative = os.path.relpath(target, os.path.dirname(path)).replace(os.sep, "/")
        lines.append(f"- [{phrase}]({relative})")
    return "\n".join(lines)


def apply_links(concept: dict, related: list, phrases: list) -> str:
    """Rewrite one concept file with its keyphrases and related links.

    Returns the file text written. Any previous `## Related` section is removed
    first, so running the linker twice replaces the edges rather than appending
    a second set of them.
    """
    # strip_related also strips: parse_okf_file hands back the newline that
    # follows the closing '---', and render_okf adds its own, so leaving it would
    # grow a blank line into the body every time the bundle is re-linked.
    body = strip_related(concept["body"])
    if related:
        body = f"{body}\n\n{render_related(concept['path'], related)}"

    frontmatter = dict(concept["frontmatter"])
    if phrases:
        frontmatter["keyphrases"] = phrases

    text = render_okf(frontmatter, body)
    with open(concept["path"], "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def link_bundle(
    bundle_dir: str,
    model: SentenceTransformer = None,
    top_n: int = TOP_N,
    max_links: int = MAX_LINKS,
    max_df_ratio: float = MAX_DF_RATIO,
    max_df_absolute: int = MAX_DF_ABSOLUTE,
    min_df: int = MIN_CANDIDATE_DF,
    dry_run: bool = False,
) -> dict:
    """Derive and write keyphrase links across a converted bundle.

    Returns {"concepts": n, "phrases": n, "edges": n, "links": {path: [...]}}.
    """
    concepts = [c for c in parse_bundle(bundle_dir) if not c.get("is_index")]
    if not concepts:
        return {"concepts": 0, "phrases": 0, "edges": 0, "links": {}}

    model = model or SentenceTransformer(MODEL_NAME)
    keyphrase_map = bundle_keyphrases(concepts, model, top_n=top_n, min_df=min_df)
    links = derive_links(
        keyphrase_map,
        max_links=max_links,
        max_df_ratio=max_df_ratio,
        max_df_absolute=max_df_absolute,
    )

    if not dry_run:
        for concept in concepts:
            apply_links(
                concept,
                links.get(concept["path"], []),
                keyphrase_map.get(concept["path"], []),
            )

    return {
        "concepts": len(concepts),
        "phrases": len({p for ps in keyphrase_map.values() for p in ps}),
        "edges": sum(len(v) for v in links.values()),
        "links": links,
    }


def main() -> None:
    """Link a converted bundle in place and report what was written."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="path to the converted bundle directory")
    ap.add_argument(
        "--top-n", type=int, default=TOP_N,
        help=f"keyphrases per document (default: {TOP_N})",
    )
    ap.add_argument(
        "--max-links", type=int, default=MAX_LINKS,
        help=f"most related documents to link from each (default: {MAX_LINKS})",
    )
    ap.add_argument(
        "--max-df-ratio", type=float, default=MAX_DF_RATIO,
        help=(
            "ignore phrases held by more than this fraction of the bundle "
            f"(default: {MAX_DF_RATIO})"
        ),
    )
    ap.add_argument(
        "--min-df", type=int, default=MIN_CANDIDATE_DF,
        help=(
            "fewest documents a candidate phrase must appear in to be embedded "
            f"(default: {MIN_CANDIDATE_DF}; 1 embeds every candidate, which is "
            "much slower and cannot add an edge)"
        ),
    )
    ap.add_argument(
        "--max-df-absolute", type=int, default=MAX_DF_ABSOLUTE,
        help=(
            "ignore phrases held by more than this many documents, whatever "
            f"the bundle size (default: {MAX_DF_ABSOLUTE})"
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="report the edges without writing them into the files",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.bundle):
        raise SystemExit(f"not a directory: {args.bundle}")

    result = link_bundle(
        args.bundle,
        top_n=args.top_n,
        max_links=args.max_links,
        max_df_ratio=args.max_df_ratio,
        max_df_absolute=args.max_df_absolute,
        min_df=args.min_df,
        dry_run=args.dry_run,
    )
    if not result["concepts"]:
        raise SystemExit(f"no concepts found in {args.bundle}")

    for path, related in sorted(result["links"].items()):
        if related:
            name = os.path.basename(path)
            for phrase, target in related:
                print(f"  {name} --[{phrase}]--> {os.path.basename(target)}")

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"{result['concepts']} concepts, {result['phrases']} keyphrases, "
        f"{verb} {result['edges']} links"
    )


if __name__ == "__main__":
    main()

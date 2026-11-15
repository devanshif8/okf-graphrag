"""How far to trust an answer: a support check per claim, and a retrieval score.

`answer.py` already makes the model cite passages and strips citations that
name nothing. That proves a claim *points* at a passage, not that the passage
*says* it. This module checks the second thing, with a model that is not the
one that wrote the answer: an NLI (natural language inference) cross-encoder
reads each cited passage next to the sentence citing it and scores how strongly
the passage entails the sentence.

An independent checker matters because asking the writer how sure it is
measures fluency, not faithfulness - a model that invented a claim is usually
just as confident about it. The checker never sees the question and cannot
add anything; it only compares two texts.

What the score is not: proof. NLI models miss paraphrases that lean on world
knowledge, and a passage can entail a claim the book itself has wrong. So the
output is "this claim is / is not backed by its source", which catches the
common hallucination - a fluent sentence the passages never said - and nothing
stronger is claimed for it.

The retrieval score is the other half: how close the best passage handed to
the model was to the question, by the same embedding similarity vector search
ranks on. A well-grounded answer built from weakly related passages is still
an answer to a slightly different question, and a low score says so.
"""

import re
import threading

# Small (~280MB on disk, fast on CPU) and trained on MNLI + SNLI. Its labels are
# read from the model config rather than assumed, because NLI checkpoints do
# not agree on the order of contradiction / entailment / neutral.
NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"

# A claim counts as supported when the passage entails it with at least this
# probability. The model is sharply bimodal in practice (true paraphrases score
# ~0.99, unrelated or invented claims ~0.00), so the exact cut matters little;
# 0.5 is simply "entailment is the likeliest reading".
SUPPORTED = 0.5

# Retrieval score bands, on the cosine similarity of all-MiniLM-L6-v2 between
# the question and the best passage. Heuristic, not calibrated: MiniLM puts
# loosely related text around 0.2-0.35 and a passage on the question's actual
# subject above ~0.5. Calibrate them against eval/judged_queries.jsonl before
# quoting them as anything firmer.
HIGH_SIMILARITY = 0.5
MEDIUM_SIMILARITY = 0.35

_CITATION = re.compile(r"\[(\d+)\]")

# Sentence ends: terminal punctuation followed by whitespace. Citations written
# after the full stop ("...gas. [1] Then") are re-attached to the sentence
# they follow by `split_claims`, not left to open the next one.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# A sentence about the passages rather than from them - "The passages do not
# say how long to knead." The prompt asks for exactly these, so they carry no
# citation by design and are reported as notes instead of counted as
# unsupported claims, which would penalise the answer for being honest.
_META = re.compile(
    r"\b(passages?|sources?|the (?:book|text)s? (?:do(?:es)?n[o']t|do(?:es)? not))\b",
    re.IGNORECASE,
)

# An attribution in front of a claim - "The book claims that yeast is a
# vegetable." answer.py's prompt asks for exactly this when a passage is
# outdated, and an NLI model reads "the book claims" as one more fact the
# passage would have to state, scoring a faithful sentence as unsupported. The
# check is on what follows; the claim is still shown as written.
_ATTRIBUTION = re.compile(
    r"^(?:according to (?:the |this |one )?(?:book|author|text|passage)s?,?\s*"
    r"|(?:the |this |one )?(?:book|author|text|passage)s?\s+"
    r"(?:claims?|says?|states?|suggests?|argues?|reports?|notes?|explains?|"
    r"describes?|recommends?|advises?|holds?|believes?|maintains?)"
    r"(?:\s+that)?,?\s*)",
    re.IGNORECASE,
)

_MODEL = None
_LOCK = threading.Lock()


def _checkable(text: str) -> str:
    """The claim as the checker should read it: without a leading attribution."""
    stripped = _ATTRIBUTION.sub("", text, count=1)
    return stripped[:1].upper() + stripped[1:] if stripped else text


def _model():
    """The NLI cross-encoder, loaded once on first use."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from sentence_transformers import CrossEncoder

            _MODEL = CrossEncoder(NLI_MODEL)
        return _MODEL


def split_claims(answer: str) -> list[dict]:
    """Split an answer into sentences, each with the passage numbers it cites.

    Returns [{"text": sentence with citations removed, "cited": [n, ...]}].
    Lines are split first (answers are often bulleted), then sentences.
    Fragments with no letters - a stray "[2]" or a bullet marker - are dropped.
    """
    pieces: list[str] = []
    for line in answer.splitlines():
        for sentence in _SENTENCE_END.split(line.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Leading citations belong to the sentence before: "gas. [1] Then".
            lead = re.match(r"^((?:\[\d+\]\s*)+)(.*)$", sentence)
            if lead and pieces:
                pieces[-1] += " " + lead.group(1).strip()
                sentence = lead.group(2).strip()
            if sentence:
                pieces.append(sentence)

    claims = []
    for piece in pieces:
        cited = sorted({int(n) for n in _CITATION.findall(piece)})
        text = re.sub(r"\s{2,}", " ", _CITATION.sub("", piece))
        text = re.sub(r"\s+([.,;:!?])", r"\1", text).strip(" -*•\t")
        if re.search(r"[A-Za-z]", text):
            claims.append({"text": text, "cited": cited})
    return claims


def _entailment(pairs: list[tuple[str, str]], model) -> list[float]:
    """P(entailment) for each (premise, hypothesis) pair, in one batch."""
    if not pairs:
        return []
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    column = labels["entailment"]
    probabilities = model.predict(pairs, apply_softmax=True)
    return [float(row[column]) for row in probabilities]


def check_claims(answer: str, sources: list[dict], model=None) -> dict:
    """Score how well each sentence of `answer` is supported by what it cites.

    `sources` is answer.py's numbered list ({"n", "text", ...}). A sentence
    citing several passages scores as its best one: a citation list means "see
    these", and one passage that says it is enough. Returns:

        {"claims": [{"text", "cited", "support", "status"}],
         "supported": int, "checked": int, "score": float | None}

    status is "supported", "unsupported", "uncited" (a claim that cites
    nothing - counted, as unsupported) or "note" (a sentence about the
    passages themselves - listed, not counted). `score` is supported/checked,
    or None when nothing was checkable.
    """
    by_number = {s["n"]: s["text"] for s in sources}
    claims = split_claims(answer)

    pairs, owners = [], []
    for index, claim in enumerate(claims):
        for n in claim["cited"]:
            if n in by_number:
                pairs.append((by_number[n], _checkable(claim["text"])))
                owners.append(index)

    scores = _entailment(pairs, model or _model()) if pairs else []
    best: dict[int, float] = {}
    for index, score in zip(owners, scores):
        best[index] = max(best.get(index, 0.0), score)

    for index, claim in enumerate(claims):
        if index in best:
            claim["support"] = round(best[index], 3)
            claim["status"] = "supported" if best[index] >= SUPPORTED else "unsupported"
        else:
            claim["support"] = None
            claim["status"] = "note" if _META.search(claim["text"]) else "uncited"

    counted = [c for c in claims if c["status"] != "note"]
    supported = sum(c["status"] == "supported" for c in counted)
    return {
        "claims": claims,
        "supported": supported,
        "checked": len(counted),
        "score": round(supported / len(counted), 3) if counted else None,
    }


def retrieval_score(results: list[dict]) -> dict:
    """How closely the passages handed to the model matched the question.

    Uses the best `similarity` among the results - set by hybrid_search on
    every vector hit; graph-only hits have none and are skipped. Returns
    {"similarity": float | None, "level": "high" | "medium" | "low" | None}.
    """
    values = [r["similarity"] for r in results if isinstance(r.get("similarity"), (int, float))]
    if not values:
        return {"similarity": None, "level": None}
    top = max(values)
    if top >= HIGH_SIMILARITY:
        level = "high"
    elif top >= MEDIUM_SIMILARITY:
        level = "medium"
    else:
        level = "low"
    return {"similarity": round(float(top), 3), "level": level}

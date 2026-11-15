"""Answer a question from a loaded OKF bundle: hybrid retrieval, then an LLM.

Retrieval picks the passages; the model only writes. It is shown numbered
passages and told to answer from them alone, cite them as [n], and say so when
they do not contain the answer. That last instruction is the one that matters
most for this library: it is pre-1920, and a question about jet lag or
passports will still retrieve *something* - the nearest Victorian chapter on
travel - which a model left to its own devices will happily dress up as an
answer. The probe needs in eval/needs.csv are exactly those questions.

Citations are checked after the fact rather than trusted: a number that names
no passage is dropped, and the returned sources are the ones actually cited.

This module never feeds back into ranking. `hybrid_search` is called with its
defaults, so the retrieval evaluation in eval/ measures the same system this
answers from, and the generation step cannot move those numbers.
"""

import argparse
import os
import re

from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import retrieval
from neo4j_loader import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from passages import with_passage
from qdrant_loader import QDRANT_URL

# A small, cheap model is enough: the hard part - finding the passages - is
# already done, and the prompt forbids using anything the passages don't say.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Passages handed to the model. Each is one ~160-word chunk, so six is about a
# thousand words of context: enough to cover a need that spans two or three
# books, small enough that the model reads all of it.
TOP_K = 6

# Bounds, not targets, for the same reason as retrieval's store timeouts: a
# provider that accepts the request and then stalls should cost one request,
# not a worker.
OPENAI_TIMEOUT = retrieval._seconds("OPENAI_TIMEOUT", 30.0)
OPENAI_MAX_RETRIES = 2

# What the model says, verbatim, when the passages don't answer the question.
# A fixed token rather than "I don't know" in prose, so declining is detected
# exactly instead of guessed at from wording.
NO_ANSWER = "NOT_IN_SOURCES"

SYSTEM_PROMPT = f"""You answer questions using only the numbered passages you are given.

Rules:
- Use only what the passages say. Do not add facts from your own knowledge.
- Cite every claim with the passage number in square brackets, like [2] or [1][3].
- The passages come from old books. Report what they say in plain modern
  English, and if a passage's view is outdated, say it is what the book claims.
- The question may be phrased loosely or indirectly. Work out what it is really
  asking, then answer whatever part of it the passages support, even if they
  answer only part of it. Say plainly what the passages leave unanswered.
- Reply with exactly {NO_ANSWER} and nothing else only when no passage says
  anything that helps answer the question. Sharing a word or a general subject
  with the question is not enough.
"""

_CITATION = re.compile(r"\[(\d+)\]")

_CLIENT = None


def _client():
    """The OpenAI client, built on first use so importing this stays cheap."""
    global _CLIENT
    if _CLIENT is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LookupError("set OPENAI_API_KEY (e.g. in a .env file) to generate answers")
        # Imported here so search-only installs never need the package.
        from openai import OpenAI

        _CLIENT = OpenAI(timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
    return _CLIENT


def build_sources(results: list[dict]) -> list[dict]:
    """Number the results that have something to quote, best first.

    A result whose source file can't be re-read has no passage; its
    description is the next best thing, and one with neither is dropped -
    giving the model a title alone invites it to fill in the rest.
    """
    sources = []
    for result in results:
        passage = (result.get("passage") or {}).get("text") or result.get("description")
        if not passage:
            continue
        sources.append({
            "n": len(sources) + 1,
            "id": result["id"],
            "title": result.get("title") or result["id"],
            "text": passage,
        })
    return sources


def build_prompt(question: str, sources: list[dict]) -> str:
    """The user message: the numbered passages, then the question."""
    blocks = [f"[{s['n']}] {s['title']}\n{s['text']}" for s in sources]
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def parse_reply(reply: str, sources: list[dict]) -> dict:
    """Split the model's reply into the answer and the sources it really cited.

    Citations to passage numbers that don't exist are stripped from the text,
    so a reader never follows a [7] to nothing. An answer that cites nothing
    at all is still returned - but flagged, because the prompt required
    citations and an uncited answer is the likeliest place for invention.
    """
    declined = {"answered": False, "answer": None, "cited": [], "uncited": False}
    reply = reply.strip()
    if not reply or reply.startswith(NO_ANSWER):
        return declined

    by_number = {s["n"]: s for s in sources}
    if NO_ANSWER in reply:
        # The model often explains itself first and puts the token last
        # ("The passages do not cover X... Therefore, NOT_IN_SOURCES."). With
        # no valid citation that is a decline in prose, not an uncited answer.
        # With citations it is a partial answer: keep it, minus the token.
        if not any(int(n) in by_number for n in _CITATION.findall(reply)):
            return declined
        reply = reply.replace(NO_ANSWER, "").strip()

    text = _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in by_number else "", reply)
    numbers = sorted({int(n) for n in _CITATION.findall(text)})
    return {
        "answered": True,
        "answer": re.sub(r"[ \t]{2,}", " ", text).strip(),
        "cited": [{k: by_number[n][k] for k in ("n", "id", "title")} for n in numbers],
        "uncited": not numbers,
    }


def generate(question: str, results: list[dict], client=None) -> dict:
    """Answer `question` from already-retrieved results that carry passages.

    Separate from `ask` so the generation step can be tested, or reused by a
    caller that has its own results, without either store running.
    """
    sources = build_sources(results)
    if not sources:
        # Nothing to quote: don't spend a call asking the model to say so.
        return {"question": question, "answered": False, "answer": None,
                "cited": [], "uncited": False, "sources": []}

    response = (client or _client()).chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, sources)},
        ],
    )
    parsed = parse_reply(response.choices[0].message.content or "", sources)
    return {
        "question": question,
        **parsed,
        "sources": [{k: s[k] for k in ("n", "id", "title")} for s in sources],
    }


def ask(question: str, top_k: int = TOP_K, client=None) -> dict:
    """Retrieve passages for `question` with hybrid search and answer from them."""
    results = retrieval.hybrid_search(question, top_k=top_k)
    return generate(question, [with_passage(r, question) for r in results], client=client)


def main() -> None:
    """Answer one question from the loaded bundle and print it with sources."""
    ap = argparse.ArgumentParser(
        description="Answer a question from a loaded OKF bundle, with citations."
    )
    ap.add_argument("question", help="the question to answer")
    ap.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"passages to give the model (default: {TOP_K})",
    )
    args = ap.parse_args()

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise SystemExit(
            "set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD (e.g. in a .env file)"
        )

    from openai import APIConnectionError, APIStatusError, AuthenticationError

    try:
        result = ask(args.question, top_k=args.top_k)
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
            f"Neo4j rejected the credentials for user '{NEO4J_USER}' - "
            "check NEO4J_USER and NEO4J_PASSWORD"
        ) from None
    except ConfigurationError as exc:
        raise SystemExit(f"bad Neo4j configuration (NEO4J_URI={NEO4J_URI}): {exc}") from None
    except AuthenticationError:
        raise SystemExit("OpenAI rejected the API key - check OPENAI_API_KEY") from None
    except APIConnectionError:
        raise SystemExit("could not reach the OpenAI API - check the network") from None
    except APIStatusError as exc:
        raise SystemExit(f"OpenAI returned an error ({exc.status_code}): {exc.message}") from None

    if not result["answered"]:
        print("The library doesn't answer that.")
        if result["sources"]:
            print("\nClosest passages, none of which answered it:")
            for s in result["sources"]:
                print(f"  [{s['n']}] {s['title']}  ({s['id']})")
        return

    print(result["answer"])
    if result["uncited"]:
        print("\n(warning: the answer cites no passage - treat it with suspicion)")
    print("\nSources:")
    for s in result["cited"]:
        print(f"  [{s['n']}] {s['title']}  ({s['id']})")


if __name__ == "__main__":
    main()

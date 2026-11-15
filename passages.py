"""Recover the passage behind a search result, for showing it or quoting it.

Qdrant stores no chunk text, only vectors, so anything that wants the words a
result matched on - the web UI displaying them, `answer.py` handing them to an
LLM - has to re-chunk the concept's source file exactly as the loader did. Both
callers share this module rather than one importing the other.
"""

import os

from sentence_transformers.util import cos_sim

import qdrant_loader
import retrieval
from parser import parse_okf_file


def with_passage(result: dict, query: str) -> dict:
    """Return the result with the passage that explains it, when one can be found.

    A vector hit names the chunk that matched; a graph-only hit was reached as
    a whole concept, so it gets the chunk closest to the query instead,
    labelled as such - the walk did not choose that passage, and a caller
    should not imply it did. A missing or unreadable file just means no
    passage, never an error.
    """
    source = result.get("source_path")
    if not source or not os.path.isfile(source):
        return result
    try:
        concept = parse_okf_file(source)
    except ValueError:  # undecodable text: parse_okf_file only catches OSError
        return result
    if concept is None:
        return result
    chunks = qdrant_loader.concept_chunks(concept)

    index = result.get("chunk_index")
    if isinstance(index, int) and 0 <= index < len(chunks):
        how = "matched"
    else:
        model = retrieval._model()
        scores = cos_sim(model.encode(query), model.encode(chunks))[0]
        index, how = int(scores.argmax()), "closest"

    text = drop_heading(chunks[index], concept)
    if not text:
        return result
    return {**result, "passage": {"text": text, "how": how, "chunk_index": index}}


def drop_heading(text: str, concept: dict) -> str:
    """Strip leading repeats of the concept's description and title.

    The loader embeds the description ahead of the body, and a converted body
    usually opens with the same words as its heading, so the first chunk reads
    "Part 66 Part 66 elastic glutinous mass...". The result already carries the
    title, so the passage starts at the text. Only the copy changes; the chunk
    as searched is untouched.
    """
    frontmatter = concept.get("frontmatter") or {}
    labels = [
        " ".join(str(frontmatter[key]).split())
        for key in ("description", "title")
        if str(frontmatter.get(key) or "").strip()
    ]
    text = " ".join(text.split())
    stripped = True
    while stripped:
        stripped = False
        for label in labels:
            if text.lower().startswith(label.lower()):
                text = text[len(label):].lstrip(" -—:")
                stripped = True
    return text

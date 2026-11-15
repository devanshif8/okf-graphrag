"""The retrieval quality gate: fixed corpus, fixed queries, floors that fail CI.

Every other test here asserts that the pipeline *runs*. This one asserts that it
still *works*. Retrieval quality has no compiler and no stack trace: a change to
chunk width, fusion weights, degree damping or the embedding text can leave
every test green while quietly making search worse, and nobody finds out until
someone notices bad results months later. A floor is what turns that silent
drift into a failed build.

The corpus is `eval/regression/bundle` — eleven hand-written OKF files across
brewing, baking and microbiology, committed to the repo so the measurement is
reproducible. It is small on purpose: CI has to finish in minutes, and a floor
only means something if the input is byte-identical every run. It is emphatically
not an accuracy benchmark — eleven documents cannot tell you how good retrieval
is, only whether it got worse. Measuring accuracy needs a real corpus and real
queries; see the README on the evaluation harness.

The floors sit well below the measured scores. A gate that fails on noise gets
disabled, so the margin has to absorb an embedding-model point release without
absorbing a real regression. Measured on 2026-09-10 with all-MiniLM-L6-v2:

    run        mrr    recall@5
    vector    0.913     1.000
    graph     0.632     1.000
    hybrid    0.883     1.000

When a floor fails, read the printed table before lowering it: the number moved
because something changed, and the change is the thing to look at.
"""

import json
import os
import shutil
import uuid

import pytest

import neo4j_loader
import qdrant_loader
import retrieval
from eval.evaluate import _table, evaluate
from parser import parse_bundle

from .conftest import _purge

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_BUNDLE = os.path.join(REPO_ROOT, "eval", "regression", "bundle")
QUERY_SET = os.path.join(REPO_ROOT, "eval", "regression", "queries.jsonl")

# Floors, not targets. See the module docstring for the measured values.
VECTOR_MRR_FLOOR = 0.80
HYBRID_MRR_FLOOR = 0.75
HYBRID_RECALL_5_FLOOR = 0.90

# The graph list is the half most likely to fail silently: break the linker, the
# edge resolution or the walk and vector search alone still returns something
# plausible. This floor is what notices.
GRAPH_MRR_FLOOR = 0.45


@pytest.fixture(scope="module")
def loaded_fixture(stores, tmp_path_factory):
    """The committed regression bundle, loaded into both stores under a unique id.

    Copied to a uniquely-named directory rather than loaded in place, because
    concept ids are derived from the bundle directory's name: under its own name
    this bundle would collide with a developer's real one in a shared database.
    """
    root = tmp_path_factory.mktemp("gate") / f"okfgate{uuid.uuid4().hex[:10]}"
    shutil.copytree(FIXTURE_BUNDLE, root)
    bundle_root = str(root)

    concepts = parse_bundle(bundle_root)
    neo4j_loader.load_bundle_to_neo4j(concepts, bundle_root)
    qdrant_loader.load_bundle_to_qdrant(concepts, bundle_root)

    yield bundle_root

    _purge(bundle_root)


@pytest.fixture(scope="module")
def queries(loaded_fixture):
    """The committed query set, with ids resolved against the loaded bundle.

    The file stores each answer as a bundle-relative path, so the query set is
    independent of where the bundle was loaded from and of the unique name it
    was loaded under.
    """
    prefix = neo4j_loader.bundle_prefix(loaded_fixture)
    resolved = []
    with open(QUERY_SET, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry["relevant"] = [prefix + path for path in entry["relevant"]]
            entry["region"] = entry.get("tag", "any")
            resolved.append(entry)
    return resolved


@pytest.fixture(scope="module")
def scores(queries):
    """Every metric for every retriever over the fixed query set, measured once."""
    return evaluate(queries, progress_every=0)


def _report(scores) -> str:
    """The table to print alongside a failed floor, so the failure is readable."""
    return f"\n\n{_table(scores['runs'])}\n"


class TestQualityFloors:
    def test_vector_search_meets_its_mrr_floor(self, scores):
        measured = scores["runs"]["vector"]["mrr"]

        assert measured >= VECTOR_MRR_FLOOR, (
            f"vector MRR {measured:.3f} < floor {VECTOR_MRR_FLOOR}"
            f"{_report(scores)}"
        )

    def test_hybrid_search_meets_its_mrr_floor(self, scores):
        measured = scores["runs"]["hybrid"]["mrr"]

        assert measured >= HYBRID_MRR_FLOOR, (
            f"hybrid MRR {measured:.3f} < floor {HYBRID_MRR_FLOOR}"
            f"{_report(scores)}"
        )

    def test_hybrid_search_finds_the_answer_within_five_results(self, scores):
        measured = scores["runs"]["hybrid"]["recall@5"]

        assert measured >= HYBRID_RECALL_5_FLOOR, (
            f"hybrid recall@5 {measured:.3f} < floor {HYBRID_RECALL_5_FLOOR}"
            f"{_report(scores)}"
        )

    def test_the_graph_list_still_contributes(self, scores):
        """Break the linker, the edge resolution or the walk and vector search
        alone still looks fine. This is the assertion that notices."""
        measured = scores["runs"]["graph"]["mrr"]

        assert measured >= GRAPH_MRR_FLOOR, (
            f"graph MRR {measured:.3f} < floor {GRAPH_MRR_FLOOR} - the graph "
            f"half of the hybrid has stopped contributing{_report(scores)}"
        )


class TestChunkingIsLoadBearing:
    def test_content_late_in_a_long_document_is_retrievable(self, queries):
        """The chunking canary.

        `brewing/yeast-pitching.md` puts its diacetyl material past word 160, so
        it is only reachable if the document was split into more than one vector.
        Verified as a negative control: with chunking disabled — one point per
        concept, truncated by the model at 256 word-pieces — two of these three
        queries do not retrieve the document *at all* in the top 5, and with
        chunking both come back at rank 1.

        So this test fails the moment chunking regresses, which no other
        assertion in the suite would catch.
        """
        late = [entry for entry in queries if entry.get("tag") == "late-chunk"]
        assert late, "the query set has lost its late-chunk queries"

        missed = []
        for entry in late:
            found = {hit["id"] for hit in retrieval.vector_search(entry["query"], top_k=5)}
            if not found & set(entry["relevant"]):
                missed.append(entry["query"])

        assert not missed, (
            "content past the embedding window is unreachable - chunking is "
            f"broken. Queries that found nothing: {missed}"
        )

    def test_a_long_document_is_stored_as_several_points(self, loaded_fixture):
        """The direct form of the same check, independent of any query."""
        concepts = parse_bundle(loaded_fixture)
        long_doc = next(
            c for c in concepts if c["path"].endswith("yeast-pitching.md")
        )

        assert len(qdrant_loader.concept_chunks(long_doc)) > 1

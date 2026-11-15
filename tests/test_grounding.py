"""Unit tests for grounding.py - claim splitting, the support check, and the
retrieval score.

The NLI model is replaced by a fake that entails a claim when every word of it
appears in the passage, so the logic is tested without the ~280MB download.
The real model is exercised by the opt-in test at the bottom.
"""

import os
import re
from types import SimpleNamespace

import pytest

import grounding
from grounding import check_claims, retrieval_score, split_claims


class WordOverlapNLI:
    """Entails (premise, hypothesis) iff the hypothesis' words are all in the premise.

    Labels are deliberately in a different order from the real checkpoint, so
    a test fails if the entailment column is ever assumed instead of looked up.
    """

    config = SimpleNamespace(id2label={0: "entailment", 1: "neutral", 2: "contradiction"})

    def __init__(self):
        self.calls = 0

    def predict(self, pairs, apply_softmax=True):
        self.calls += 1
        words = lambda text: set(re.findall(r"[a-z]+", text.lower()))
        return [
            [1.0, 0.0, 0.0] if words(h) <= words(p) else [0.0, 1.0, 0.0]
            for p, h in pairs
        ]


SOURCES = [
    {"n": 1, "text": "Yeast feeds on sugar and makes gas."},
    {"n": 2, "text": "A hot oven sets the crust."},
]


class TestSplitClaims:
    def test_one_claim_per_sentence_with_its_citations(self):
        assert split_claims("Yeast makes gas [1]. Ovens set crust [2][1].") == [
            {"text": "Yeast makes gas.", "cited": [1]},
            {"text": "Ovens set crust.", "cited": [1, 2]},
        ]

    def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before(self):
        claims = split_claims("Yeast makes gas. [1] Ovens set crust. [2]")
        assert [c["cited"] for c in claims] == [[1], [2]]
        assert claims[1]["text"] == "Ovens set crust."

    def test_bullets_and_lines_are_split_and_markers_dropped(self):
        claims = split_claims("- Yeast makes gas [1]\n- Ovens set crust [2]")
        assert [c["text"] for c in claims] == ["Yeast makes gas", "Ovens set crust"]

    def test_fragments_without_words_are_dropped(self):
        assert split_claims("[1]\n\n  ") == []

    def test_a_sentence_without_citations_has_none(self):
        assert split_claims("Bread is old.") == [{"text": "Bread is old.", "cited": []}]


class TestCheckClaims:
    def test_a_claim_its_passage_says_is_supported(self):
        out = check_claims("Yeast makes gas [1].", SOURCES, model=WordOverlapNLI())
        [claim] = out["claims"]
        assert claim["status"] == "supported" and claim["support"] == 1.0
        assert (out["supported"], out["checked"], out["score"]) == (1, 1, 1.0)

    def test_a_claim_its_passage_does_not_say_is_unsupported(self):
        out = check_claims("Bread rises faster in summer [1].", SOURCES, model=WordOverlapNLI())
        assert out["claims"][0]["status"] == "unsupported"
        assert out["score"] == 0.0

    def test_a_claim_is_checked_only_against_the_passage_it_cites(self):
        # True of passage 2, but it cites passage 1: that is a bad citation.
        out = check_claims("A hot oven sets the crust [1].", SOURCES, model=WordOverlapNLI())
        assert out["claims"][0]["status"] == "unsupported"

    def test_several_citations_score_as_the_best_of_them(self):
        out = check_claims("A hot oven sets the crust [1][2].", SOURCES, model=WordOverlapNLI())
        assert out["claims"][0]["status"] == "supported"

    def test_an_uncited_claim_counts_against_the_score(self):
        out = check_claims("Yeast makes gas [1]. Bread is ancient.", SOURCES,
                           model=WordOverlapNLI())
        assert [c["status"] for c in out["claims"]] == ["supported", "uncited"]
        assert (out["supported"], out["checked"], out["score"]) == (1, 2, 0.5)

    def test_a_remark_about_the_passages_is_a_note_not_a_failure(self):
        out = check_claims(
            "Yeast makes gas [1]. The passages do not say how long to knead.",
            SOURCES, model=WordOverlapNLI(),
        )
        assert out["claims"][1]["status"] == "note"
        assert (out["supported"], out["checked"]) == (1, 1)

    def test_a_citation_to_a_missing_passage_is_treated_as_uncited(self):
        out = check_claims("Yeast makes gas [9].", SOURCES, model=WordOverlapNLI())
        assert out["claims"][0]["status"] == "uncited"

    @pytest.mark.parametrize("prefix", [
        "The book claims that ", "According to the book, ", "The author says ",
    ])
    def test_an_attribution_is_not_checked_as_part_of_the_claim(self, prefix):
        out = check_claims(prefix + "yeast makes gas [1].", SOURCES, model=WordOverlapNLI())
        assert out["claims"][0]["status"] == "supported"
        # Shown as written: the prefix is dropped only for the check.
        assert out["claims"][0]["text"].startswith(prefix.strip(" ,"))

    def test_all_pairs_go_to_the_model_in_one_batch(self):
        model = WordOverlapNLI()
        check_claims("Yeast makes gas [1]. A hot oven sets the crust [2].", SOURCES, model=model)
        assert model.calls == 1

    def test_nothing_checkable_scores_none_without_calling_the_model(self):
        model = WordOverlapNLI()
        out = check_claims("The passages do not cover this.", SOURCES, model=model)
        assert out["score"] is None and model.calls == 0


class TestRetrievalScore:
    def test_uses_the_best_similarity(self):
        assert retrieval_score([{"similarity": 0.4}, {"similarity": 0.62}]) == {
            "similarity": 0.62, "level": "high"}

    @pytest.mark.parametrize("value, level", [
        (grounding.HIGH_SIMILARITY, "high"),
        (grounding.MEDIUM_SIMILARITY, "medium"),
        (grounding.MEDIUM_SIMILARITY - 0.01, "low"),
    ])
    def test_bands(self, value, level):
        assert retrieval_score([{"similarity": value}])["level"] == level

    def test_graph_only_results_carry_no_similarity_and_are_skipped(self):
        assert retrieval_score([{"id": "c"}, {"similarity": 0.3}])["similarity"] == 0.3
        assert retrieval_score([{"id": "c"}]) == {"similarity": None, "level": None}


@pytest.mark.skipif(
    not os.getenv("OKF_TEST_NLI"),
    reason="downloads the real NLI model (~280MB); set OKF_TEST_NLI=1 to run",
)
class TestRealChecker:
    PASSAGE = ("Yeast is a living plant. When it is mixed with flour and water it "
               "feeds on the sugar and gives off carbonic acid gas, which puffs up the dough.")

    def test_separates_a_paraphrase_from_an_invention(self):
        out = check_claims(
            "Yeast feeds on sugar and gives off gas [1]. Bread rises faster in summer [1].",
            [{"n": 1, "text": self.PASSAGE}],
        )
        assert [c["status"] for c in out["claims"]] == ["supported", "unsupported"]

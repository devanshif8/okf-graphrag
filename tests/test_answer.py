"""Unit tests for answer.py - prompt building, reply parsing, and the call.

No store, no network: the OpenAI client is replaced by a fake that records the
request and returns a canned reply.
"""

from types import SimpleNamespace

import pytest

import answer
from answer import NO_ANSWER, build_prompt, build_sources, generate, parse_reply


def result(identifier, text=None, description=None, title=None):
    r = {"id": identifier, "title": title, "description": description}
    if text is not None:
        r["passage"] = {"text": text, "how": "matched", "chunk_index": 0}
    return r


class FakeClient:
    """Stands in for openai.OpenAI: records each request, returns `reply`."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        message = SimpleNamespace(content=self.reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChecker:
    """Stands in for the NLI cross-encoder: entails every pair it is shown."""

    config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})

    def __init__(self):
        self.pairs = []

    def predict(self, pairs, apply_softmax=True):
        self.pairs.extend(pairs)
        return [[0.0, 1.0, 0.0] for _ in pairs]


@pytest.fixture(autouse=True)
def no_real_checker(monkeypatch):
    """Never load the real NLI model here: it is ~280MB and tested on its own."""
    monkeypatch.setattr(answer.grounding, "_model", lambda: FakeChecker())


SOURCES = [
    {"n": 1, "id": "shelf/a.md", "title": "Yeast", "text": "Yeast makes gas."},
    {"n": 2, "id": "shelf/b.md", "title": "Ovens", "text": "Ovens are hot."},
]


class TestBuildSources:
    def test_numbers_results_from_one_in_rank_order(self):
        sources = build_sources([result("a", "x", title="A"), result("b", "y", title="B")])
        assert [(s["n"], s["id"], s["text"]) for s in sources] == [(1, "a", "x"), (2, "b", "y")]

    def test_falls_back_on_the_description_when_there_is_no_passage(self):
        [source] = build_sources([result("a", description="About bread")])
        assert source["text"] == "About bread"

    def test_drops_a_result_with_nothing_to_quote_without_leaving_a_gap(self):
        sources = build_sources([result("a"), result("b", "y")])
        assert [(s["n"], s["id"]) for s in sources] == [(1, "b")]

    def test_uses_the_id_when_there_is_no_title(self):
        [source] = build_sources([result("shelf/a.md", "x")])
        assert source["title"] == "shelf/a.md"


class TestBuildPrompt:
    def test_lists_numbered_passages_before_the_question(self):
        prompt = build_prompt("Why does bread rise?", SOURCES)
        assert prompt.index("[1] Yeast\nYeast makes gas.") < prompt.index("[2] Ovens")
        assert prompt.rstrip().endswith("Question: Why does bread rise?")


class TestParseReply:
    def test_keeps_the_answer_and_lists_the_sources_it_cited(self):
        parsed = parse_reply("Yeast makes gas [1].", SOURCES)
        assert parsed["answered"] is True
        assert parsed["answer"] == "Yeast makes gas [1]."
        assert [c["id"] for c in parsed["cited"]] == ["shelf/a.md"]
        assert parsed["uncited"] is False

    def test_the_no_answer_token_means_not_answered(self):
        parsed = parse_reply(f"  {NO_ANSWER}\n", SOURCES)
        assert parsed == {"answered": False, "answer": None, "cited": [], "uncited": False}

    def test_a_decline_explained_in_prose_with_the_token_last_is_a_decline(self):
        # Seen live: 5 of 41 needs came back like this and showed as
        # "uncited answers" instead of declines.
        reply = f"The passages do not cover cross-contamination. Therefore, {NO_ANSWER}."
        assert parse_reply(reply, SOURCES) == {
            "answered": False, "answer": None, "cited": [], "uncited": False,
        }

    def test_a_cited_answer_that_also_says_the_token_keeps_the_answer(self):
        parsed = parse_reply(f"Yeast makes gas [1]. The rest is {NO_ANSWER}", SOURCES)
        assert parsed["answered"] is True
        assert NO_ANSWER not in parsed["answer"]
        assert [c["n"] for c in parsed["cited"]] == [1]

    def test_the_token_with_only_missing_citations_is_a_decline(self):
        assert parse_reply(f"See [9]. {NO_ANSWER}", SOURCES)["answered"] is False

    def test_an_empty_reply_means_not_answered(self):
        assert parse_reply("", SOURCES)["answered"] is False

    def test_strips_citations_to_passages_that_do_not_exist(self):
        parsed = parse_reply("Yeast makes gas [1] [7] and rises [2].", SOURCES)
        assert parsed["answer"] == "Yeast makes gas [1] and rises [2]."
        assert [c["n"] for c in parsed["cited"]] == [1, 2]

    def test_cites_each_source_once_in_number_order(self):
        parsed = parse_reply("A [2]. B [1][2].", SOURCES)
        assert [c["n"] for c in parsed["cited"]] == [1, 2]

    def test_flags_an_answer_that_cites_nothing(self):
        parsed = parse_reply("Bread rises because of magic.", SOURCES)
        assert parsed["answered"] is True
        assert parsed["uncited"] is True
        assert parsed["cited"] == []

    def test_an_answer_citing_only_missing_passages_counts_as_uncited(self):
        assert parse_reply("Something [9].", SOURCES)["uncited"] is True


class TestGenerate:
    def test_sends_the_system_prompt_and_numbered_passages_deterministically(self):
        client = FakeClient("Yeast makes gas [1].")
        generate("Why?", [result("a", "Yeast makes gas.", title="Yeast")], client=client)

        [request] = client.requests
        assert request["model"] == answer.OPENAI_MODEL
        assert request["temperature"] == 0
        system, user = request["messages"]
        assert system == {"role": "system", "content": answer.SYSTEM_PROMPT}
        assert "[1] Yeast\nYeast makes gas." in user["content"]
        assert NO_ANSWER in answer.SYSTEM_PROMPT

    def test_returns_the_answer_with_all_sources_offered_and_those_cited(self):
        client = FakeClient("Gas [2].")
        out = generate("Why?", [result("a", "x"), result("b", "y")], client=client)
        assert out["question"] == "Why?"
        assert out["answer"] == "Gas [2]."
        assert [s["id"] for s in out["sources"]] == ["a", "b"]
        assert [c["id"] for c in out["cited"]] == ["b"]
        # Passage text is for the model, not the response.
        assert "text" not in out["sources"][0]

    def test_does_not_call_the_model_when_there_is_nothing_to_quote(self):
        client = FakeClient("should not be used")
        out = generate("Why?", [result("a")], client=client)
        assert client.requests == []
        assert out["answered"] is False
        assert out["sources"] == []

    def test_a_null_reply_is_treated_as_no_answer(self):
        assert generate("Why?", [result("a", "x")], client=FakeClient(None))["answered"] is False

    def test_an_answer_is_checked_claim_by_claim_against_what_it_cites(self):
        checker = FakeChecker()
        out = generate(
            "Why?",
            [result("a", "Yeast makes gas."), result("b", "Ovens are hot.")],
            client=FakeClient("Yeast makes gas [1]. Ovens are hot [2]."),
            checker=checker,
        )
        assert out["grounding"]["supported"] == out["grounding"]["checked"] == 2
        assert checker.pairs == [
            ("Yeast makes gas.", "Yeast makes gas."),
            ("Ovens are hot.", "Ovens are hot."),
        ]

    def test_a_declined_answer_has_no_support_check(self):
        out = generate("Why?", [result("a", "x")], client=FakeClient(NO_ANSWER))
        assert out["grounding"] is None

    def test_reports_the_retrieval_score_with_or_without_an_answer(self):
        hits = [{**result("a", "x"), "similarity": 0.61}]
        assert generate("Why?", hits, client=FakeClient("X [1]."))["retrieval"] == {
            "similarity": 0.61, "level": "high"}
        assert generate("Why?", hits, client=FakeClient(NO_ANSWER))["retrieval"]["level"] == "high"

    def test_a_checker_that_cannot_load_marks_the_answer_unchecked(self, monkeypatch):
        def broken():
            raise OSError("no network")

        monkeypatch.setattr(answer.grounding, "_model", broken)
        out = generate("Why?", [result("a", "x")], client=FakeClient("X [1]."))
        assert out["answered"] is True
        assert "unavailable" in out["grounding"]["error"]


class TestAsk:
    def test_retrieves_with_default_ranking_and_attaches_passages(self, monkeypatch):
        calls = []

        def fake_search(query, **kwargs):
            calls.append((query, kwargs))
            return [{"id": "a", "title": "A", "description": "desc"}]

        monkeypatch.setattr(answer.retrieval, "hybrid_search", fake_search)
        monkeypatch.setattr(answer, "with_passage", lambda r, q: {**r, "passage": {"text": "p"}})
        client = FakeClient("Yes [1].")

        out = answer.ask("Q?", top_k=3, client=client)

        # Only top_k is passed: the fusion knobs stay at the defaults the
        # retrieval evaluation measured.
        assert calls == [("Q?", {"top_k": 3})]
        assert "[1] A\np" in client.requests[0]["messages"][1]["content"]
        assert out["answered"] is True


class TestClient:
    def test_a_missing_api_key_is_a_lookup_error_naming_the_variable(self, monkeypatch):
        monkeypatch.setattr(answer, "_CLIENT", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(LookupError, match="OPENAI_API_KEY"):
            answer._client()

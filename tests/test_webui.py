"""Unit tests for webui.py — the request layer, not the retrieval it wraps.

Every test stubs `retrieval.hybrid_search`, so nothing here needs Neo4j,
Qdrant or the embedding model running. What is under test is the translation
between an HTTP query string and a `hybrid_search` call, and between a raised
exception and the JSON the page renders.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import retrieval
import webui

RESULT = {
    "id": "ga4/references/metrics/purchasers.md",
    "score": 0.0239,
    "ranks": {"vector": 1, "graph": 7},
    "title": "Purchasers Audience Metric",
    "description": "Counts users who completed a purchase.",
    "source_path": "knowledge-catalog/okf/bundles/ga4/references/metrics/purchasers.md",
}


@pytest.fixture
def captured(monkeypatch):
    """Stub hybrid_search, recording the arguments it was called with."""
    seen = {}

    def fake(query, **kwargs):
        seen["query"] = query
        seen.update(kwargs)
        return [RESULT]

    monkeypatch.setattr(retrieval, "hybrid_search", fake)
    return seen


@pytest.fixture
def client(monkeypatch):
    """A test client for the real ASGI app, with startup stubbed out.

    Entering TestClient as a context manager runs the lifespan hook, which
    would otherwise load the embedding model — seconds these tests neither need
    nor should spend. `close_clients` on the way out is left real: it is a
    no-op when nothing was ever opened.
    """
    monkeypatch.setattr(retrieval, "_model", lambda: None)
    with TestClient(webui.app) as test_client:
        yield test_client


class TestSearchParams:
    def test_defaults_match_the_retrieval_module(self, captured):
        webui._search({"q": ["purchase"]})
        assert captured["query"] == "purchase"
        assert captured["top_k"] == 10
        assert captured["vector_weight"] == retrieval.VECTOR_WEIGHT
        assert captured["graph_weight"] == retrieval.GRAPH_WEIGHT
        assert captured["degree_damping"] == retrieval.DEGREE_DAMPING
        assert captured["exclude_index"] is True

    def test_passes_each_knob_through(self, captured):
        webui._search(
            {
                "q": ["purchase"],
                "top_k": ["3"],
                "vector_weight": ["0.2"],
                "graph_weight": ["1.4"],
                "degree_damping": ["0"],
                "include_index": ["1"],
            }
        )
        assert captured["top_k"] == 3
        assert captured["vector_weight"] == 0.2
        assert captured["graph_weight"] == 1.4
        assert captured["degree_damping"] == 0.0
        assert captured["exclude_index"] is False

    def test_unparseable_number_falls_back_to_the_default(self, captured):
        webui._search({"q": ["purchase"], "top_k": ["abc"], "graph_weight": [""]})
        assert captured["top_k"] == 10
        assert captured["graph_weight"] == retrieval.GRAPH_WEIGHT

    @pytest.mark.parametrize("raw,expected", [("0", 1), ("999", 50), ("-4", 1)])
    def test_top_k_is_clamped(self, captured, raw, expected):
        # A hand-edited URL shouldn't be able to ask for zero results (which
        # Qdrant rejects) or tens of thousands (which would hang the page).
        webui._search({"q": ["purchase"], "top_k": [raw]})
        assert captured["top_k"] == expected

    def test_blank_query_short_circuits_without_searching(self, captured):
        assert webui._search({"q": ["   "]}) == {"query": "", "results": []}
        assert webui._search({}) == {"query": "", "results": []}
        assert captured == {}

    def test_query_is_stripped(self, captured):
        webui._search({"q": ["  purchase  "]})
        assert captured["query"] == "purchase"


class TestErrorMapping:
    @pytest.mark.parametrize(
        "exc,fragment",
        [
            (LookupError("collection 'okf_concepts' does not exist"), "does not exist"),
            (ResponseHandlingException("boom"), "could not reach Qdrant"),
            (ConnectionError("boom"), "could not reach Qdrant"),
            (ServiceUnavailable("boom"), "could not reach Neo4j"),
            (AuthError("boom"), "rejected the credentials"),
            (ConfigurationError("boom"), "bad Neo4j configuration"),
        ],
    )
    def test_store_failures_become_an_error_message(self, monkeypatch, exc, fragment):
        def raise_it(*args, **kwargs):
            raise exc

        monkeypatch.setattr(retrieval, "hybrid_search", raise_it)
        assert fragment in webui._search({"q": ["purchase"]})["error"]

    def test_unexpected_response_is_reported_verbatim(self, monkeypatch):
        def raise_it(*args, **kwargs):
            raise UnexpectedResponse(400, "Bad Request", b"bad vector", {})

        monkeypatch.setattr(retrieval, "hybrid_search", raise_it)
        assert "Qdrant rejected the query" in webui._search({"q": ["x"]})["error"]


class TestRoutes:
    def test_root_serves_the_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>okf-graphrag</title>" in response.text
        # The page drives itself off this endpoint; a rename should fail here.
        assert "/api/search?" in response.text

    def test_search_returns_the_annotated_results(self, client, captured, monkeypatch):
        # RESULT's source_path exists in this repo, so the passage lookup would
        # read it and embed; that path has its own tests in TestPassage.
        monkeypatch.setattr(webui, "_with_passage", lambda result, query: result)
        response = client.get("/api/search?q=purchase")
        assert response.status_code == 200
        assert response.json()["query"] == "purchase"
        assert response.json()["results"] == [RESULT]

    def test_store_failure_is_reported_as_upstream_not_a_server_fault(
        self, client, monkeypatch
    ):
        # An unloaded collection is the caller's to fix, so it must not look
        # like a bug in this service or every alert rule misroutes.
        def raise_it(*args, **kwargs):
            raise LookupError("no collection")

        monkeypatch.setattr(retrieval, "hybrid_search", raise_it)
        response = client.get("/api/search?q=purchase")
        assert response.status_code == webui.UPSTREAM_STATUS == 503
        assert response.json() == {"error": "no collection"}

    def test_an_unhandled_error_becomes_a_500_with_an_incident_id(
        self, client, monkeypatch
    ):
        # Under http.server this closed the connection with no response at all.
        def raise_it(*args, **kwargs):
            raise RuntimeError("something nobody anticipated")

        monkeypatch.setattr(retrieval, "hybrid_search", raise_it)
        response = client.get("/api/search?q=purchase")
        assert response.status_code == 500
        assert "incident" in response.json()["error"]
        # The traceback must not reach the client.
        assert "nobody anticipated" not in response.text

    def test_unknown_path_is_404(self, client):
        assert client.get("/admin").status_code == 404

    def test_healthz_reports_the_process_is_up(self, client):
        assert client.get("/healthz").json()["status"] == "ok"



class TestRequestObservability:
    def test_every_response_carries_a_request_id(self, captured):
        with TestClient(webui.app) as client:
            response = client.get("/api/search", params={"q": "purchase"})

        assert len(response.headers["X-Request-Id"]) == 12

    def test_each_request_gets_its_own_id(self, captured):
        with TestClient(webui.app) as client:
            first = client.get("/healthz").headers["X-Request-Id"]
            second = client.get("/healthz").headers["X-Request-Id"]

        assert first != second

    def test_a_request_is_logged_with_its_route_status_and_duration(
        self, captured, caplog
    ):
        with caplog.at_level("INFO", logger="okf-graphrag"):
            with TestClient(webui.app) as client:
                client.get("/api/search", params={"q": "purchase"})

        record = next(r for r in caplog.records if r.message == "request")
        assert record.route == "/api/search"
        assert record.status == 200
        assert record.duration_ms >= 0

    def test_metrics_count_requests_by_route_and_status(self, captured):
        with TestClient(webui.app) as client:
            client.get("/api/search", params={"q": "purchase"})
            body = client.get("/metrics").text

        assert 'okf_requests_total{route="/api/search",status="200"}' in body
        assert "okf_request_duration_seconds_count" in body

    def test_metrics_count_searches_separately_from_requests(self, captured):
        before = webui.metrics.searches

        with TestClient(webui.app) as client:
            client.get("/api/search", params={"q": "purchase"})

        assert webui.metrics.searches == before + 1

    def test_an_upstream_failure_is_counted_as_a_search_error(self, monkeypatch):
        def boom(query, **kwargs):
            raise ServiceUnavailable("neo4j is down")

        monkeypatch.setattr(retrieval, "hybrid_search", boom)
        before = webui.metrics.search_errors

        with TestClient(webui.app) as client:
            response = client.get("/api/search", params={"q": "purchase"})

        assert response.status_code == webui.UPSTREAM_STATUS
        assert webui.metrics.search_errors == before + 1

    def test_the_histogram_buckets_are_cumulative(self):
        metrics = webui._Metrics()
        metrics.observe("/x", 200, 0.3)

        body = metrics.render()

        # 0.3s falls above the 0.25 bound and at or below every bound after it.
        assert 'okf_request_duration_seconds_bucket{le="0.25"} 0' in body
        assert 'okf_request_duration_seconds_bucket{le="0.5"} 1' in body
        assert 'okf_request_duration_seconds_bucket{le="+Inf"} 1' in body


class TestJsonLogging:
    def test_a_record_is_one_json_object(self):
        import json
        import logging

        record = logging.LogRecord(
            "okf-graphrag", logging.INFO, __file__, 1, "request", (), None
        )
        record.route = "/api/search"

        payload = json.loads(webui.JsonFormatter().format(record))

        assert payload["message"] == "request"
        assert payload["level"] == "INFO"
        assert payload["route"] == "/api/search"

    def test_a_value_json_cannot_encode_does_not_break_the_line(self):
        import json
        import logging

        record = logging.LogRecord(
            "okf-graphrag", logging.INFO, __file__, 1, "request", (), None
        )
        record.weird = object()

        assert json.loads(webui.JsonFormatter().format(record))["message"] == "request"


class TestAuthentication:
    def test_no_token_configured_means_no_authentication(self, captured):
        with TestClient(webui.app) as client:
            assert client.get("/api/search", params={"q": "x"}).status_code == 200

    def test_a_configured_token_is_required(self, captured, monkeypatch):
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")

        with TestClient(webui.app) as client:
            response = client.get("/api/search", params={"q": "x"})

        assert response.status_code == 401
        assert "unauthorized" in response.json()["error"]

    @pytest.mark.parametrize(
        "make_request",
        [
            lambda c: c.get(
                "/api/search",
                params={"q": "x"},
                headers={"Authorization": "Bearer s3cret"},
            ),
            lambda c: c.get(
                "/api/search", params={"q": "x"}, headers={"X-API-Key": "s3cret"}
            ),
            lambda c: c.get("/api/search", params={"q": "x", "token": "s3cret"}),
        ],
        ids=["bearer", "api-key-header", "query-param"],
    )
    def test_the_token_is_accepted_by_every_documented_route(
        self, captured, monkeypatch, make_request
    ):
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")

        with TestClient(webui.app) as client:
            assert make_request(client).status_code == 200

    def test_a_wrong_token_is_rejected(self, captured, monkeypatch):
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")

        with TestClient(webui.app) as client:
            response = client.get("/api/search", params={"q": "x", "token": "s3cre"})

        assert response.status_code == 401

    def test_liveness_stays_open_so_a_probe_can_reach_it(self, monkeypatch):
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")

        with TestClient(webui.app) as client:
            assert client.get("/healthz").status_code == 200

    def test_metrics_are_behind_the_token(self, monkeypatch):
        """Request counts and latencies describe usage, so they are not public."""
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")

        with TestClient(webui.app) as client:
            assert client.get("/metrics").status_code == 401
            assert client.get("/metrics", params={"token": "s3cret"}).status_code == 200

    def test_the_page_itself_is_behind_the_token(self, monkeypatch):
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")

        with TestClient(webui.app) as client:
            assert client.get("/").status_code == 401


class TestLoopbackDetection:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5", ""])
    def test_loopback_addresses_are_recognised(self, host):
        assert webui.is_loopback(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "::", "example.com"])
    def test_everything_else_is_treated_as_reachable(self, host):
        """A name that would need DNS to resolve counts as non-loopback: erring
        this way only ever asks for a token that was not strictly needed."""
        assert not webui.is_loopback(host)


class TestPassage:
    """The passage shown under a result, recovered by re-chunking its source."""

    BODY = "First paragraph about the dough.\n\nSecond paragraph about the oven."

    @pytest.fixture
    def source(self, tmp_path):
        path = tmp_path / "chapter.md"
        path.write_text(f"---\ntitle: Bread\n---\n{self.BODY}\n", encoding="utf-8")
        return str(path)

    @pytest.fixture
    def one_chunk_each(self, monkeypatch):
        # Real chunks would merge two short paragraphs; split them so the test
        # can tell which one was picked.
        monkeypatch.setattr(
            webui.qdrant_loader, "concept_chunks",
            lambda concept: concept["body"].strip().split("\n\n"),
        )

    def test_a_vector_hit_shows_the_chunk_that_matched(self, source, one_chunk_each):
        result = webui._with_passage({"id": "x", "source_path": source, "chunk_index": 1}, "q")
        assert result["passage"] == {
            "text": "Second paragraph about the oven.", "how": "matched", "chunk_index": 1,
        }

    def test_a_graph_only_hit_shows_the_chunk_nearest_the_query(
        self, source, one_chunk_each, monkeypatch
    ):
        class Model:
            def encode(self, text):
                vectors = {"oven?": [0.0, 1.0], "First paragraph about the dough.": [1.0, 0.0],
                           "Second paragraph about the oven.": [0.1, 1.0]}
                if isinstance(text, list):
                    return np.array([vectors[t] for t in text])
                return np.array(vectors[text])

        monkeypatch.setattr(retrieval, "_model", lambda: Model())
        result = webui._with_passage({"id": "x", "source_path": source}, "oven?")
        assert result["passage"]["how"] == "closest"
        assert result["passage"]["text"] == "Second paragraph about the oven."

    def test_a_missing_source_file_leaves_the_result_alone(self, tmp_path):
        result = {"id": "x", "source_path": str(tmp_path / "gone.md"), "chunk_index": 0}
        assert webui._with_passage(result, "q") == result

    def test_an_out_of_range_chunk_falls_back_rather_than_failing(
        self, source, one_chunk_each, monkeypatch
    ):
        # A file edited since the load can have fewer chunks than Qdrant recorded.
        class Model:
            def encode(self, text):
                return np.ones((len(text), 2)) if isinstance(text, list) else np.ones(2)

        monkeypatch.setattr(retrieval, "_model", lambda: Model())
        result = webui._with_passage({"id": "x", "source_path": source, "chunk_index": 9}, "q")
        assert result["passage"]["how"] == "closest"

    def test_the_repeated_heading_is_dropped_from_the_passage(self):
        concept = {"frontmatter": {"title": "Kitchen 066", "description": "Kitchen — Part 66"}}
        text = "Kitchen — Part 66 Kitchen — Part 66 elastic glutinous mass"
        assert webui._drop_heading(text, concept) == "elastic glutinous mass"

    def test_a_passage_that_merely_mentions_the_title_later_is_kept_whole(self):
        concept = {"frontmatter": {"title": "Bread"}}
        assert webui._drop_heading("Good Bread rises", concept) == "Good Bread rises"


class TestAnswer:
    """The /api/answer route. `answer.ask` is stubbed; its own logic is
    covered in tests/test_answer.py."""

    ANSWERED = {
        "question": "why?", "answered": True, "answer": "Gas [1].",
        "cited": [{"n": 1, "id": "a", "title": "A"}], "uncited": False,
        "sources": [{"n": 1, "id": "a", "title": "A"}],
    }

    def test_returns_the_answer_from_ask(self, client, monkeypatch):
        asked = []
        monkeypatch.setattr(webui.answer, "ask", lambda q: asked.append(q) or self.ANSWERED)
        response = client.get("/api/answer?q=%20why%3F%20")
        assert response.status_code == 200
        assert response.json() == self.ANSWERED
        assert asked == ["why?"]

    def test_an_empty_question_answers_nothing_without_asking(self, monkeypatch):
        monkeypatch.setattr(webui.answer, "ask", lambda q: pytest.fail("asked"))
        assert webui._answer({"q": ["  "]})["answered"] is False

    def test_a_missing_api_key_is_upstream(self, client, monkeypatch):
        def raise_it(q):
            raise LookupError("set OPENAI_API_KEY")

        monkeypatch.setattr(webui.answer, "ask", raise_it)
        response = client.get("/api/answer?q=why")
        assert response.status_code == 503
        assert response.json() == {"error": "set OPENAI_API_KEY"}

    def test_a_store_failure_while_retrieving_is_upstream(self, client, monkeypatch):
        def raise_it(q):
            raise ServiceUnavailable("boom")

        monkeypatch.setattr(webui.answer, "ask", raise_it)
        response = client.get("/api/answer?q=why")
        assert response.status_code == 503
        assert "could not reach Neo4j" in response.json()["error"]

    @pytest.mark.parametrize(
        "make, fragment",
        [
            (lambda openai, r: openai.AuthenticationError("bad", response=r, body=None), "API key"),
            (lambda openai, r: openai.APIConnectionError(request=r.request), "could not reach the OpenAI API"),
            (lambda openai, r: openai.RateLimitError("slow down", response=r, body=None), "OpenAI returned an error (429)"),
        ],
    )
    def test_openai_failures_become_an_error_message(self, monkeypatch, make, fragment):
        openai = pytest.importorskip("openai")
        import httpx

        status = 429 if "429" in fragment else 401
        response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com"))

        def raise_it(q):
            raise make(openai, response)

        monkeypatch.setattr(webui.answer, "ask", raise_it)
        assert fragment in webui._answer({"q": ["why"]})["error"]

    def test_an_unhandled_error_becomes_a_500_with_an_incident_id(self, client, monkeypatch):
        def raise_it(q):
            raise RuntimeError("something nobody anticipated")

        monkeypatch.setattr(webui.answer, "ask", raise_it)
        response = client.get("/api/answer?q=why")
        assert response.status_code == 500
        assert "incident" in response.json()["error"]
        assert "nobody anticipated" not in response.text

    def test_the_route_is_behind_the_token(self, client, monkeypatch):
        monkeypatch.setattr(webui, "API_TOKEN", "s3cret")
        monkeypatch.setattr(webui.answer, "ask", lambda q: self.ANSWERED)
        assert client.get("/api/answer?q=why").status_code == 401
        assert client.get("/api/answer?q=why&token=s3cret").status_code == 200

    def test_the_page_calls_the_answer_endpoint(self, client):
        assert "/api/answer?" in client.get("/").text

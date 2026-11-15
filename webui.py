"""A small web UI for hybrid search over a loaded OKF bundle.

Serves a single search page plus two JSON endpoints: one wraps
`retrieval.hybrid_search`, so the fusion knobs the CLI exposes as flags are
sliders in a browser instead, and one wraps `answer.ask`, which writes a cited
answer from the retrieved passages.

This is a FastAPI app run under uvicorn rather than the stdlib's `http.server`,
which documents itself as not suitable for production: it has no request
timeouts, no body-size limits and no graceful shutdown, and `ThreadingHTTPServer`
spawns an unbounded thread per connection. uvicorn gives all of that, plus a
lifespan hook — which is what lets the expensive clients in `retrieval` be
opened once at startup and closed once at shutdown instead of per request.

The page itself is still one static file and the API is two endpoints;
FastAPI is here for the server underneath it, not for a framework's features.
"""

import argparse
import contextlib
import json
import logging
import os
import secrets
import threading
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

import answer
import qdrant_loader
import retrieval
# Re-exported under the old private names: the search path below calls them
# through this module, which is also where the tests patch them.
from passages import drop_heading as _drop_heading
from passages import with_passage as _with_passage
from neo4j_loader import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from qdrant_loader import QDRANT_URL

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "static", "index.html")

log = logging.getLogger("okf-graphrag")

# Errors the caller can fix by loading a bundle or starting a store, as opposed
# to faults in this service. They are reported with 503 so that a health check
# or an alert rule can tell "my dependency is down" from "my code is broken".
UPSTREAM_STATUS = 503

# Shared secret required on every request when it is set. Unset means no
# authentication, which is the right default for the local single-user tool
# this mostly is — but see `main`, which refuses to bind a non-loopback address
# without one, because "no auth" and "reachable from the network" is the
# combination that turns a dev convenience into an exposure.
API_TOKEN = os.getenv("OKF_API_TOKEN", "").strip()

# Liveness stays open even when a token is set: a probe is usually a kubelet or
# a container healthcheck that has no way to carry one, and it reveals nothing
# but whether the process is up. Everything else, /metrics included, is behind
# the token — request counts and latencies describe usage.
UNAUTHENTICATED_PATHS = frozenset({"/healthz"})

# Latency buckets in seconds, cumulative upper bounds in the Prometheus sense.
# Chosen around the shape of this service: a warm hybrid search is ~0.3s, so
# the interesting resolution is between 100ms and 2s, and anything past 5s
# means a store is in trouble rather than the query being hard.
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


class _Metrics:
    """In-process counters and latency histogram, exposed in Prometheus format.

    Hand-rolled rather than pulling in `prometheus_client`: what is needed here
    is three counters and a histogram, and the text exposition format is a
    dozen lines to emit. A scrape-format dependency would be most of a megabyte
    to save that.

    Every method takes the lock. Requests are served from a thread pool, so
    `+= 1` on a shared dict is not safe without one.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = {}
        self.buckets: dict[float, int] = {bound: 0 for bound in LATENCY_BUCKETS}
        self.duration_sum = 0.0
        self.duration_count = 0
        self.searches = 0
        self.search_errors = 0

    def observe(self, route: str, status: int, seconds: float) -> None:
        with self._lock:
            key = (route, status)
            self.requests[key] = self.requests.get(key, 0) + 1
            self.duration_sum += seconds
            self.duration_count += 1
            for bound in LATENCY_BUCKETS:
                if seconds <= bound:
                    self.buckets[bound] += 1

    def searched(self, failed: bool) -> None:
        with self._lock:
            self.searches += 1
            self.search_errors += int(failed)

    def render(self) -> str:
        """The metrics in Prometheus text exposition format."""
        with self._lock:
            lines = [
                "# HELP okf_requests_total Requests by route and status.",
                "# TYPE okf_requests_total counter",
            ]
            for (route, status), count in sorted(self.requests.items()):
                lines.append(
                    f'okf_requests_total{{route="{route}",status="{status}"}} {count}'
                )

            lines += [
                "# HELP okf_request_duration_seconds Request latency.",
                "# TYPE okf_request_duration_seconds histogram",
            ]
            cumulative = 0
            for bound in LATENCY_BUCKETS:
                cumulative = self.buckets[bound]
                lines.append(
                    f'okf_request_duration_seconds_bucket{{le="{bound}"}} {cumulative}'
                )
            lines += [
                f'okf_request_duration_seconds_bucket{{le="+Inf"}} {self.duration_count}',
                f"okf_request_duration_seconds_sum {self.duration_sum:.6f}",
                f"okf_request_duration_seconds_count {self.duration_count}",
                "# HELP okf_searches_total Hybrid searches run.",
                "# TYPE okf_searches_total counter",
                f"okf_searches_total {self.searches}",
                "# HELP okf_search_errors_total Searches that returned an error.",
                "# TYPE okf_search_errors_total counter",
                f"okf_search_errors_total {self.search_errors}",
                "# HELP okf_model_loaded Whether the embedding model is resident.",
                "# TYPE okf_model_loaded gauge",
                f"okf_model_loaded {int(qdrant_loader._MODEL is not None)}",
            ]
            return "\n".join(lines) + "\n"


metrics = _Metrics()


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object.

    Plain text logs are readable by a person watching one process; a service
    emits them to something that has to parse them. Fields set by the request
    middleware ride along in `record.__dict__` and are merged in, so a search
    line carries its request id, route, status and duration without any
    string formatting to unpick later.
    """

    RESERVED = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self.RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _presented_token(request: Request) -> str:
    """The token the caller offered, by whichever of the three routes they used.

    A header for programmatic callers; a query parameter because the page is
    opened by pasting a URL into a browser, which cannot set headers on a
    top-level navigation.
    """
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key") or request.query_params.get("token") or ""


def is_loopback(host: str) -> bool:
    """Whether binding this address keeps the server on the local machine.

    A hostname that is not an IP literal cannot be resolved to an answer here
    without a DNS lookup, so it is treated as non-loopback: guessing wrong in
    that direction only ever asks for a token that was not strictly needed.
    """
    import ipaddress

    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def authorized(request: Request) -> bool:
    """Whether this request may proceed.

    `compare_digest` rather than `==` so that a wrong token takes the same time
    to reject however much of its prefix was right.
    """
    if not API_TOKEN:
        return True
    if request.url.path in UNAUTHENTICATED_PATHS:
        return True
    return secrets.compare_digest(_presented_token(request), API_TOKEN)


def _search(params: dict) -> dict:
    """Run one hybrid search from parsed query-string params.

    Takes the `parse_qs` shape — every value a list — so that the parsing rules
    stay in one place and stay testable without a server. Numbers that don't
    parse fall back to their default rather than rejecting the request: the
    sliders can only send valid values, so a bad one means a hand-edited URL,
    and degrading to the default is friendlier than a 422.

    Returns the response body as a dict. Every failure mode the CLI turns into
    a SystemExit becomes an {"error": ...} here, so the page can show the same
    "is Neo4j running?" guidance the terminal does.
    """
    query = (params.get("q", [""])[0]).strip()
    if not query:
        return {"query": "", "results": []}

    def number(name, default, cast=float):
        try:
            return cast(params.get(name, [default])[0])
        except (TypeError, ValueError):
            return default

    try:
        results = retrieval.hybrid_search(
            query,
            top_k=max(1, min(number("top_k", 10, int), 50)),
            vector_weight=number("vector_weight", retrieval.VECTOR_WEIGHT),
            graph_weight=number("graph_weight", retrieval.GRAPH_WEIGHT),
            exclude_index=params.get("include_index", ["0"])[0] != "1",
            degree_damping=number("degree_damping", retrieval.DEGREE_DAMPING),
        )
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is upstream
        message = _upstream_error(exc)
        if message is None:
            raise
        return {"error": message}

    return {"query": query, "results": [_with_passage(r, query) for r in results]}


def _answer(params: dict) -> dict:
    """Answer one question from parsed query-string params, with citations.

    Same shape and the same error contract as `_search`: an empty question is
    an empty answer, and a dependency that is down - a store, or OpenAI - is an
    {"error": ...} the page can show.
    """
    question = (params.get("q", [""])[0]).strip()
    if not question:
        return {"question": "", "answered": False, "answer": None,
                "cited": [], "uncited": False, "sources": []}
    try:
        return answer.ask(question)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is upstream
        message = _upstream_error(exc)
        if message is None:
            raise
        return {"error": message}


def _upstream_error(exc: Exception) -> str | None:
    """The message for a failure the operator can fix outside this service.

    Covers the stores, a missing bundle or API key (`LookupError`), and the
    OpenAI API. Anything else is a bug here and returns None, so the caller
    re-raises it into the incident-id path instead of mislabelling it.
    """
    if isinstance(exc, LookupError):
        return str(exc)
    if isinstance(exc, (ResponseHandlingException, ConnectionError)):
        return f"could not reach Qdrant at {QDRANT_URL} - is it running?"
    if isinstance(exc, UnexpectedResponse):
        return f"Qdrant rejected the query: {exc}"
    if isinstance(exc, ServiceUnavailable):
        return f"could not reach Neo4j at {NEO4J_URI} - is the database running?"
    if isinstance(exc, AuthError):
        return (
            f"Neo4j rejected the credentials for user '{NEO4J_USER}' - "
            "check NEO4J_USER and NEO4J_PASSWORD"
        )
    if isinstance(exc, ConfigurationError):
        return f"bad Neo4j configuration (NEO4J_URI={NEO4J_URI}): {exc}"
    try:
        import openai
    except ImportError:
        return None
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key - check OPENAI_API_KEY"
    if isinstance(exc, openai.APIConnectionError):
        return "could not reach the OpenAI API - check the network"
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI returned an error ({exc.status_code}): {exc.message}"
    return None


# Set by main() when a browser should be opened; the tab is launched from the
# lifespan hook rather than before uvicorn.run, because the model load below
# blocks for seconds and a tab opened first would only show a refused
# connection. None means "don't open one" — which is every non-CLI caller.
_BROWSE_URL: str | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model at startup and close the shared stores at shutdown.

    Paying the model load here rather than inside the first request means the
    server is slow to *start* instead of slow to answer, which is what a
    readiness check wants.
    """
    log.info("loading the embedding model...")
    retrieval._model()
    log.info("ready")

    if _BROWSE_URL:
        import webbrowser

        webbrowser.open(_BROWSE_URL)

    yield
    retrieval.close_clients()
    log.info("stopped")


app = FastAPI(title="okf-graphrag", lifespan=lifespan)


@app.middleware("http")
async def observe_and_authenticate(request: Request, call_next):
    """Time every request, log one structured line for it, and enforce the token.

    Auth lives here rather than in a dependency on each route so that a route
    added later is protected by default: forgetting to add a decorator is a
    silent hole, whereas forgetting to exempt a path is a visible 401.

    The request id is generated here and returned in `X-Request-Id`, so a user
    reporting a slow or failed search can quote the exact line in the log.
    """
    request_id = uuid.uuid4().hex[:12]
    route = request.url.path
    started = time.perf_counter()

    if not authorized(request):
        elapsed = time.perf_counter() - started
        metrics.observe(route, 401, elapsed)
        log.warning(
            "unauthorized",
            extra={"request_id": request_id, "route": route, "status": 401},
        )
        return JSONResponse(
            {"error": "unauthorized - supply the API token"},
            status_code=401,
            headers={"X-Request-Id": request_id},
        )

    response = await call_next(request)
    elapsed = time.perf_counter() - started

    metrics.observe(route, response.status_code, elapsed)
    response.headers["X-Request-Id"] = request_id
    log.info(
        "request",
        extra={
            "request_id": request_id,
            "route": route,
            "method": request.method,
            "status": response.status_code,
            "duration_ms": round(elapsed * 1000, 2),
        },
    )
    return response


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def page():
    """Serve the search page."""
    if not os.path.exists(PAGE):
        return JSONResponse({"error": "missing static/index.html"}, status_code=500)
    return FileResponse(PAGE, media_type="text/html; charset=utf-8")


@app.get("/api/search")
async def search(request: Request):
    """Run a hybrid search and return the ranked results as JSON.

    The raw query string is handed to `_search` in `parse_qs` shape rather than
    declared as typed parameters, so the lenient coercion documented there
    stays the single definition of how a request is read.
    """
    params: dict[str, list[str]] = {}
    for key, value in request.query_params.multi_items():
        params.setdefault(key, []).append(value)

    # Anything `_search` does not recognise would otherwise escape into the
    # server and close the connection with no response at all. Log it against
    # an id the user can quote, and return that id rather than the traceback.
    try:
        payload = _search(params)
    except Exception:  # noqa: BLE001 - deliberate catch-all at the boundary
        incident = uuid.uuid4().hex[:8]
        metrics.searched(failed=True)
        log.exception("search failed", extra={"incident": incident})
        return JSONResponse(
            {"error": f"internal error while searching (incident {incident})"},
            status_code=500,
        )

    metrics.searched(failed="error" in payload)
    if "error" in payload:
        # A missing collection or an unreachable store is an upstream problem,
        # not a bug in this service; 500 here would misroute every alert.
        return JSONResponse(payload, status_code=UPSTREAM_STATUS)
    return payload


@app.get("/api/answer")
def answer_question(request: Request):
    """Answer a question from retrieved passages and return it with its sources.

    A plain `def`, not `async def`: the model call takes seconds, and FastAPI
    runs sync routes on its thread pool, so one slow answer does not stall
    every other request on the event loop.
    """
    params: dict[str, list[str]] = {}
    for key, value in request.query_params.multi_items():
        params.setdefault(key, []).append(value)

    try:
        payload = _answer(params)
    except Exception:  # noqa: BLE001 - deliberate catch-all at the boundary
        incident = uuid.uuid4().hex[:8]
        log.exception("answer failed", extra={"incident": incident})
        return JSONResponse(
            {"error": f"internal error while answering (incident {incident})"},
            status_code=500,
        )

    if "error" in payload:
        return JSONResponse(payload, status_code=UPSTREAM_STATUS)
    return payload


@app.get("/healthz")
async def healthz():
    """Liveness probe: the process is up and the model is loaded."""
    return {"status": "ok", "model_loaded": qdrant_loader._MODEL is not None}


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Request counts and latencies in Prometheus text exposition format."""
    return PlainTextResponse(
        metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


def main() -> None:
    """Serve the search UI until interrupted."""
    global _BROWSE_URL
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument(
        "--text-logs",
        action="store_true",
        help="human-readable logs instead of one JSON object per line",
    )
    ap.add_argument(
        "--insecure",
        action="store_true",
        help="allow binding a non-loopback address with no OKF_API_TOKEN set",
    )
    args = ap.parse_args()

    if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
        raise SystemExit(
            "set NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD (e.g. in a .env file)"
        )

    if not is_loopback(args.host) and not API_TOKEN and not args.insecure:
        raise SystemExit(
            f"refusing to bind {args.host} with no authentication - this would "
            "expose the search API to everything that can reach this host. Set "
            "OKF_API_TOKEN to require a token, bind 127.0.0.1 to stay local, or "
            "pass --insecure if the network is genuinely trusted."
        )

    handler = logging.StreamHandler()
    if not args.text_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)

    url = f"http://{args.host}:{args.port}"
    print(f"okf-graphrag UI on {url}  (ctrl-c to stop)")
    if API_TOKEN:
        print("authentication is on - the browser needs ?token=... on the URL")
    if not args.no_browser:
        _BROWSE_URL = f"{url}/?token={API_TOKEN}" if API_TOKEN else url

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

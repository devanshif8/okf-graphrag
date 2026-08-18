# Contributing

Thanks for your interest in `okf-graphrag`. It's a small project with a single
maintainer, so the process is deliberately light.

## Context

`okf-graphrag` is a reference implementation of a consumer tool for
[OKF (Open Knowledge Format)](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf),
contributed to
[GoogleCloudPlatform/knowledge-catalog](https://github.com/GoogleCloudPlatform/knowledge-catalog).
See [Discussion #281](https://github.com/GoogleCloudPlatform/knowledge-catalog/discussions/281)
for why it exists and what it's trying to demonstrate — worth reading before
proposing anything that changes the scope.

## Development environment

Python 3.10 or newer.

```bash
git clone https://github.com/devanshif8/okf-graphrag.git
cd okf-graphrag
pip install -r requirements-dev.txt
```

That installs the runtime dependencies plus `pytest`. A virtual environment is
recommended but not required.

The tests need no services, but running the loaders against a real database
does. Copy the environment template and fill in your Neo4j credentials:

```bash
cp .env.example .env
```

## Running the tests

From the repo root:

```bash
pytest
```

`pyproject.toml` puts the repo root on `sys.path` and points pytest at
`tests/`, so this works with no install step and no arguments.

The suite covers the **pure logic** in all four modules — parsing, concept-id
derivation, property and payload coercion, and rank fusion. It needs no Neo4j,
no Qdrant and no embedding-model download, so it runs offline in seconds. CI
runs the same command on every push and pull request across Python 3.10–3.13
(`.github/workflows/tests.yml`).

## Testing the parts the suite doesn't cover

Anything that talks to a service is deliberately left out of the automated
suite rather than mocked. These functions need live instances and have to be
verified by hand:

| Function | Needs |
| --- | --- |
| `neo4j_loader.load_bundle_to_neo4j` | a running Neo4j |
| `qdrant_loader.load_bundle_to_qdrant` | a running Qdrant, plus the `all-MiniLM-L6-v2` model (downloaded on first run) |
| `retrieval.vector_search` | Qdrant with a loaded collection, plus the model |
| `retrieval.graph_expand` | Neo4j with a loaded bundle |
| `retrieval.hybrid_search` | all of the above |

If you change one of them, say in the PR description what you ran it against
and what you saw. The `ga4` bundle in the OKF `knowledge-catalog` repo is the
reference target — the README's Status section records the expected counts
(14 nodes, 21 relationships, 14 Qdrant points), so a manual run that matches
those is good evidence.

Neo4j connection details come from a git-ignored `.env` file in the repo root;
Qdrant reads `QDRANT_URL` and defaults to `http://localhost:6333`. See the
README's Setup section.

## Code style

There is **no linter or formatter configured** for this repo, and adding one
isn't a prerequisite for contributing. Match the style of the code already
there:

- Module-level docstring saying what the module is for; a one-line docstring on
  every function.
- Type hints on function signatures.
- Leading underscore for module-private helpers.
- Lines wrapped at roughly 88 characters, double-quoted strings.
- Comments explain *why*, not what. Several non-obvious decisions in this code
  are load-bearing (the RRF constants, the hub and degree damping, the UUIDv5
  point ids) and carry a comment saying what would break without them — keep
  that up if you touch them.

If you'd like a formatter adopted, open an issue first rather than bundling the
reformat into a functional PR — a repo-wide reformat is hard to review
alongside a real change. The same goes for any new dependency or framework:
check whether one is already in use before introducing another. The retrieval
layer is plain Python on purpose (see the README's Status section on why there
is no LangGraph).

## Submitting a pull request

No branch-naming convention, no commit-message convention, no CLA.

1. Fork the repo and create a branch.
2. Make your change, and add tests for it if it's pure logic.
3. Run `pytest` and make sure everything passes.
4. Open a PR against `master` with a clear description of what changed and
   why — plus what you verified manually, if it touches the service-backed
   code above.

Small PRs get reviewed faster than large ones. If you're planning something
substantial, open an issue first so we can agree on the shape of it before you
write it.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same license that covers the project.

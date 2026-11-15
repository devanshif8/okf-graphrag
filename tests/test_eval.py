"""Tests for the evaluation harness: metrics, book splitting, query sampling.

These cover the parts that decide whether a reported accuracy number means
anything — a wrong metric or a badly sampled query set produces a plausible
table that is simply false, which is worse than no measurement. Nothing here
needs Neo4j, Qdrant or a model download; the store-backed half of `evaluate.py`
is exercised by running it against a real bundle.
"""

import io
import json
import os
import sys

import pytest

import eval.evaluate as evaluate_module
import eval.judge as judge_module
import eval.mark as mark_module
from eval.evaluate import (
    _table,
    bootstrap_ci,
    compare_runs,
    sign_test,
    load_queries,
    ndcg_at,
    precision_at,
    recall_at,
    reciprocal_rank,
    run_query,
    score_ranking,
)
from eval.edge_coherence import assign_nodes, measure
from eval.fetch_library import filename, title_matches
from eval.make_domain_queries import book_of, book_domains, build_domain_queries
from eval.make_need_queries import build_need_queries, load_needs
from eval.make_queries import build_queries, sample_passages
from eval.split_books import (
    slugify,
    split_book,
    split_chapters,
    strip_boilerplate,
)

import random


PROSE = (
    "the lighthouse keeper walked the shingle beach counting gulls and mending "
    "the broken lamp housing before the winter storms arrived from the north "
) * 30


# --- metrics ---------------------------------------------------------------


def test_reciprocal_rank_is_one_over_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_relevant_retrieved():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0
    assert reciprocal_rank([], {"z"}) == 0.0


def test_recall_counts_only_within_the_cutoff():
    ranked = ["a", "b", "c", "d"]
    assert recall_at(ranked, {"a", "d"}, 1) == 0.5
    assert recall_at(ranked, {"a", "d"}, 10) == 1.0


def test_ndcg_discounts_by_position():
    assert ndcg_at(["a", "b"], {"a"}, 10) == 1.0
    # A single relevant document at rank 2 scores 1/log2(3).
    assert ndcg_at(["b", "a"], {"a"}, 10) == pytest.approx(0.6309, abs=1e-4)


def test_ndcg_normalizes_against_the_achievable_ideal():
    """Two relevant documents ranked first and second is a perfect ranking."""
    assert ndcg_at(["a", "b", "c"], {"a", "b"}, 10) == 1.0


def test_ndcg_is_zero_for_an_empty_relevant_set_rather_than_dividing_by_zero():
    assert ndcg_at(["a"], set(), 5) == 0.0


def test_score_ranking_reports_every_cutoff():
    scores = score_ranking(["x", "a"], {"a"})
    assert scores["recall@1"] == 0.0
    assert scores["recall@5"] == 1.0
    assert scores["mrr"] == 0.5


def test_precision_counts_relevant_within_the_cutoff():
    assert precision_at(["a", "x", "b", "y"], {"a", "b"}, 2) == 0.5
    assert precision_at(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5


def test_precision_counts_an_unfilled_slot_as_a_miss():
    """Two results returned at k=5 is not a precision of 1.0."""
    assert precision_at(["a", "b"], {"a", "b"}, 5) == 0.4


def test_score_ranking_reports_precision_at_every_cutoff():
    scores = score_ranking(["a", "x"], {"a"})
    assert scores["precision@1"] == 1.0
    assert scores["precision@5"] == 0.2


def test_table_adds_a_chance_row_only_under_precision():
    runs = {"vector": score_ranking(["a"], {"a"})}
    chance_row = _table(runs, chance=0.25).splitlines()[-1].split()
    assert chance_row[0] == "chance"
    # mrr, 3 recall, 3 precision, 3 ndcg
    assert chance_row[1:] == ["-"] * 4 + ["0.250"] * 3 + ["-"] * 3


def test_table_omits_the_chance_row_without_a_base_rate():
    runs = {"vector": score_ranking(["a"], {"a"})}
    assert "chance" not in _table(runs)


def _fake_retrieval(monkeypatch, vector_ids, graph_ids):
    """Replace both store-backed retrievers with fixed rankings."""
    seen = {}

    def vector_search(query, top_k):
        seen["top_k"] = top_k
        return [{"id": i, "score": 1.0} for i in vector_ids[:top_k]]

    def graph_expand(ids, exclude_index, degree_damping):
        return [{"id": i} for i in graph_ids]

    monkeypatch.setattr(evaluate_module, "vector_search", vector_search)
    monkeypatch.setattr(evaluate_module, "graph_expand", graph_expand)
    return seen


def test_run_query_hides_excluded_ids_from_every_ranking(monkeypatch):
    _fake_retrieval(monkeypatch, ["own1", "a", "own2", "b"], ["own2", "c"])
    rankings = run_query(
        "q", 10, [("hybrid", 1.0, 1.0)], 0.5, True, frozenset({"own1", "own2"})
    )
    for name in ("vector", "graph", "hybrid"):
        assert not {"own1", "own2"} & set(rankings[name])
    assert rankings["vector"] == ["a", "b"]


def test_run_query_overfetches_to_make_up_for_exclusions(monkeypatch):
    """Excluded ids crowd the top, so depth survivors need a deeper fetch."""
    seen = _fake_retrieval(monkeypatch, [], [])
    run_query("q", 20, [("hybrid", 1.0, 1.0)], 0.5, True, frozenset({"x", "y", "z"}))
    with_exclusions = seen["top_k"]
    run_query("q", 20, [("hybrid", 1.0, 1.0)], 0.5, True)
    assert with_exclusions == seen["top_k"] + 3


def test_run_query_caps_the_overfetch(monkeypatch):
    seen = _fake_retrieval(monkeypatch, [], [])
    run_query("q", 20, [("hybrid", 1.0, 1.0)], 0.5, True)
    baseline = seen["top_k"]
    huge = frozenset(f"id{n}" for n in range(5000))
    run_query("q", 20, [("hybrid", 1.0, 1.0)], 0.5, True, huge)
    assert seen["top_k"] == baseline + evaluate_module.MAX_EXCLUDE_OVERFETCH


def test_run_query_without_exclusions_truncates_to_depth(monkeypatch):
    _fake_retrieval(monkeypatch, [f"v{n}" for n in range(50)], [])
    rankings = run_query("q", 5, [("hybrid", 1.0, 1.0)], 0.5, True)
    assert rankings["vector"] == [f"v{n}" for n in range(5)]
    assert len(rankings["hybrid"]) == 5


def test_load_queries_rejects_a_line_missing_its_ground_truth(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({"query": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_queries(str(path))


def test_load_queries_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    entry = json.dumps({"query": "a", "relevant": ["b/c.md"]})
    path.write_text(f"{entry}\n\n{entry}\n", encoding="utf-8")
    assert len(load_queries(str(path))) == 2


# --- splitting -------------------------------------------------------------


def test_strip_boilerplate_removes_the_gutenberg_wrapper():
    text = (
        "front matter\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "real content\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "licence\n"
    )
    assert strip_boilerplate(text) == "real content"


def test_strip_boilerplate_leaves_a_book_without_the_wrapper_alone():
    assert strip_boilerplate("  just a book  ") == "just a book"


def test_split_chapters_splits_on_headings_and_drops_the_front_matter():
    text = f"title page\n\nCHAPTER I\n\n{PROSE}\n\nCHAPTER II\n\n{PROSE}\n"
    chapters = split_chapters(text)
    assert [title for title, _ in chapters] == ["CHAPTER I", "CHAPTER II"]
    assert "title page" not in chapters[0][1]


def test_split_chapters_ignores_a_table_of_contents():
    """TOC entries match the heading pattern but carry no body, so they drop."""
    text = f"CHAPTER I\nCHAPTER II\n\nCHAPTER I\n\n{PROSE}\n\nCHAPTER II\n\n{PROSE}\n"
    assert len(split_chapters(text)) == 2


def test_split_chapters_falls_back_to_chunking_without_headings():
    chapters = split_chapters(PROSE, chunk_words=200)
    assert len(chapters) > 1
    assert all(title.startswith("Part ") for title, _ in chapters)


def test_split_chapters_does_not_split_on_a_chapter_mentioned_in_prose():
    text = f"CHAPTER I\n\nhe read chapter 4 aloud and then {PROSE}\n\nCHAPTER II\n\n{PROSE}"
    assert len(split_chapters(text)) == 2


def test_split_book_prefixes_each_file_with_the_book_so_ids_stay_distinct(tmp_path):
    """The converter flattens the tree, so the book must be in the filename."""
    source = tmp_path / "moby_dick.txt"
    source.write_text(f"CHAPTER I\n\n{PROSE}\n\nCHAPTER II\n\n{PROSE}\n", encoding="utf-8")
    out = tmp_path / "out"

    assert split_book(str(source), str(out)) == 2

    names = sorted(os.listdir(out / "moby-dick"))
    assert all(name.startswith("moby-dick-") for name in names)
    assert names[0].startswith("moby-dick-001-")


def test_split_book_leads_each_chapter_with_book_and_chapter_title(tmp_path):
    source = tmp_path / "moby_dick.txt"
    source.write_text(f"CHAPTER I\n\n{PROSE}\n\nCHAPTER II\n\n{PROSE}\n", encoding="utf-8")
    out = tmp_path / "out"
    split_book(str(source), str(out))

    first = sorted((out / "moby-dick").iterdir())[0].read_text(encoding="utf-8")
    assert first.startswith("moby dick — CHAPTER I")


def test_slugify_is_filesystem_safe():
    assert slugify("Moby-Dick; or, The Whale") == "moby-dick-or-the-whale"
    assert slugify("!!!") == "untitled"


# --- query sampling --------------------------------------------------------


def test_sample_passages_draws_from_the_requested_region():
    words = ["alpha"] * 200 + ["omega"] * 200
    body = " ".join(words)
    rng = random.Random(0)

    head = sample_passages(body, 5, "head", rng)
    assert head and all(region == "head" for _, region in head)
    assert all("omega" not in passage for passage, _ in head)

    tail = sample_passages(body, 5, "tail", rng)
    assert tail and all("alpha" not in passage for passage, _ in tail)


def test_sample_passages_yields_no_tail_for_a_short_document():
    """A short document has nothing past the embedding window - it is skipped,
    not padded with a head sample mislabelled as tail."""
    body = " ".join(["alpha"] * 100)
    assert sample_passages(body, 3, "tail", random.Random(0)) == []


def test_sample_passages_splits_evenly_for_both():
    body = " ".join(["alpha"] * 600)
    regions = [region for _, region in sample_passages(body, 4, "both", random.Random(0))]
    assert regions.count("head") == 2
    assert regions.count("tail") == 2


def test_sample_passages_rejects_a_passage_that_is_not_prose():
    body = " ".join(["1.", "2.", "3.", "|", "-"] * 60)
    assert sample_passages(body, 3, "any", random.Random(0)) == []


def test_sample_passages_is_reproducible_for_a_seed():
    body = " ".join(f"word{n}" for n in range(400))
    first = sample_passages(body, 3, "any", random.Random(7))
    second = sample_passages(body, 3, "any", random.Random(7))
    assert first == second


def test_build_queries_keys_ground_truth_on_the_concept_id(tmp_path):
    bundle = tmp_path / "shelf"
    bundle.mkdir()
    (bundle / "chapter.md").write_text(
        f"---\ntitle: Chapter\n---\n\n{PROSE}\n", encoding="utf-8"
    )

    queries = build_queries(str(bundle), per_document=1, region="head")

    assert len(queries) == 1
    assert queries[0]["relevant"] == ["shelf/chapter.md"]
    assert queries[0]["query"] in PROSE


def test_build_queries_excludes_index_hubs(tmp_path):
    """An index file is structure, not content: a passage from one is not a
    question about a document."""
    bundle = tmp_path / "shelf"
    bundle.mkdir()
    (bundle / "index.md").write_text(f"{PROSE}\n", encoding="utf-8")
    (bundle / "chapter.md").write_text(
        f"---\ntitle: C\n---\n\n{PROSE}\n", encoding="utf-8"
    )

    ids = {q["relevant"][0] for q in build_queries(str(bundle), region="head")}
    assert ids == {"shelf/chapter.md"}


def test_build_queries_ignores_the_linkers_related_section(tmp_path):
    """Sampling a query out of '## Related' would test the linker's own output
    rather than the document's content."""
    bundle = tmp_path / "shelf"
    bundle.mkdir()
    (bundle / "chapter.md").write_text(
        f"---\ntitle: C\n---\n\n{PROSE}\n\n## Related\n\n- [zzzz](other.md)\n",
        encoding="utf-8",
    )

    queries = build_queries(str(bundle), per_document=3, region="any")
    assert queries
    assert all("zzzz" not in q["query"] for q in queries)


# --- the evaluation library ------------------------------------------------


def test_filename_drops_characters_windows_rejects():
    assert filename("Birds: A Guide?") == "Birds A Guide.txt"
    assert filename("Miss Parloa's New Cook Book") == "Miss Parloa's New Cook Book.txt"


def test_title_matches_accepts_a_subtitled_title():
    header = "Title: The Chemical History of a Candle, with notes\n"
    assert title_matches("The Chemical History of a Candle", header)


def test_title_matches_rejects_the_wrong_book():
    """A wrong Gutenberg id must not put a novel on the chemistry shelf."""
    assert not title_matches("The Chemical History of a Candle", "Title: Moby Dick\n")


def test_title_matches_rejects_a_file_with_no_title_header():
    assert not title_matches("Anything", "no header here")


def test_book_domains_keys_on_the_slug_split_books_writes():
    manifest = [{"gutenberg_id": "1", "domain": "cookery", "title": "Miss Parloa's New Cook Book"}]
    assert book_domains(manifest) == {"miss-parloa-s-new-cook-book": "cookery"}


def test_book_of_prefers_the_longest_matching_slug():
    """'ethics' must not claim the chapters of 'ethics-of-the-dust'."""
    slugs = ["ethics", "ethics-of-the-dust"]
    assert book_of("ethics-of-the-dust-001-part-1.md", slugs) == "ethics-of-the-dust"
    assert book_of("ethics-001-part-1.md", slugs) == "ethics"


def test_book_of_rejects_a_file_from_no_known_book():
    with pytest.raises(LookupError):
        book_of("stray-001.md", ["ethics"])


def _library(tmp_path):
    """A four-book, two-domain shelf with two chapters per book."""
    bundle = tmp_path / "shelf"
    bundle.mkdir()
    books = {"bread-book": "cookery", "soup-book": "cookery",
             "star-book": "astronomy", "moon-book": "astronomy"}
    for book in books:
        for n in (1, 2):
            (bundle / f"{book}-00{n}-part-{n}.md").write_text(
                f"---\ntitle: {book} {n}\n---\n\n{PROSE}\n", encoding="utf-8"
            )
    manifest = [
        {"gutenberg_id": str(i), "domain": domain, "title": book.replace("-", " ")}
        for i, (book, domain) in enumerate(books.items())
    ]
    return bundle, manifest


def test_domain_queries_count_other_books_in_the_domain_as_relevant(tmp_path):
    bundle, manifest = _library(tmp_path)
    queries = build_domain_queries(str(bundle), manifest, per_domain=2)

    query = next(q for q in queries if q["book"] == "bread-book")
    assert query["domain"] == "cookery"
    assert set(query["relevant"]) == {
        "shelf/soup-book-001-part-1.md", "shelf/soup-book-002-part-2.md"
    }


def test_domain_queries_exclude_the_querys_own_book(tmp_path):
    """Same-book hits measure repetition, not the domain."""
    bundle, manifest = _library(tmp_path)
    for query in build_domain_queries(str(bundle), manifest, per_domain=2):
        assert all(i.startswith(f"shelf/{query['book']}-") for i in query["exclude"])
        assert not set(query["exclude"]) & set(query["relevant"])


def test_domain_queries_are_balanced_across_domains(tmp_path):
    bundle, manifest = _library(tmp_path)
    queries = build_domain_queries(str(bundle), manifest, per_domain=3)
    assert sum(q["domain"] == "cookery" for q in queries) == 3
    assert sum(q["domain"] == "astronomy" for q in queries) == 3


def test_domain_queries_carry_the_random_ranking_base_rate(tmp_path):
    """8 chapters, 2 excluded: 2 of the remaining 6 are relevant."""
    bundle, manifest = _library(tmp_path)
    query = build_domain_queries(str(bundle), manifest, per_domain=1)[0]
    assert query["base_rate"] == pytest.approx(2 / 6)


def test_edge_coherence_separates_same_book_edges_from_the_real_claim(tmp_path):
    """Same-book edges are in-domain by construction and must not flatter the score."""
    _, manifest = _library(tmp_path)
    assignment = {
        "shelf/bread-book-001.md": ("bread-book", "cookery"),
        "shelf/bread-book-002.md": ("bread-book", "cookery"),
        "shelf/soup-book-001.md": ("soup-book", "cookery"),
        "shelf/star-book-001.md": ("star-book", "astronomy"),
    }
    edges = [
        ("shelf/bread-book-001.md", "shelf/bread-book-002.md"),  # same book
        ("shelf/bread-book-001.md", "shelf/soup-book-001.md"),   # cross-book, in domain
        ("shelf/bread-book-002.md", "shelf/star-book-001.md"),   # cross-book, leak
    ]
    result = measure(edges, assignment)

    assert result["edges"] == 3
    assert result["same_domain"] == pytest.approx(2 / 3)
    assert result["same_book"] == pytest.approx(1 / 3)
    assert result["cross_book"]["edges"] == 2
    assert result["cross_book"]["same_domain"] == pytest.approx(0.5)
    assert result["leaks"] == [{"domains": ["astronomy", "cookery"], "edges": 1}]


def test_edge_coherence_chance_follows_the_source_domain(tmp_path):
    """A big domain is easy to stay inside, so chance is resampled per source."""
    assignment = {
        "shelf/big-a.md": ("big", "large"),
        "shelf/big-b.md": ("big", "large"),
        "shelf/big2-a.md": ("big2", "large"),
        "shelf/small-a.md": ("small", "tiny"),
    }
    # One edge out of the large domain, one out of the tiny one.
    result = measure(
        [("shelf/big-a.md", "shelf/big2-a.md"), ("shelf/small-a.md", "shelf/big-a.md")],
        assignment,
    )
    # large: (3-1)/3; tiny: (1-1)/3.
    assert result["same_domain_chance"] == pytest.approx((2 / 3 + 0) / 2)
    # Cross-book chance drops the source's own book from the pool:
    # large (3-2)/(4-2); tiny (1-1)/(4-1).
    assert result["cross_book"]["chance"] == pytest.approx((0.5 + 0) / 2)


def test_edge_coherence_skips_edges_to_unassigned_nodes():
    """An index hub or another bundle's document is not a miss, it is not judged."""
    assignment = {
        "shelf/a-001.md": ("a", "cookery"),
        "shelf/b-001.md": ("b", "cookery"),
    }
    result = measure(
        [("shelf/a-001.md", "shelf/b-001.md"), ("shelf/a-001.md", "ga4/other.md")],
        assignment,
    )
    assert result["edges"] == 1
    assert result["skipped"] == 1
    assert result["same_domain"] == 1.0


def test_edge_coherence_drops_index_hubs(tmp_path):
    """A hub links its whole bundle by construction; judging it measures nothing."""
    bundle, manifest = _library(tmp_path)
    (bundle / "index.md").write_text(
        "[a](bread-book-001-part-1.md)\n", encoding="utf-8"
    )
    assignment = assign_nodes(str(bundle), manifest)
    assert "shelf/index.md" not in assignment
    assert len(assignment) == 8


def test_edge_coherence_needs_two_nodes_to_have_a_chance_baseline():
    with pytest.raises(ValueError):
        measure([], {"shelf/a-001.md": ("a", "cookery")})


def _needs_file(tmp_path, rows: str):
    path = tmp_path / "needs.csv"
    path.write_text("domains,need\n" + rows, encoding="utf-8")
    return str(path)


def test_need_queries_expand_a_domain_label_into_its_whole_shelf(tmp_path):
    bundle, manifest = _library(tmp_path)
    needs = load_needs(_needs_file(tmp_path, 'astronomy,"What causes an eclipse?"\n'))
    query = build_need_queries(str(bundle), manifest, needs)[0]

    assert query["query"] == "What causes an eclipse?"
    assert query["kind"] == "single"
    assert set(query["relevant"]) == {
        "shelf/star-book-001-part-1.md", "shelf/star-book-002-part-2.md",
        "shelf/moon-book-001-part-1.md", "shelf/moon-book-002-part-2.md",
    }
    assert query["base_rate"] == pytest.approx(4 / 8)


def test_need_queries_mark_a_multi_domain_need_cross_cutting(tmp_path):
    """Two shelves means twice the chance precision, so it cannot be read beside a single."""
    bundle, manifest = _library(tmp_path)
    needs = load_needs(
        _needs_file(tmp_path, 'cookery|astronomy,"How did sailors eat at sea?"\n')
    )
    query = build_need_queries(str(bundle), manifest, needs)[0]

    assert query["kind"] == "cross-cutting"
    assert query["domain"] == "cookery|astronomy"
    assert len(query["relevant"]) == 8
    assert query["base_rate"] == pytest.approx(1.0)


def test_need_queries_reject_an_unknown_domain_label(tmp_path):
    """A typo would score zero however well the need was answered."""
    bundle, manifest = _library(tmp_path)
    needs = load_needs(_needs_file(tmp_path, 'cookery|astronmy,"A typo."\n'))
    with pytest.raises(SystemExit, match="astronmy"):
        build_need_queries(str(bundle), manifest, needs)


def test_load_needs_skips_blank_rows_but_not_half_filled_ones(tmp_path):
    path = _needs_file(tmp_path, 'cookery,"A need."\n,\nlaw,"Another."\n')
    assert len(load_needs(path)) == 2

    with pytest.raises(SystemExit, match="both"):
        load_needs(_needs_file(tmp_path, 'cookery,\n'))


# --- hand judging ----------------------------------------------------------


def _sheet(tmp_path, rows: str, name: str = "j.csv"):
    path = tmp_path / name
    path.write_text("need,chapter,title,relevant,snippet\n" + rows, encoding="utf-8-sig")
    return str(path)


def test_pool_dedupes_across_runs_and_hides_which_run_found_what(monkeypatch):
    """One judgement per chapter, and no column a judge could read as a hint."""
    monkeypatch.setattr(
        judge_module, "run_query",
        lambda *a, **k: {"vector": ["a", "b"], "graph": ["b", "c"], "hybrid": ["b", "a"]},
    )
    texts = {i: {"title": i.upper(), "snippet": "..."} for i in "abc"}
    rows = judge_module.build_pool([{"query": "a need", "relevant": []}], texts, {})

    assert sorted(row["chapter"] for row in rows) == ["a", "b", "c"]
    assert set(rows[0]) == set(judge_module.FIELDS)


def test_pool_keeps_judgements_already_made(monkeypatch):
    """Editing the needs must not cost an afternoon of marking."""
    monkeypatch.setattr(judge_module, "run_query", lambda *a, **k: {"vector": ["a", "b"]})
    texts = {i: {"title": "", "snippet": ""} for i in "ab"}
    rows = judge_module.build_pool(
        [{"query": "a need", "relevant": []}], texts, {("a need", "a"): "1"}
    )
    marks = {row["chapter"]: row["relevant"] for row in rows}
    assert marks == {"a": "1", "b": ""}


def test_read_judgements_accepts_either_spelling_of_yes_and_no(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,1,s\nn,c2,t,yes,s\nn,c3,t,0,s\nn,c4,t,no,s\n")
    judged, blanks = judge_module.read_judgements(path)
    assert judged == {"n": {"c1", "c2"}}
    assert blanks == 0


def test_read_judgements_counts_blanks_rather_than_assuming_irrelevant(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,1,s\nn,c2,t,,s\n")
    judged, blanks = judge_module.read_judgements(path)
    assert judged == {"n": {"c1"}} and blanks == 1


def test_read_judgements_rejects_a_cell_that_is_neither(tmp_path):
    """A typo read as 'irrelevant' would lower a score with nothing to show for it."""
    path = _sheet(tmp_path, "n,c1,t,maybe,s\n")
    with pytest.raises(SystemExit, match="maybe"):
        judge_module.read_judgements(path)


def test_apply_judgements_replaces_the_shelf_with_the_judged_chapters():
    queries = [{"query": "n", "relevant": ["x", "y", "z"], "kind": "single"}]
    built = judge_module.apply_judgements(queries, {"n": {"y"}}, total=100)
    assert built[0]["relevant"] == ["y"]
    assert built[0]["base_rate"] == pytest.approx(0.01)
    assert built[0]["kind"] == "single"


def test_apply_judgements_drops_a_need_nothing_was_marked_for():
    """An unjudged need measures the judge, not the retriever."""
    queries = [{"query": "judged", "relevant": ["x"]}, {"query": "not", "relevant": ["x"]}]
    built = judge_module.apply_judgements(queries, {"judged": {"x"}}, total=10)
    assert [entry["query"] for entry in built] == ["judged"]


# --- the terminal marker ---------------------------------------------------


def test_marker_queues_only_unmarked_rows(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,1,s\nn,c2,t,,s\nn,c3,t,0,s\nn,c4,t,,s\n")
    rows = mark_module.load_rows(path)
    assert mark_module.pending(rows) == [1, 3]


def test_marker_can_revisit_marks_already_made(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,1,s\nn,c2,t,,s\n")
    rows = mark_module.load_rows(path)
    assert mark_module.pending(rows, revisit=True) == [0, 1]


def test_marker_keeps_the_sheets_row_order(tmp_path):
    """pool shuffled the rows to hide the ranking; reordering would give it back."""
    path = _sheet(tmp_path, "n,zebra,t,,s\nn,apple,t,,s\n")
    rows = mark_module.load_rows(path)
    queue = mark_module.pending(rows)
    assert [rows[index]["chapter"] for index in queue] == ["zebra", "apple"]


def test_marker_needs_cap_stops_at_whole_needs(tmp_path):
    """15 needs judged strictly beats 41 judged in a hurry."""
    path = _sheet(tmp_path, "n1,c1,t,,s\nn2,c2,t,,s\nn1,c3,t,,s\nn3,c4,t,,s\n")
    rows = mark_module.load_rows(path)
    queue = mark_module.pending(rows, needs=2)
    assert {rows[index]["need"] for index in queue} == {"n1", "n2"}
    assert len(queue) == 3


def test_marker_writes_a_sheet_judge_can_still_read(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,,snip\nn,c2,t,,snip\n")
    rows = mark_module.load_rows(path)
    rows[0]["relevant"] = "1"
    mark_module.save_rows(rows, path)

    judged, blanks = judge_module.read_judgements(path)
    assert judged == {"n": {"c1"}} and blanks == 1


def test_marker_save_leaves_no_temporary_file_behind(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,,s\n")
    mark_module.save_rows(mark_module.load_rows(path), path)
    assert not os.path.exists(f"{path}.tmp")


def _keys(monkeypatch, presses):
    """Feed the marker a fixed sequence of keypresses."""
    pressed = iter(presses)
    monkeypatch.setattr(mark_module, "getkey", lambda: next(pressed, "q"))


def test_marker_saves_after_every_keystroke(monkeypatch, tmp_path):
    """A closed terminal must not cost the afternoon, so the sheet is the state."""
    path = _sheet(tmp_path, "n,c1,t,,s\nn,c2,t,,s\nn,c3,t,,s\n")
    rows = mark_module.load_rows(path)
    _keys(monkeypatch, ["1", "0", "q"])
    mark_module._loop(rows, mark_module.pending(rows), path)

    marks = {row["chapter"]: row["relevant"] for row in mark_module.load_rows(path)}
    assert marks == {"c1": "1", "c2": "0", "c3": ""}


def test_marker_leaves_a_skipped_row_blank_rather_than_irrelevant(monkeypatch, tmp_path):
    """A blank is 'not judged'; a 0 is a judgement. Inferring one from the other lies."""
    path = _sheet(tmp_path, "n,c1,t,,s\nn,c2,t,,s\n")
    rows = mark_module.load_rows(path)
    _keys(monkeypatch, ["s", "1"])
    mark_module._loop(rows, mark_module.pending(rows), path)

    marks = {row["chapter"]: row["relevant"] for row in mark_module.load_rows(path)}
    assert marks == {"c1": "", "c2": "1"}


def test_marker_undo_restores_the_previous_cell_and_re_asks(monkeypatch, tmp_path):
    path = _sheet(tmp_path, "n,c1,t,,s\nn,c2,t,,s\n")
    rows = mark_module.load_rows(path)
    _keys(monkeypatch, ["1", "u", "0", "1", "q"])
    mark_module._loop(rows, mark_module.pending(rows), path)

    marks = {row["chapter"]: row["relevant"] for row in mark_module.load_rows(path)}
    assert marks == {"c1": "0", "c2": "1"}


def test_marker_undo_with_nothing_marked_yet_is_harmless(monkeypatch, tmp_path):
    path = _sheet(tmp_path, "n,c1,t,,s\n")
    rows = mark_module.load_rows(path)
    _keys(monkeypatch, ["u", "1", "q"])
    mark_module._loop(rows, mark_module.pending(rows), path)
    assert mark_module.load_rows(path)[0]["relevant"] == "1"


def test_marker_ignores_a_key_that_means_nothing(monkeypatch, tmp_path):
    """An unrecognised key must not be read as a judgement or skip the row."""
    path = _sheet(tmp_path, "n,c1,t,,s\n")
    rows = mark_module.load_rows(path)
    _keys(monkeypatch, ["x", "1", "q"])
    mark_module._loop(rows, mark_module.pending(rows), path)
    assert mark_module.load_rows(path)[0]["relevant"] == "1"


def test_marker_accepts_judges_spellings_of_yes_and_no(monkeypatch, tmp_path):
    path = _sheet(tmp_path, "n,c1,t,,s\nn,c2,t,,s\n")
    rows = mark_module.load_rows(path)
    _keys(monkeypatch, ["y", "n"])
    mark_module._loop(rows, mark_module.pending(rows), path)
    marks = {row["chapter"]: row["relevant"] for row in mark_module.load_rows(path)}
    assert marks == {"c1": "1", "c2": "0"}


def test_marker_rejects_a_sheet_missing_its_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("need,chapter\nn,c1\n", encoding="utf-8-sig")
    with pytest.raises(SystemExit, match="relevant"):
        mark_module.load_rows(str(path))


def test_marker_sees_no_console_behind_a_plain_stream(monkeypatch):
    """The trap this guards: on Windows a bare isatty() says True for these."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert mark_module.has_console() is False


def test_marker_refuses_to_run_without_a_console(monkeypatch):
    """Waiting forever on a key nobody can press looks like a slow judge."""
    monkeypatch.setattr(mark_module, "has_console", lambda: False)
    with pytest.raises(SystemExit, match="console"):
        mark_module.require_terminal()


def test_marker_starts_when_a_console_is_attached(monkeypatch):
    monkeypatch.setattr(mark_module, "has_console", lambda: True)
    assert mark_module.require_terminal() is None


def test_marker_tally_separates_blank_from_judged(tmp_path):
    path = _sheet(tmp_path, "n,c1,t,1,s\nn,c2,t,no,s\nn,c3,t,,s\n")
    assert mark_module.tally(mark_module.load_rows(path)) == {
        "rows": 3, "yes": 1, "no": 1, "blank": 1,
    }


def test_marker_render_shows_no_domain_or_retriever_hint():
    row = {"need": "why bread rises", "chapter": "shelf/x.md",
           "title": "Household Management 034", "relevant": "", "snippet": "yeast ferments"}
    screen = mark_module.render(row, 1, 18, 1, 41)
    assert "yeast ferments" in screen and "why bread rises" in screen
    assert "vector" not in screen and "hybrid" not in screen and "graph" not in screen


# --- error bars ------------------------------------------------------------


def test_bootstrap_interval_brackets_the_mean():
    values = [0.2, 0.4, 0.6, 0.8] * 10
    low, high = bootstrap_ci(values, resamples=500)
    assert low < sum(values) / len(values) < high


def test_bootstrap_interval_is_empty_when_every_query_agrees():
    """No spread in the sample means no uncertainty about its mean."""
    assert bootstrap_ci([0.5] * 20, resamples=200) == (0.5, 0.5)


def test_bootstrap_interval_narrows_as_queries_are_added():
    """The interval is a statement about the size of the query set."""
    few = bootstrap_ci([0.0, 1.0] * 5, resamples=1000)
    many = bootstrap_ci([0.0, 1.0] * 100, resamples=1000)
    assert (many[1] - many[0]) < (few[1] - few[0])


def test_sign_test_calls_a_coin_flip_insignificant():
    assert sign_test(5, 5) == 1.0
    assert sign_test(3, 4) > 0.5


def test_sign_test_finds_a_lopsided_split():
    """13 better, 3 worse is the shape the needs run actually produced."""
    assert sign_test(13, 3) == pytest.approx(0.021, abs=1e-3)


def test_sign_test_ignores_ties_by_construction():
    """Ties say nothing about direction, so they cannot be passed in at all."""
    assert sign_test(0, 0) == 1.0


def test_compare_runs_pairs_the_queries_rather_than_pooling_them():
    """A tiny but perfectly consistent gain is real; unpaired it would vanish."""
    baseline = [0.1, 0.5, 0.9] * 10
    treatment = [value + 0.05 for value in baseline]
    result = compare_runs(treatment, baseline, resamples=500)
    assert result["difference"] == pytest.approx(0.05)
    assert result["ci"][0] > 0.0
    assert (result["wins"], result["losses"], result["ties"]) == (30, 0, 0)


def test_compare_runs_reports_an_interval_across_zero_when_there_is_no_effect():
    values = [0.2, 0.7, 0.4, 0.9] * 5
    result = compare_runs(list(values), list(values), resamples=500)
    assert result["ci"] == [0.0, 0.0]
    assert result["p_value"] == 1.0
    assert result["ties"] == 20

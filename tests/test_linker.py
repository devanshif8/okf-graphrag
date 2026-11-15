"""Unit tests for linker.py — candidate generation, edge derivation, rewriting.

The embedding model is stubbed with a deterministic bag-of-words encoder, so the
suite needs no model download and stays offline like the rest of the tests. The
stub is only asked to rank phrases *within* one document; the parts that decide
which edges exist (`derive_links`) are pure and are tested directly.
"""

import math
import os
import random
import zlib

import numpy as np
import pytest

from linker import (
    MAX_LINKS,
    _mmr,
    apply_links,
    bundle_keyphrases,
    candidate_document_frequency,
    candidate_phrases,
    derive_links,
    document_vector,
    keyphrases,
    link_bundle,
    render_related,
)
from parser import parse_okf_file

DIM = 64


class FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    Embeds text as a normalized bag of words over hashed dimensions: texts that
    share words come out close, which is all the ranking logic needs.
    """

    def encode(self, texts, normalize_embeddings=False):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vectors = np.zeros((len(items), DIM))
        for row, text in enumerate(items):
            for word in text.lower().split():
                vectors[row, zlib.crc32(word.encode()) % DIM] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors[0] if single else vectors


def write_concept(path, title, body):
    """Write a minimal OKF file and return its path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"---\ntitle: {title}\ntype: Document\n---\n\n{body}\n")
    return path


# --- candidate generation ------------------------------------------------


def test_candidates_exclude_any_stopword():
    phrases = candidate_phrases("Training must cover password rotation for the employee.")

    assert "password rotation" in phrases
    # A candidate may not contain a function word anywhere, not just at its ends,
    # so 'must cover password' and 'for the employee' are both out.
    assert not any(
        word in {"for", "the", "must"} for p in phrases for word in p.split()
    )


def test_candidates_do_not_cross_sentence_or_heading_boundaries():
    phrases = candidate_phrases("# Scope\n\nQuarterly revenue. Freight costs rose.\n")

    assert "quarterly revenue" in phrases
    assert "freight costs" in phrases
    # Nothing may be stitched across the '.' or across the heading.
    assert "revenue freight" not in phrases
    assert "scope quarterly" not in phrases


def test_candidates_are_deduplicated_and_ordered():
    phrases = candidate_phrases("Audit log. Audit log.")

    assert phrases.count("audit log") == 1
    assert phrases[0] == "audit"


def test_candidates_drop_digits_and_stray_characters():
    phrases = candidate_phrases("Section 2119 x covers RFC compliance.")

    assert "2119" not in phrases
    assert "x" not in phrases
    assert "rfc compliance" in phrases


def test_empty_body_yields_no_candidates():
    assert candidate_phrases("   \n\n") == []


# --- per-document ranking ------------------------------------------------


def test_keyphrases_are_capped_and_free_of_contained_duplicates():
    body = (
        "Security training is mandatory. Phishing simulation runs quarterly. "
        "Password rotation applies to every contractor account."
    )

    phrases = keyphrases(body, FakeModel(), top_n=5)

    assert 0 < len(phrases) <= 5
    for phrase in phrases:
        assert not any(other != phrase and phrase in other for other in phrases)


def test_keyphrases_ignore_a_previously_written_related_section():
    """A prior run's link text must not become the next run's keyphrase."""
    source = "Security training is mandatory."
    linked = f"{source}\n\n## Related\n\n- [freight surcharge](freight.md)\n"

    assert keyphrases(linked, FakeModel()) == keyphrases(source, FakeModel())


def test_keyphrases_of_an_unphrased_body_is_empty():
    assert keyphrases("The and of to.", FakeModel()) == []


def test_bundle_keyphrases_skips_index_nodes():
    concepts = [
        {"path": "a.md", "body": "Retention policy applies.", "is_index": False},
        {"path": "index.md", "body": "- [a](a.md)", "is_index": True},
    ]

    mapping = bundle_keyphrases(concepts, FakeModel())

    assert set(mapping) == {"a.md"}


# --- edge derivation -----------------------------------------------------


def test_shared_phrase_links_both_directions():
    links = derive_links(
        {
            "a.md": ["security training", "badge access"],
            "b.md": ["security training", "phishing drill"],
            "c.md": ["freight cost"],
        }
    )

    assert links["a.md"] == [("security training", "b.md")]
    assert links["b.md"] == [("security training", "a.md")]
    assert links["c.md"] == []


def test_phrase_held_by_most_of_the_bundle_creates_no_edge():
    common = {f"doc{i}.md": ["the employee"] for i in range(4)}

    assert all(v == [] for v in derive_links(common).values())


def test_rarer_shared_phrase_outranks_a_common_one():
    keyphrase_map = {
        "a.md": ["annual leave", "cobra election"],
        "b.md": ["annual leave"],
        "c.md": ["annual leave"],
        "d.md": ["annual leave"],
        "e.md": ["cobra election"],
        "f.md": ["unrelated topic"],
    }

    links = derive_links(keyphrase_map, max_links=5)

    # 'annual leave' sits at df=4 of 6, over the 0.5 ratio, so it is ignored
    # entirely and only the df=2 phrase survives.
    assert links["a.md"] == [("cobra election", "e.md")]
    assert links["b.md"] == []


def test_max_links_caps_the_strongest_partners():
    keyphrase_map = {"hub.md": ["alpha term", "beta term", "gamma term"]}
    keyphrase_map.update({f"a{i}.md": ["alpha term"] for i in range(3)})
    keyphrase_map.update({f"b{i}.md": ["beta term"] for i in range(3)})
    keyphrase_map.update({f"g{i}.md": ["gamma term"] for i in range(3)})

    links = derive_links(keyphrase_map, max_links=2)

    assert len(links["hub.md"]) == 2
    assert all(len(v) <= 2 for v in links.values())


def test_single_document_bundle_has_no_edges():
    assert derive_links({"only.md": ["some phrase"]}) == {"only.md": []}


def test_derive_links_is_deterministic():
    keyphrase_map = {
        "a.md": ["shared phrase"],
        "b.md": ["shared phrase"],
        "c.md": ["shared phrase", "other phrase"],
        "d.md": ["other phrase"],
    }

    assert derive_links(keyphrase_map) == derive_links(keyphrase_map)


# --- rendering and rewriting --------------------------------------------


def test_related_links_are_relative_and_posix(tmp_path):
    source = str(tmp_path / "hr" / "leave.md")
    target = str(tmp_path / "finance" / "payroll.md")

    section = render_related(source, [("annual leave", target)])

    assert section.startswith("## Related\n")
    assert "- [annual leave](../finance/payroll.md)" in section
    assert "\\" not in section


def test_applied_links_round_trip_through_the_parser(tmp_path):
    path = write_concept(str(tmp_path / "a.md"), "A", "Security training body.")
    concept = parse_okf_file(path)

    apply_links(concept, [("security training", str(tmp_path / "b.md"))], ["security training"])

    reparsed = parse_okf_file(path)
    assert reparsed["links"] == [{"text": "security training", "target": "b.md"}]
    assert reparsed["frontmatter"]["keyphrases"] == ["security training"]
    assert reparsed["frontmatter"]["title"] == "A"
    assert "Security training body." in reparsed["body"]


def test_relinking_replaces_the_section_instead_of_stacking_it(tmp_path):
    path = write_concept(str(tmp_path / "a.md"), "A", "Body text.")

    apply_links(parse_okf_file(path), [("first", str(tmp_path / "b.md"))], ["first"])
    apply_links(parse_okf_file(path), [("second", str(tmp_path / "c.md"))], ["second"])

    reparsed = parse_okf_file(path)
    assert reparsed["body"].count("## Related") == 1
    assert reparsed["links"] == [{"text": "second", "target": "c.md"}]


def test_removing_every_edge_removes_the_section(tmp_path):
    path = write_concept(str(tmp_path / "a.md"), "A", "Body text.")

    apply_links(parse_okf_file(path), [("first", str(tmp_path / "b.md"))], ["first"])
    apply_links(parse_okf_file(path), [], [])

    body = parse_okf_file(path)["body"]
    assert "## Related" not in body
    assert body.strip() == "Body text."


def test_related_section_does_not_swallow_a_following_heading(tmp_path):
    path = write_concept(
        str(tmp_path / "a.md"),
        "A",
        "Body text.\n\n## Related\n\n- [old](old.md)\n\n## Notes\n\nKeep me.",
    )

    apply_links(parse_okf_file(path), [], [])

    body = parse_okf_file(path)["body"]
    assert "## Notes" in body and "Keep me." in body
    assert "old.md" not in body


# --- end to end over a bundle -------------------------------------------


@pytest.fixture
def bundle(tmp_path):
    """Three concepts: two that share a rare phrase, one that shares nothing."""
    write_concept(
        str(tmp_path / "training.md"), "Training", "Security training is mandatory."
    )
    write_concept(
        str(tmp_path / "phishing.md"),
        "Phishing",
        "Phishing drills are part of security training.",
    )
    write_concept(
        str(tmp_path / "revenue.md"), "Revenue", "Quarterly revenue rose twelve percent."
    )
    return str(tmp_path)


def test_link_bundle_writes_edges_the_loader_can_resolve(bundle):
    result = link_bundle(bundle, model=FakeModel())

    assert result["concepts"] == 3
    assert result["edges"] == 2

    linked = {
        os.path.basename(c["path"]): [link["target"] for link in c["links"]]
        for c in (parse_okf_file(os.path.join(bundle, n)) for n in os.listdir(bundle))
    }
    assert linked["training.md"] == ["phishing.md"]
    assert linked["phishing.md"] == ["training.md"]
    assert linked["revenue.md"] == []


def test_dry_run_reports_without_touching_the_files(bundle):
    before = {
        name: open(os.path.join(bundle, name), encoding="utf-8").read()
        for name in os.listdir(bundle)
    }

    result = link_bundle(bundle, model=FakeModel(), dry_run=True)

    assert result["edges"] == 2
    after = {
        name: open(os.path.join(bundle, name), encoding="utf-8").read()
        for name in os.listdir(bundle)
    }
    assert after == before


def test_link_bundle_is_idempotent(bundle):
    first = link_bundle(bundle, model=FakeModel())
    texts = {
        name: open(os.path.join(bundle, name), encoding="utf-8").read()
        for name in os.listdir(bundle)
    }

    second = link_bundle(bundle, model=FakeModel())

    assert second["edges"] == first["edges"]
    assert {
        name: open(os.path.join(bundle, name), encoding="utf-8").read()
        for name in os.listdir(bundle)
    } == texts


def test_empty_bundle_is_reported_not_crashed(tmp_path):
    assert link_bundle(str(tmp_path), model=FakeModel()) == {
        "concepts": 0,
        "phrases": 0,
        "edges": 0,
        "links": {},
    }


# --- edge derivation at scale --------------------------------------------


def _reference_derive_links(keyphrase_map, max_links=5, max_df_ratio=0.5):
    """The original O(N²) implementation, kept as an oracle.

    The fast version has to agree with this exactly on any input where the
    absolute df cap does not bite, or the rewrite changed the graph rather than
    just the cost of building it.

    One deliberate difference from the original: the sort key rounds the score.
    The original compared raw sums, so two partners with mathematically equal
    scores could order either way depending on the order floating-point addition
    happened to accumulate them in — a latent nondeterminism, not behaviour
    worth preserving. This oracle encodes the intended semantics, and
    `derive_links` rounds to match.
    """
    total = len(keyphrase_map)
    if total < 2:
        return {path: [] for path in keyphrase_map}

    holders = {}
    for path, phrases in keyphrase_map.items():
        for phrase in phrases:
            holders.setdefault(phrase, set()).add(path)

    max_df = max(2, math.floor(max_df_ratio * total))

    shared = {}
    for phrase, paths in holders.items():
        df = len(paths)
        if df < 2 or df > max_df:
            continue
        weight = math.log(total / df)
        for path in paths:
            for other in paths:
                if path != other:
                    shared.setdefault((path, other), []).append((weight, phrase))

    links = {path: [] for path in keyphrase_map}
    for path in keyphrase_map:
        partners = []
        for other in keyphrase_map:
            hits = shared.get((path, other))
            if hits:
                partners.append((sum(w for w, _ in hits), max(hits)[1], other))
        partners.sort(key=lambda p: (-round(p[0], 12), p[2]))
        links[path] = [(phrase, other) for _, phrase, other in partners[:max_links]]
    return links


def _random_keyphrase_map(seed, documents, vocabulary, per_document=6):
    rng = random.Random(seed)
    phrases = [f"phrase {n}" for n in range(vocabulary)]
    return {
        f"doc{n:04d}.md": rng.sample(phrases, per_document)
        for n in range(documents)
    }


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_original_implementation_exactly(seed):
    """Randomised equivalence: same edges, same order, same naming phrase."""
    keyphrase_map = _random_keyphrase_map(seed, documents=40, vocabulary=25)

    assert derive_links(keyphrase_map, max_links=5) == _reference_derive_links(
        keyphrase_map, max_links=5
    )


@pytest.mark.parametrize("seed", range(4))
def test_matches_the_original_on_a_sparse_corpus(seed):
    """A large vocabulary means most documents share nothing - the case the
    rewrite is built for, and the one where a grouping bug would hide."""
    keyphrase_map = _random_keyphrase_map(seed, documents=60, vocabulary=400)

    assert derive_links(keyphrase_map) == _reference_derive_links(keyphrase_map)


def test_the_absolute_cap_rejects_a_phrase_the_ratio_would_admit():
    # 300 documents, so the 0.5 ratio would allow a phrase held by 150 of them.
    # The absolute cap of 50 is what stops it.
    keyphrase_map = {f"doc{n}.md": ["common theme"] for n in range(300)}
    keyphrase_map["odd.md"] = ["common theme", "rare phrase"]
    keyphrase_map["pair.md"] = ["rare phrase"]

    links = derive_links(keyphrase_map, max_df_absolute=50)

    assert links["odd.md"] == [("rare phrase", "pair.md")]
    assert links["doc0.md"] == []


def test_the_absolute_cap_does_not_bite_on_a_small_bundle():
    """A bundle smaller than the cap must behave exactly as it always has."""
    keyphrase_map = {
        "a.md": ["shared phrase"],
        "b.md": ["shared phrase"],
        "c.md": ["other phrase"],
        "d.md": ["other phrase"],
    }

    assert derive_links(keyphrase_map) == _reference_derive_links(keyphrase_map)


def test_a_raised_absolute_cap_admits_more_phrases():
    keyphrase_map = {f"doc{n}.md": ["theme"] for n in range(8)}

    capped = derive_links(keyphrase_map, max_df_ratio=1.0, max_df_absolute=4)
    raised = derive_links(keyphrase_map, max_df_ratio=1.0, max_df_absolute=8)

    assert all(v == [] for v in capped.values())
    assert any(v for v in raised.values())


def test_edges_stay_symmetric():
    keyphrase_map = _random_keyphrase_map(1, documents=30, vocabulary=20)

    links = derive_links(keyphrase_map, max_links=100)
    partners = {path: {other for _, other in edges} for path, edges in links.items()}

    for path, others in partners.items():
        for other in others:
            assert path in partners[other]


def test_a_sparse_corpus_of_thousands_completes():
    """The regression this rewrite exists for.

    Three thousand documents sharing almost nothing was nine million dictionary
    lookups under the old grouping pass. Correct either way - but this is the
    size at which 'correct eventually' stopped being useful.
    """
    keyphrase_map = _random_keyphrase_map(0, documents=3000, vocabulary=20000)

    links = derive_links(keyphrase_map)

    assert len(links) == 3000
    assert all(len(edges) <= MAX_LINKS for edges in links.values())


def test_equal_scores_break_on_the_target_id_not_on_rounding_noise():
    """Two partners sharing phrases of equal weight score equally in exact
    arithmetic, but the sums accumulate in dictionary order and floating-point
    addition is not associative. Without rounding, which of them ranks first
    depends on that accumulation order rather than on the tie-break."""
    keyphrase_map = {
        "source.md": ["alpha one", "alpha two", "alpha three"],
        "zulu.md": ["alpha one", "alpha two", "alpha three"],
        "alfa.md": ["alpha one", "alpha two", "alpha three"],
    }

    links = derive_links(keyphrase_map, max_links=5, max_df_ratio=1.0)

    assert [other for _, other in links["source.md"]] == ["alfa.md", "zulu.md"]


# --- corpus-scale keyphrase extraction ------------------------------------


class CountingModel(FakeModel):
    """FakeModel that records how many texts it was asked to embed."""

    def __init__(self):
        self.encoded = []

    def encode(self, texts, normalize_embeddings=False):
        self.encoded.extend([texts] if isinstance(texts, str) else list(texts))
        return super().encode(texts, normalize_embeddings=normalize_embeddings)


def test_candidate_document_frequency_counts_documents_not_occurrences():
    frequency = candidate_document_frequency(
        {"a.md": ["shared", "shared", "only a"], "b.md": ["shared"]}
    )

    assert frequency["shared"] == 2
    assert frequency["only a"] == 1


def test_singleton_candidates_are_never_embedded():
    """A phrase no other document contains cannot produce an edge, so paying to
    embed it is the largest avoidable cost in the linker."""
    concepts = [
        {"path": "a.md", "body": "Security training matters. Unique alpha thing."},
        {"path": "b.md", "body": "Security training matters. Unique beta thing."},
    ]
    model = CountingModel()

    bundle_keyphrases(concepts, model)

    assert "security training" in model.encoded
    assert "alpha thing" not in model.encoded
    assert "beta thing" not in model.encoded


def test_a_shared_phrase_is_embedded_once_not_once_per_document():
    concepts = [
        {"path": f"doc{n}.md", "body": "Security training matters here."}
        for n in range(6)
    ]
    model = CountingModel()

    bundle_keyphrases(concepts, model)

    assert model.encoded.count("security training") == 1


def test_filtering_keeps_the_phrases_that_can_actually_link():
    concepts = [
        {"path": "a.md", "body": "Password rotation policy. Distinct alpha wording."},
        {"path": "b.md", "body": "Password rotation policy. Distinct beta wording."},
    ]

    mapping = bundle_keyphrases(concepts, FakeModel())

    assert set(mapping) == {"a.md", "b.md"}
    assert all("password rotation" in " ".join(v) for v in mapping.values())


def test_min_df_of_one_embeds_every_candidate():
    concepts = [{"path": "a.md", "body": "Unique alpha wording only."}]
    model = CountingModel()

    bundle_keyphrases(concepts, model, min_df=1)

    assert "alpha wording" in model.encoded


def test_a_corpus_where_nothing_is_shared_yields_no_keyphrases():
    """Nothing can link, so nothing is worth embedding - and the result is
    empty rather than an error."""
    concepts = [
        {"path": "a.md", "body": "Alpha wording here."},
        {"path": "b.md", "body": "Beta phrasing there."},
    ]

    assert bundle_keyphrases(concepts, FakeModel()) == {"a.md": [], "b.md": []}


def test_bundle_keyphrases_still_skips_index_nodes():
    concepts = [
        {"path": "a.md", "body": "Retention policy applies.", "is_index": False},
        {"path": "b.md", "body": "Retention policy applies too.", "is_index": False},
        {"path": "index.md", "body": "- [a](a.md)", "is_index": True},
    ]

    assert set(bundle_keyphrases(concepts, FakeModel())) == {"a.md", "b.md"}


def test_bundle_keyphrases_ignores_a_previous_related_section():
    plain = [
        {"path": "a.md", "body": "Security training matters."},
        {"path": "b.md", "body": "Security training matters."},
    ]
    linked = [
        {"path": "a.md", "body": "Security training matters.\n\n## Related\n\n- [freight surcharge](b.md)\n"},
        {"path": "b.md", "body": "Security training matters."},
    ]

    assert bundle_keyphrases(linked, FakeModel()) == bundle_keyphrases(plain, FakeModel())


class TestDocumentVector:
    def test_covers_text_past_the_embedding_window(self):
        """model.encode(body) truncates at ~180 words, so a keyphrase from late
        in a long document used to be ranked against a vector that had never
        seen it."""
        head = " ".join(["alpha"] * 400)
        tail = " ".join(["omega"] * 400)
        model = CountingModel()

        document_vector(f"{head} {tail}", model)

        # The tail reached the encoder as its own chunk rather than being cut.
        assert any("omega" in text for text in model.encoded)

    def test_is_a_unit_vector(self):
        """The MMR relevance term is a dot product standing in for a cosine, and
        the mean of unit vectors is not itself one."""
        vector = document_vector(" ".join(["alpha beta"] * 300), FakeModel())

        assert np.isclose(np.linalg.norm(vector), 1.0)

    def test_an_empty_body_does_not_divide_by_zero(self):
        assert np.all(np.isfinite(document_vector("", FakeModel())))


class TestMmrPool:
    def test_only_the_pool_takes_part_in_the_pairwise_matrix(self):
        """The quadratic term is capped, so a document with thousands of
        candidates does not build a thousands-square matrix."""
        rng = np.random.default_rng(0)
        phrases = [f"phrase {n}" for n in range(500)]
        vectors = rng.normal(size=(500, DIM))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        doc = vectors[7]

        selected = _mmr(doc, vectors, phrases, top_n=10, diversity=0.3, pool=50)

        assert len(selected) == 10
        assert len(set(selected)) == 10

    def test_the_pool_does_not_change_the_result_when_it_does_not_bite(self):
        rng = np.random.default_rng(1)
        phrases = [f"phrase {n}" for n in range(30)]
        vectors = rng.normal(size=(30, DIM))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        doc = vectors[3]

        assert _mmr(doc, vectors, phrases, 10, 0.3, pool=200) == _mmr(
            doc, vectors, phrases, 10, 0.3, pool=30
        )

    def test_selection_is_deterministic(self):
        rng = np.random.default_rng(2)
        phrases = [f"phrase {n}" for n in range(300)]
        vectors = rng.normal(size=(300, DIM))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        first = _mmr(vectors[0], vectors, phrases, 10, 0.3, pool=50)
        second = _mmr(vectors[0], vectors, phrases, 10, 0.3, pool=50)

        assert first == second

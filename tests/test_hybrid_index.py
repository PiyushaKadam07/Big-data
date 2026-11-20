import pytest

from hybrid_index import (Document, HybridIndexingSystem, build_demo_system, even_range_bounds,
                          tokenize, years_to_months)

QUERIES = [
    "distributed indexing nodes",
    "search engines inverted index",
    "NoSQL databases",
    "B-Tree indexes",
    "consistent hashing",
    "fault tolerance replication",
    "words that appear nowhere",
]


@pytest.fixture
def system():
    return build_demo_system(num_nodes=4)


def ids(result):
    return [(r.doc_id, round(r.score, 9)) for r in result["results"]]


def test_tokenizer_keeps_hyphens_and_drops_punctuation():
    assert tokenize("B-Tree indexes, key-value stores and real-time S3 data!") == [
        "b-tree", "indexes", "key-value", "stores", "real-time", "s3", "data"]


@pytest.mark.parametrize("match_all", [True, False])
@pytest.mark.parametrize("query", QUERIES)
def test_all_three_modes_return_the_same_ranked_results(system, query, match_all):
    hybrid = system.hybrid_search(query, match_all=match_all)
    assert (ids(hybrid) == ids(system.distributed_search(query, match_all=match_all))
            == ids(system.centralized_search(query, match_all=match_all)))


@pytest.mark.parametrize("query", QUERIES)
def test_hybrid_never_contacts_more_nodes_than_distributed(system, query):
    assert system.hybrid_search(query)["nodes_queried"] <= system.distributed_search(query)["nodes_queried"]


def test_routing_skips_nodes_without_the_terms(system):
    result = system.hybrid_search("NoSQL databases")
    assert result["nodes_queried"] == 1
    assert system.hybrid_search("words that appear nowhere")["nodes_queried"] == 0


def test_bm25_ranks_a_document_with_both_words_first():
    s = HybridIndexingSystem(num_nodes=2)
    s.add_document(Document("doc_0", "hashing is used in many places"))
    s.add_document(Document("doc_1", "consistent hashing spreads keys across nodes"))
    s.add_document(Document("doc_2", "consistent results need careful testing"))
    top = s.hybrid_search("consistent hashing")["results"]
    assert top[0].doc_id == "doc_1"
    assert [r.doc_id for r in top].count("doc_1") == 1


def test_rare_words_score_higher_than_common_ones():
    s = HybridIndexingSystem(num_nodes=2)
    for i in range(5):
        s.add_document(Document(f"doc_{i}", "data pipeline"))
    s.add_document(Document("doc_9", "data cassandra"))
    top = s.centralized_search("data cassandra")["results"]
    assert top[0].doc_id == "doc_9"


def test_top_k_limits_results_but_reports_all_matches(system):
    result = system.hybrid_search("distributed indexing nodes", top_k=3, match_all=False)
    assert len(result["results"]) == 3
    assert result["total_matches"] > 3


def test_topic_placement_keeps_a_topic_on_one_node_and_still_matches_centralized():
    s = build_demo_system(num_nodes=4, strategy="topic")
    homes = {}
    for node_id, node in s.nodes.items():
        for doc in node.documents.values():
            homes.setdefault(doc.metadata["category"], set()).add(node_id)
    assert all(len(nodes) == 1 for nodes in homes.values())
    for query in QUERIES:
        assert ids(s.hybrid_search(query)) == ids(s.centralized_search(query))


def test_unknown_placement_is_rejected():
    with pytest.raises(ValueError):
        HybridIndexingSystem(num_nodes=2, strategy="random")


def test_range_placement_needs_bounds():
    with pytest.raises(ValueError):
        HybridIndexingSystem(num_nodes=2, strategy="range")


def test_range_placement_gives_each_node_its_own_date_range():
    s = build_demo_system(num_nodes=4, strategy="range")
    spans = sorted(s.node_date_range(n) for n in s.nodes)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end < start  # no overlap between neighbouring nodes
    assert all(len(node.documents) == 5 for node in s.nodes.values())  # 20 documents, evenly split


def test_even_range_bounds_balance_the_nodes():
    months = [201901] * 10 + [202001] * 10 + [202101] * 10 + [202201] * 10
    assert even_range_bounds(months, 4) == [202001, 202101, 202201]


@pytest.mark.parametrize("strategy", ["hash", "topic", "range"])
@pytest.mark.parametrize("query", QUERIES)
def test_date_filtered_results_are_identical_in_every_mode(strategy, query):
    s = build_demo_system(num_nodes=4, strategy=strategy)
    months = years_to_months(2017, 2020)
    hybrid = s.hybrid_search(query, months=months)
    assert ids(hybrid) == ids(s.distributed_search(query, months=months)) == ids(s.centralized_search(query, months=months))
    for r in hybrid["results"]:
        assert months[0] <= s.get_document(r.doc_id).metadata["month"] <= months[1]


def test_range_placement_lets_a_date_search_skip_other_years():
    by_range = build_demo_system(num_nodes=4, strategy="range")
    by_hash = build_demo_system(num_nodes=4, strategy="hash")
    one_year = years_to_months(2016, 2016)
    query = "distributed search data indexes"
    assert by_range.hybrid_search(query, months=one_year, match_all=False)["nodes_queried"] == 1
    assert by_hash.hybrid_search(query, months=one_year, match_all=False)["nodes_queried"] > 1


def test_topic_placement_balances_documents_when_topic_sizes_differ():
    sizes = {"big": 50, "medium": 30, "small_a": 12, "small_b": 8}
    docs = [Document(f"doc_{t}_{i}", "text", {"category": t}) for t, n in sizes.items() for i in range(n)]
    s = HybridIndexingSystem.build(docs, num_nodes=2, strategy="topic")
    counts = sorted(len(node.documents) for node in s.nodes.values())
    assert counts == [50, 50]  # "big" alone, everything else together


def test_all_words_is_the_default_and_any_word_is_looser():
    s = HybridIndexingSystem(num_nodes=2)
    s.add_document(Document("doc_0", "graph neural networks for molecules"))
    s.add_document(Document("doc_1", "wireless networks and routing"))
    s.add_document(Document("doc_2", "neural networks for speech"))
    assert [r.doc_id for r in s.hybrid_search("graph neural networks")["results"]] == ["doc_0"]
    loose = {r.doc_id for r in s.hybrid_search("graph neural networks", match_all=False)["results"]}
    assert loose == {"doc_0", "doc_1", "doc_2"}


def test_all_words_lets_the_directory_skip_servers_missing_any_word():
    s = HybridIndexingSystem(num_nodes=2, strategy="topic")
    s.add_document(Document("doc_0", "graph neural networks", {"category": "ml"}))
    s.add_document(Document("doc_1", "wireless networks", {"category": "networking"}))
    assert s.hybrid_search("graph networks")["nodes_queried"] == 1
    assert s.hybrid_search("graph networks", match_all=False)["nodes_queried"] == 2

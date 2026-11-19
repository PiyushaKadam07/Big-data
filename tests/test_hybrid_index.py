import pytest

from hybrid_index import Document, HybridIndexingSystem, build_demo_system, tokenize

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


@pytest.mark.parametrize("query", QUERIES)
def test_all_three_modes_return_the_same_ranked_results(system, query):
    hybrid = system.hybrid_search(query)
    assert ids(hybrid) == ids(system.distributed_search(query)) == ids(system.centralized_search(query))


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
    result = system.hybrid_search("distributed indexing nodes", top_k=3)
    assert len(result["results"]) == 3
    assert result["total_matches"] > 3

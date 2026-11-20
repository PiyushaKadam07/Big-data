"""
Hybrid Indexing System: Centralized Metadata + Distributed Index Layer
CS63016 (Big Data), Kent State University - Piyusha Kadam

Three ways to search the same documents:
  centralized  - one index holds every document
  distributed  - documents are split across nodes; every query goes to every node
  hybrid       - documents are split across nodes; a central metadata index
                 sends each query only to the nodes that hold its terms

All three return the same ranked results (BM25). They differ in how many nodes
they contact and how much index data they read, which is what the benchmarks compare.
"""

import bisect
import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


# ─── Tokenizing ───────────────────────────────────────────────────────────────

# Common words that say nothing about a document's topic. Real abstracts are full of them.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "in", "of", "and", "or",
    "to", "for", "with", "on", "at", "by", "from", "as", "into", "that", "this", "these", "those",
    "it", "its", "we", "our", "us", "they", "their", "which", "who", "can", "could", "may", "has",
    "have", "had", "do", "does", "not", "but", "also", "such", "than", "then", "there", "both",
    "each", "other", "more", "most", "some", "any", "all", "how", "what", "when", "where", "while",
    "using", "based", "paper", "propose", "proposed", "show", "via",
}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase words; hyphenated words stay whole ("b-tree"), punctuation is dropped."""
    return [w for w in TOKEN_PATTERN.findall(text.lower()) if w not in STOPWORDS]


# ─── BM25 ─────────────────────────────────────────────────────────────────────
# BM25 is the ranking formula used by Lucene and Elasticsearch. These are their defaults.
BM25_K1 = 1.2   # how quickly repeating a word stops adding to the score
BM25_B = 0.75   # how much long documents are penalized


# A date range as (first month, last month), each written YYYYMM, e.g. (201901, 202112).
Months = tuple[int, int] | None


def years_to_months(first_year: int, last_year: int) -> tuple[int, int]:
    return first_year * 100 + 1, last_year * 100 + 12


@dataclass
class CorpusStats:
    """Collection-wide numbers BM25 needs. The metadata layer keeps them, so every
    node scores with the same numbers and all three modes rank identically."""
    doc_count: int
    avg_doc_len: float
    doc_freq: dict[str, int]  # term -> number of documents containing it

    def idf(self, term: str) -> float:
        """Rare words count more than common ones."""
        n = self.doc_freq.get(term, 0)
        return math.log(1 + (self.doc_count - n + 0.5) / (n + 0.5))


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class NodeSummary:
    node_id: str
    doc_count: int
    term_count: int
    load: float  # 0.0 to 1.0


@dataclass
class QueryResult:
    doc_id: str
    node_id: str
    score: float


# ─── Index Node ───────────────────────────────────────────────────────────────

class IndexNode:
    """One node of the index: an inverted index over the documents stored on it."""

    def __init__(self, node_id: str, network_delay_ms: float = 0.0):
        self.node_id = node_id
        # term -> {doc_id: how many times the term appears in that document}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self.documents: dict[str, Document] = {}
        self.doc_len: dict[str, int] = {}
        self.doc_month: dict[str, int] = {}  # doc_id -> YYYYMM, when the document has a date
        self.query_count = 0
        self.postings_scanned = 0
        # Optional, fixed cost per request to stand in for a network round trip.
        # 0 means only real work is measured.
        self.network_delay_ms = network_delay_ms

    def add_document(self, doc: Document):
        terms = tokenize(doc.content)
        self.documents[doc.doc_id] = doc
        self.doc_len[doc.doc_id] = len(terms)
        if "month" in doc.metadata:
            self.doc_month[doc.doc_id] = doc.metadata["month"]
        for term, tf in Counter(terms).items():
            self.inverted_index[term][doc.doc_id] = tf

    def search(self, terms: list[str], stats: CorpusStats, months: Months = None,
               match_all: bool = True) -> list[QueryResult]:
        """Score the local documents that match the query: by default they must contain
        every query term (match_all=False accepts any of them). If `months` is given,
        only documents dated within that range count."""
        if self.network_delay_ms:
            time.sleep(self.network_delay_ms / 1000)
        self.query_count += 1
        unique = list(dict.fromkeys(terms))
        allowed = None
        if match_all:
            # Start from the shortest list of documents and keep those that have every term.
            postings_by_term = sorted((self.inverted_index.get(t, {}) for t in unique), key=len)
            allowed = set(postings_by_term[0]) if postings_by_term else set()
            for postings in postings_by_term[1:]:
                allowed &= postings.keys()
            if not allowed:
                return []
        scores: dict[str, float] = defaultdict(float)
        for term in unique:
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            idf = stats.idf(term)
            self.postings_scanned += len(postings)
            for doc_id, tf in postings.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                if months and not months[0] <= self.doc_month.get(doc_id, 0) <= months[1]:
                    continue
                length_norm = 1 - BM25_B + BM25_B * self.doc_len[doc_id] / stats.avg_doc_len
                scores[doc_id] += idf * tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm)
        return [QueryResult(doc_id=d, node_id=self.node_id, score=s) for d, s in scores.items()]

    def get_summary(self) -> NodeSummary:
        return NodeSummary(
            node_id=self.node_id,
            doc_count=len(self.documents),
            term_count=len(self.inverted_index),
            load=min(self.query_count / 100.0, 1.0),
        )

    # Kept so older code calling IndexNode._tokenize still works.
    _tokenize = staticmethod(tokenize)


# ─── Central Metadata Index ───────────────────────────────────────────────────

class CentralMetadataIndex:
    """
    Lightweight central layer: which node holds each document, which nodes hold
    each term, and the collection statistics BM25 needs. It stores no documents.
    """

    def __init__(self):
        self.doc_to_node: dict[str, str] = {}
        self.term_to_nodes: dict[str, set[str]] = defaultdict(set)
        self.node_summaries: dict[str, NodeSummary] = {}
        self.doc_freq: Counter = Counter()
        self.total_terms = 0
        self.lookups = 0
        self.node_months: dict[str, list[int]] = {}  # node -> [earliest, latest] month it holds
        # term -> node -> the best score any single document on that node gets for the term.
        # Each node reports these to the directory; they're refreshed when documents are added.
        self.term_node_max: dict[str, dict[str, float]] = {}
        self.bounds_for: tuple | None = None  # (document count, average length) the maxima were computed for

    def register_document(self, doc: Document, node_id: str):
        terms = tokenize(doc.content)
        self.doc_to_node[doc.doc_id] = node_id
        self.total_terms += len(terms)
        month = doc.metadata.get("month")
        if month is not None:
            span = self.node_months.setdefault(node_id, [month, month])
            span[0], span[1] = min(span[0], month), max(span[1], month)
        for term in set(terms):
            self.term_to_nodes[term].add(node_id)
            self.doc_freq[term] += 1

    def update_summary(self, summary: NodeSummary):
        self.node_summaries[summary.node_id] = summary

    def refresh_bounds(self, nodes: dict[str, "IndexNode"], stats: CorpusStats):
        """Record, for every term on every node, the best score a single document there gets
        for that term (without the rarity factor, which is applied at search time)."""
        key = (stats.doc_count, stats.avg_doc_len)
        if self.bounds_for == key:
            return
        maxima: dict[str, dict[str, float]] = defaultdict(dict)
        for node_id, node in nodes.items():
            lengths = node.doc_len
            for term, postings in node.inverted_index.items():
                maxima[term][node_id] = max(
                    tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * lengths[d] / stats.avg_doc_len))
                    for d, tf in postings.items())
        self.term_node_max = maxima
        self.bounds_for = key

    def score_bound(self, node_id: str, terms: list[str], stats: CorpusStats) -> float:
        """The highest score any document on this node could get for these terms: each term's
        best single-document score added up. It can only overestimate (the best scores may come
        from different documents), so skipping a node whose bound can't beat the current results
        never changes them."""
        return sum(stats.idf(t) * self.term_node_max.get(t, {}).get(node_id, 0.0) for t in dict.fromkeys(terms))

    def stats(self) -> CorpusStats:
        n = len(self.doc_to_node)
        return CorpusStats(doc_count=n, avg_doc_len=(self.total_terms / n) if n else 1.0,
                           doc_freq=self.doc_freq)

    def route_query(self, terms: list[str], months: Months = None, match_all: bool = True) -> set[str]:
        """The smallest set of nodes that can hold a match. With match_all a node must hold
        every term (any of them otherwise); with a date range it must also hold documents
        from that range."""
        self.lookups += 1
        node_sets = [self.term_to_nodes.get(t, set()) for t in dict.fromkeys(terms)]
        if not node_sets:
            return set()
        candidates = set.intersection(*node_sets) if match_all else set.union(*node_sets)
        if months:
            candidates = {n for n in candidates
                          if n in self.node_months
                          and self.node_months[n][0] <= months[1] and self.node_months[n][1] >= months[0]}
        return candidates


# ─── Range boundaries ─────────────────────────────────────────────────────────

def even_range_bounds(months: list[int], num_nodes: int) -> list[int]:
    """Choose where each node's date range starts so the nodes hold about the same
    number of documents (the same idea as splitting an encyclopedia into volumes of
    similar size). Returns the first month of every node after the first."""
    ordered = sorted(months)
    if not ordered or num_nodes < 2:
        return []
    cuts = {ordered[i * len(ordered) // num_nodes] for i in range(1, num_nodes)}
    return sorted(c for c in cuts if c > ordered[0])


def balanced_topic_homes(docs: list[Document], node_ids: list[str]) -> dict[str, str]:
    """Give each topic a home node so the nodes end up with similar numbers of documents.
    Real collections have a few huge topics, so the biggest topics are placed first,
    each on whichever node currently holds the fewest documents."""
    sizes = Counter(d.metadata.get("category", "other") for d in docs)
    load = {n: 0 for n in node_ids}
    homes: dict[str, str] = {}
    for topic, count in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        node = min(node_ids, key=lambda n: (load[n], n))
        homes[topic] = node
        load[node] += count
    return homes


# ─── Hybrid Indexing System ───────────────────────────────────────────────────

class HybridIndexingSystem:
    """
    Holds the same documents three ways so the strategies can be compared fairly:
    one central index (centralized baseline), and partitioned nodes with a metadata
    layer (used by both distributed and hybrid search).
    """

    def __init__(self, num_nodes: int = 4, strategy: str = "hash", network_delay_ms: float = 0.0,
                 range_bounds: list[int] | None = None):
        """
        strategy:
          "hash"  - spread documents evenly at random
          "topic" - keep documents on the same topic together
          "range" - each node holds a range of dates; range_bounds gives the first month
                    (YYYYMM) of every node after the first, e.g. [201901, 202101, 202301]
        """
        if strategy not in ("hash", "topic", "range"):
            raise ValueError(f"strategy must be 'hash', 'topic' or 'range', not {strategy!r}")
        if strategy == "range":
            if not range_bounds or list(range_bounds) != sorted(set(range_bounds)):
                raise ValueError("range placement needs increasing range_bounds; see even_range_bounds()")
            if len(range_bounds) >= num_nodes:
                raise ValueError("range placement needs fewer bounds than nodes")
        self.strategy = strategy
        self.range_bounds = list(range_bounds or [])
        self.topic_home: dict[str, str] = {}  # topic -> node, used by the "topic" strategy
        self.network_delay_ms = network_delay_ms
        self.nodes: dict[str, IndexNode] = {
            f"node_{i}": IndexNode(f"node_{i}", network_delay_ms) for i in range(num_nodes)
        }
        self.node_ids = list(self.nodes.keys())
        self.metadata = CentralMetadataIndex()
        # Centralized baseline: a single index that holds every document.
        self.central = IndexNode("central", network_delay_ms)
        self.total_docs = 0
        self._pool = ThreadPoolExecutor(max_workers=num_nodes)
        self.stats = {
            "hybrid_queries": 0,
            "centralized_queries": 0,
            "distributed_queries": 0,
            "hybrid_latency_ms": [],
            "centralized_latency_ms": [],
            "distributed_latency_ms": [],
        }

    def _assign_node(self, doc: Document) -> str:
        if self.strategy == "range":
            # Like volumes of an encyclopedia: find the range this document's date falls in.
            month = doc.metadata.get("month", 0)
            return self.node_ids[bisect.bisect_right(self.range_bounds, month)]
        if self.strategy == "topic":
            # Keep documents on the same topic together: each topic gets a "home" node.
            # A topic seen for the first time goes to the node holding the fewest documents.
            topic = doc.metadata.get("category", "other")
            if topic not in self.topic_home:
                self.topic_home[topic] = min(self.node_ids, key=lambda n: (len(self.nodes[n].documents), n))
            return self.topic_home[topic]
        # "hash": spread documents evenly and at random by hashing their id
        h = int(hashlib.md5(doc.doc_id.encode()).hexdigest(), 16)
        return self.node_ids[h % len(self.node_ids)]

    @classmethod
    def build(cls, docs: list[Document], num_nodes: int = 4, strategy: str = "hash",
              network_delay_ms: float = 0.0) -> "HybridIndexingSystem":
        """Index a list of documents. For range placement the date ranges are chosen
        from the documents so every node gets about the same number of them."""
        bounds = None
        if strategy == "range":
            bounds = even_range_bounds([d.metadata.get("month", 0) for d in docs], num_nodes)
        system = cls(num_nodes=num_nodes, strategy=strategy, network_delay_ms=network_delay_ms,
                     range_bounds=bounds)
        if strategy == "topic":
            system.topic_home = balanced_topic_homes(docs, system.node_ids)
        for doc in docs:
            system.add_document(doc)
        return system

    def node_date_range(self, node_id: str) -> tuple[int, int] | None:
        """The earliest and latest month a node holds, or None if it holds no dated documents."""
        span = self.metadata.node_months.get(node_id)
        return (span[0], span[1]) if span else None

    def add_document(self, doc: Document):
        node_id = self._assign_node(doc)
        self.nodes[node_id].add_document(doc)
        self.metadata.register_document(doc, node_id)
        self.central.add_document(doc)
        self.total_docs += 1
        self.metadata.update_summary(self.nodes[node_id].get_summary())

    def get_document(self, doc_id: str) -> Document | None:
        return self.central.documents.get(doc_id)

    # ── search ──

    def _query_nodes(self, node_ids: list[str], terms: list[str], stats: CorpusStats,
                     months: Months = None, match_all: bool = True) -> list[QueryResult]:
        """Send the query to the given nodes at the same time and merge the answers."""
        if len(node_ids) <= 1:
            return [r for nid in node_ids for r in self.nodes[nid].search(terms, stats, months, match_all)]
        futures = [self._pool.submit(self.nodes[nid].search, terms, stats, months, match_all)
                   for nid in node_ids]
        return [r for f in futures for r in f.result()]

    def _finish(self, mode: str, results: list[QueryResult], nodes_queried: int,
                t0: float, top_k: int) -> dict:
        results.sort(key=lambda r: (-r.score, r.doc_id))
        elapsed = (time.perf_counter() - t0) * 1000
        self.stats[f"{mode}_queries"] += 1
        self.stats[f"{mode}_latency_ms"].append(elapsed)
        return {"results": results[:top_k], "total_matches": len(results),
                "nodes_queried": nodes_queried, "total_nodes": len(self.nodes),
                "latency_ms": elapsed, "mode": mode}

    def hybrid_search(self, query: str, top_k: int = 10, months: Months = None,
                      match_all: bool = True, prune: bool = True) -> dict:
        """
        1. Ask the metadata index which nodes can have matches at all (words, dates).
        2. With prune=True, also skip nodes that can't change the top results: ask the
           most promising node first, then only the nodes whose best possible score could
           still beat the k-th result so far. The top results are exactly the same; only
           the total number of matches is no longer known.
        """
        terms = tokenize(query)
        t0 = time.perf_counter()
        stats = self.metadata.stats()
        targets = sorted(self.metadata.route_query(terms, months, match_all))
        skipped_by_score: list[str] = []
        if prune and top_k and len(targets) > 1:
            self.metadata.refresh_bounds(self.nodes, stats)
            bounds = {n: self.metadata.score_bound(n, terms, stats) for n in targets}
            order = sorted(targets, key=lambda n: (-bounds[n], n))
            results = self.nodes[order[0]].search(terms, stats, months, match_all)
            scores = sorted((r.score for r in results), reverse=True)
            kth = scores[top_k - 1] if len(scores) >= top_k else None
            rest = [n for n in order[1:] if kth is None or bounds[n] >= kth - 1e-9]
            skipped_by_score = [n for n in order[1:] if n not in rest]
            results += self._query_nodes(rest, terms, stats, months, match_all)
            asked = [order[0]] + rest
        else:
            asked = targets
            results = self._query_nodes(asked, terms, stats, months, match_all)
        out = self._finish("hybrid", results, len(asked), t0, top_k)
        out["asked"] = sorted(asked)
        out["skipped_by_score"] = sorted(skipped_by_score)
        out["total_is_exact"] = not skipped_by_score
        return out

    def distributed_search(self, query: str, top_k: int = 10, months: Months = None,
                           match_all: bool = True) -> dict:
        """Broadcast the query to every node, with no routing."""
        terms = tokenize(query)
        t0 = time.perf_counter()
        results = self._query_nodes(self.node_ids, terms, self.metadata.stats(), months, match_all)
        return self._finish("distributed", results, len(self.node_ids), t0, top_k)

    def centralized_search(self, query: str, top_k: int = 10, months: Months = None,
                           match_all: bool = True) -> dict:
        """One index holding every document."""
        terms = tokenize(query)
        t0 = time.perf_counter()
        results = self.central.search(terms, self.metadata.stats(), months, match_all)
        return self._finish("centralized", results, 1, t0, top_k)

    # ── reporting ──

    def get_node_stats(self) -> list[dict]:
        return [
            {
                "node_id": nid,
                "doc_count": len(node.documents),
                "term_count": len(node.inverted_index),
                "query_count": node.query_count,
                "load": round(node.get_summary().load * 100, 1),
            }
            for nid, node in self.nodes.items()
        ]

    def avg_latency(self, mode: str) -> float:
        vals = self.stats.get(f"{mode}_latency_ms", [])
        return round(sum(vals) / len(vals), 2) if vals else 0.0


# ─── Sample Dataset Generator ─────────────────────────────────────────────────

SAMPLE_DOCS = [
    ("search engines use distributed inverted indexes for efficient keyword retrieval", {"category": "search"}),
    ("NoSQL databases handle large scale data with partitioned storage", {"category": "database"}),
    ("Apache Spark enables distributed processing of big data workloads", {"category": "processing"}),
    ("consistent hashing distributes load evenly across cluster nodes", {"category": "systems"}),
    ("inverted index maps terms to document identifiers for fast search", {"category": "search"}),
    ("Elasticsearch provides distributed full text search with sharding", {"category": "search"}),
    ("B-Tree indexes are common in centralized relational databases", {"category": "database"}),
    ("query routing reduces network overhead in distributed index systems", {"category": "systems"}),
    ("metadata layer stores node summaries and partition mappings", {"category": "systems"}),
    ("IoT devices generate high velocity data streams requiring real time indexing", {"category": "iot"}),
    ("recommendation systems need fast lookup of user and item features", {"category": "ml"}),
    ("AdTech platforms use real time indexing for contextual targeting", {"category": "adtech"}),
    ("distributed hash tables provide scalable key value lookups", {"category": "systems"}),
    ("Apache Cassandra uses consistent hashing for partitioned indexing", {"category": "database"}),
    ("Google Bigtable offers distributed storage with row key indexing", {"category": "database"}),
    ("range partitioning splits data by key range across nodes", {"category": "systems"}),
    ("load balancing ensures even distribution across index nodes", {"category": "systems"}),
    ("parallel query execution improves throughput in distributed systems", {"category": "processing"}),
    ("fault tolerance requires replication of index partitions", {"category": "systems"}),
    ("social media platforms index billions of posts for fast retrieval", {"category": "search"}),
]


def sample_documents() -> list[Document]:
    """The sample sentences as documents, each given an example date between 2015 and 2024."""
    return [Document(doc_id=f"doc_{i:03d}", content=content,
                     metadata={**meta, "month": (2015 + i % 10) * 100 + 1 + (i * 5) % 12})
            for i, (content, meta) in enumerate(SAMPLE_DOCS)]


def build_demo_system(num_nodes: int = 4, strategy: str = "hash") -> HybridIndexingSystem:
    return HybridIndexingSystem.build(sample_documents(), num_nodes=num_nodes, strategy=strategy)


if __name__ == "__main__":
    print("=== Hybrid Indexing System Demo ===\n")
    system = build_demo_system(num_nodes=4)

    queries = ["distributed indexing nodes", "search engines inverted index", "NoSQL databases"]
    for q in queries:
        print(f"\nQuery: '{q}'")
        h = system.hybrid_search(q)
        d = system.distributed_search(q)
        c = system.centralized_search(q)
        print(f"  Hybrid:       {h['nodes_queried']}/{h['total_nodes']} nodes, {h['latency_ms']:.1f}ms, {len(h['results'])} results")
        print(f"  Distributed:  {d['nodes_queried']}/{d['total_nodes']} nodes, {d['latency_ms']:.1f}ms, {len(d['results'])} results")
        print(f"  Centralized:  {c['nodes_queried']}/{c['total_nodes']} nodes, {c['latency_ms']:.1f}ms, {len(c['results'])} results")

    print("\n=== Node Stats ===")
    for s in system.get_node_stats():
        print(f"  {s['node_id']}: docs={s['doc_count']}, terms={s['term_count']}, queries={s['query_count']}")

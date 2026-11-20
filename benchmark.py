"""
Measure how many servers each approach contacts on real arXiv papers.

Needs the sample made by load_arxiv.py (data/arxiv_sample.jsonl). Run:
    python benchmark.py
    python benchmark.py --servers 8 --top-k 10

For each placement it runs 30 searches, plus the same searches limited to one year, and
reports the average number of servers contacted. It also checks that hybrid search returns
exactly the same top results as centralized search every time.
"""

import argparse
import json
import os
import random
import statistics
import time

from hybrid_index import Document, HybridIndexingSystem, years_to_months

QUERIES = [
    "graph neural networks", "differential privacy", "reinforcement learning", "image segmentation",
    "speech recognition", "federated learning", "adversarial examples", "query optimization",
    "robot grasping", "machine translation", "object detection", "knowledge graph",
    "recommender systems", "blockchain consensus", "code generation", "autonomous driving",
    "sentiment analysis", "transformer attention", "anomaly detection", "edge computing",
    "quantum circuits", "malware detection", "motion planning", "question answering",
    "time series forecasting", "point clouds", "language models", "software testing",
    "social networks", "wireless channel estimation",
]


def load_papers(path: str) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        papers = [json.loads(line) for line in f]
    return [Document(p["id"], f"{p['title']}. {p['abstract']}",
                     {"category": p["category"], "month": p.get("month", 0)}) for p in papers]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.path.join("data", "arxiv_sample.jsonl"))
    parser.add_argument("--servers", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    docs = load_papers(args.data)
    years = sorted({d.metadata["month"] // 100 for d in docs if d.metadata["month"]})
    recent = [y for y in years if y >= years[-1] - 9] or years
    rng = random.Random(1)
    one_year = [years_to_months(y, y) for y in (rng.choice(recent) for _ in QUERIES)]
    print(f"{len(docs):,} papers, {args.servers} servers, top {args.top_k} results, {len(QUERIES)} searches\n")

    rows, all_identical = [], True
    for strategy, label in [("hash", "Random"), ("topic", "By topic"), ("range", "By date")]:
        t0 = time.time()
        system = HybridIndexingSystem.build(docs, num_nodes=args.servers, strategy=strategy)
        build_s = time.time() - t0
        results = {"words": [], "words_pruned": [], "year": [], "year_pruned": []}
        for query, months in zip(QUERIES, one_year):
            for key, kwargs in [("words", {}), ("year", {"months": months})]:
                plain = system.hybrid_search(query, top_k=args.top_k, prune=False, **kwargs)
                pruned = system.hybrid_search(query, top_k=args.top_k, **kwargs)
                central = system.centralized_search(query, top_k=args.top_k, **kwargs)
                same = [r.doc_id for r in pruned["results"]] == [r.doc_id for r in central["results"]]
                all_identical &= same
                results[key].append(plain["nodes_queried"])
                results[key + "_pruned"].append(pruned["nodes_queried"])
        avg = {k: statistics.mean(v) for k, v in results.items()}
        rows.append((label, avg, build_s))

    print("Average servers contacted per search (distributed always contacts "
          f"{args.servers}):\n")
    print("| Placement | Word search | + skip by score | One-year search | + skip by score |")
    print("| --- | --- | --- | --- | --- |")
    for label, a, _ in rows:
        print(f"| {label} | {a['words']:.1f} | {a['words_pruned']:.1f} | {a['year']:.1f} | {a['year_pruned']:.1f} |")
    print()
    for label, _, build_s in rows:
        print(f"{label}: index built in {build_s:.1f} s")
    print("\nTop results identical to centralized search:", "yes, every time" if all_identical else "NO")


if __name__ == "__main__":
    main()

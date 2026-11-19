"""
Make a sample of computer science papers from the arXiv dataset on Kaggle.

1. Download the dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
   (you only need the file arxiv-metadata-oai-snapshot.json, about 4-5 GB unzipped)
2. Run:
       python load_arxiv.py path/to/arxiv-metadata-oai-snapshot.json --size 20000
3. The sample is saved to data/arxiv_sample.jsonl and the app picks it up automatically.

The file has one paper per line, so it's read line by line and never loaded into memory all at once.
"""

import argparse
import json
import os
import random

OUTPUT = os.path.join("data", "arxiv_sample.jsonl")


def read_cs_papers(path: str):
    """Yield computer science papers (primary category starts with "cs.")."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            categories = paper.get("categories", "").split()
            if not categories or not categories[0].startswith("cs."):
                continue
            abstract = " ".join(paper.get("abstract", "").split())
            title = " ".join(paper.get("title", "").split())
            if not abstract or not title:
                continue
            yield {
                "id": paper["id"],
                "title": title,
                "abstract": abstract,
                "category": categories[0],
                "authors": paper.get("authors", ""),
                "date": paper.get("update_date", ""),
            }


def sample(papers, size: int, seed: int = 42) -> list[dict]:
    """Pick `size` papers uniformly at random in one pass (reservoir sampling),
    so the result isn't biased toward the oldest papers at the top of the file."""
    rng = random.Random(seed)
    chosen: list[dict] = []
    for seen, paper in enumerate(papers, 1):
        if len(chosen) < size:
            chosen.append(paper)
        else:
            j = rng.randrange(seen)
            if j < size:
                chosen[j] = paper
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", help="path to arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--size", type=int, default=20000, help="number of papers to keep (default 20000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading {args.snapshot} (this takes a few minutes for the full file)...")
    papers = sample(read_cs_papers(args.snapshot), args.size, args.seed)
    papers.sort(key=lambda p: p["id"])
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(papers)} papers to {OUTPUT}")


if __name__ == "__main__":
    main()

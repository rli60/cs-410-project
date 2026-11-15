"""Compare TF-IDF vs BM25, with and without title text, on hand-labeled queries.
Needs ranking.py in the same folder. Usage:
  python3 evaluate.py --data ../data/ml-latest-small
  python3 evaluate.py --cleaned ../cs-410-project/data/movies_cleaned.csv --min-ratings 100
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))  # repo layout: evaluation/ next to backend/
from ranking import Index, load_docs, load_cleaned, precision_recall_at_k

def run(docs, queries, ks):
    idx = Index(docs)
    out = {}
    for method in ("tfidf", "bm25"):
        search = idx.search_tfidf if method == "tfidf" else idx.search_bm25
        per_q = []
        for item in queries:
            rel = set(item["relevant"]) & set(docs)  # only count relevant movies present in this data
            ranked = [m for m, _ in search(item["q"], max(ks))]
            row = {"q": item["q"], "n_rel": len(rel)}
            for k in ks:
                p, r = precision_recall_at_k(ranked, rel, k)
                row[f"P@{k}"], row[f"R@{k}"] = p, r
            per_q.append(row)
        out[method] = per_q
    return out

def avg(rows, key):
    vals = [r[key] for r in rows if r["n_rel"] > 0]
    return sum(vals) / len(vals) if vals else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder with movies.csv/tags.csv (ml-latest-small)")
    ap.add_argument("--cleaned", help="path to movies_cleaned.csv")
    ap.add_argument("--min-ratings", type=int, default=0)
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_queries.json"))
    ap.add_argument("--per-query", action="store_true", help="also print every query")
    a = ap.parse_args()
    if not (a.data or a.cleaned):
        ap.error("give --data or --cleaned")
    queries = json.load(open(a.queries))["queries"]
    ks = (5, 10)
    print(f"{len(queries)} labeled queries, K = {ks}\n")
    print(f"{'config':<22}" + "".join(f"{m:>9}" for m in ("P@5", "R@5", "P@10", "R@10")))
    for use_title in (True, False):
        docs = (load_cleaned(a.cleaned, a.min_ratings, use_title) if a.cleaned else load_docs(a.data, use_title))
        res = run(docs, queries, ks)
        for method in ("tfidf", "bm25"):
            rows = res[method]
            name = f"{method} {'with' if use_title else 'no'} title"
            print(f"{name:<22}" + "".join(f"{avg(rows, k):>9.3f}" for k in ("P@5", "R@5", "P@10", "R@10")))
            if a.per_query:
                for r in rows:
                    print(f"   {r['q'][:40]:<42} rel={r['n_rel']}  P@10={r['P@10']:.2f}  R@10={r['R@10']:.2f}")
    print("\nrelevant sets are hand-picked judgments; averages skip queries with 0 relevant in the data.")

if __name__ == "__main__":
    main()

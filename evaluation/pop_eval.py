"""Blend BM25 text score with a popularity prior, sweep the weight, score with P@K / R@K.
final = bm25/max_bm25_in_pool + w * prior   (rerank the top POOL BM25 hits)
prior "count" = log(1+rating_count)/log(1+max_count)
prior "count_avg" = count prior * (average_rating / 5)
Usage:
  python3 pop_eval.py --data ../data/ml-latest-small
  python3 pop_eval.py --cleaned ../cs-410-project/data/movies_cleaned.csv
Needs ranking.py and eval_queries.json in the same folder.
"""
import argparse, csv, json, math, os, sys
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))  # repo layout: evaluation/ next to backend/
from ranking import Index, load_docs, load_cleaned, precision_recall_at_k

def pop_from_ratings(data_dir):
    s, n = defaultdict(float), defaultdict(int)
    with open(os.path.join(data_dir, "ratings.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s[r["movieId"]] += float(r["rating"]); n[r["movieId"]] += 1
    return {m: (n[m], s[m] / n[m]) for m in n}

def pop_from_cleaned(path):
    out = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            out[r["movieId"]] = (float(r.get("rating_count") or 0), float(r.get("average_rating") or 0))
    return out

def make_prior(pop, kind):
    mx = max((c for c, _ in pop.values()), default=1)
    den = math.log1p(mx)
    def prior(m):
        c, a = pop.get(m, (0, 0.0))
        p = math.log1p(c) / den
        return p * (a / 5.0) if kind == "count_avg" else p
    return prior

POOL = 100

def search_blend(idx, prior, q, k, w, pool=None):
    hits = idx.search_bm25(q, pool or POOL)
    if not hits:
        return []
    top = hits[0][1] or 1.0
    rer = sorted(((s / top + w * prior(m), m) for m, s in hits), reverse=True)
    return [m for _, m in rer[:k]]

def score(idx, prior, queries, w, ks=(5, 10)):
    docs_ids = set(idx.docs) if hasattr(idx, "docs") else None
    tot = {f"{n}@{k}": [] for k in ks for n in "PR"}
    per = []
    for it in queries:
        rel = set(it["relevant"])
        if docs_ids is not None:
            rel &= docs_ids
        if not rel:
            continue
        ranked = search_blend(idx, prior, it["q"], max(ks), w)
        row = {"q": it["q"]}
        for k in ks:
            p, r = precision_recall_at_k(ranked, rel, k)
            tot[f"P@{k}"].append(p); tot[f"R@{k}"].append(r); row[f"P@{k}"] = p
        per.append(row)
    return {k: sum(v) / len(v) for k, v in tot.items()}, per

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--cleaned")
    ap.add_argument("--min-ratings", type=int, default=0)
    ap.add_argument("--queries", default=os.path.join(HERE, "eval_queries.json"))
    ap.add_argument("--weights", default="0,0.1,0.25,0.5,1,2")
    ap.add_argument("--pool", type=int, default=100, help="how many BM25 hits to rerank")
    ap.add_argument("--per-query-w", type=float, default=None, help="print per-query P@10 for this weight")
    a = ap.parse_args()
    if not (a.data or a.cleaned): ap.error("give --data or --cleaned")
    global POOL
    POOL = a.pool
    queries = json.load(open(a.queries))["queries"]
    if a.cleaned:
        docs = load_cleaned(a.cleaned, a.min_ratings, use_title=False); pop = pop_from_cleaned(a.cleaned)
    else:
        docs = load_docs(a.data, use_title=False); pop = pop_from_ratings(a.data)
    idx = Index(docs); idx.docs = docs
    weights = [float(x) for x in a.weights.split(",")]
    print(f"BM25 (no title) + popularity prior, rerank top {a.pool}, {len(queries)} labeled queries\n")
    print(f"{'prior':<10}{'w':>6}{'P@5':>8}{'R@5':>8}{'P@10':>8}{'R@10':>8}")
    for kind in ("count", "count_avg"):
        prior = make_prior(pop, kind)
        for w in weights:
            m, _ = score(idx, prior, queries, w)
            print(f"{kind:<10}{w:>6}{m['P@5']:>8.3f}{m['R@5']:>8.3f}{m['P@10']:>8.3f}{m['R@10']:>8.3f}")
        print()
    if a.per_query_w is not None:
        prior = make_prior(pop, "count_avg")
        _, per = score(idx, prior, queries, a.per_query_w)
        for r in per: print(f"  {r['q'][:44]:<46}P@10={r['P@10']:.2f}")
    print("w=0 is plain BM25. Weights were swept on the same 13 queries, so treat the best row as optimistic.")

if __name__ == "__main__":
    main()

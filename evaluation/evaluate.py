import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))  # repo layout: evaluation/ next to backend/
from ranking import Index, load_docs, load_cleaned, precision_recall_at_k

def run(docs, queries, ks, labels=None):
    idx = Index(docs)
    out = {}
    for method in ("tfidf", "bm25"):
        search = idx.search_tfidf if method == "tfidf" else idx.search_bm25
        per_q = []
        for i, item in enumerate(queries, 1):
            rel = set(item["relevant"]) & set(docs)  # only count relevant movies present in this data
            ranked = [m for m, _ in search(item["q"], max(ks))]
            if labels is not None:
                for m in ranked:
                    if m not in labels[i]:
                        raise ValueError("Missing label for query " + str(i) + ", movie " + m)
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

def read_labels(path, queries):
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            i = int(row["query_number"])
            if i < 1 or i > len(queries):
                raise ValueError("Invalid query number: " + str(i))
            if row["query"] != queries[i - 1]["q"]:
                raise ValueError("Query text does not match query " + str(i))
            if row["relevant"] not in ("0", "1"):
                raise ValueError("Invalid label for movie " + row["movie_id"])
            if i not in labels:
                labels[i] = {}
            m = row["movie_id"]
            if m in labels[i]:
                raise ValueError("Duplicate label for movie " + m)
            labels[i][m] = int(row["relevant"])

    new_queries = []
    for i, item in enumerate(queries, 1):
        if i not in labels:
            raise ValueError("No labels for query " + str(i))
        relevant = [m for m, value in labels[i].items() if value == 1]
        new_queries.append({"q": item["q"], "relevant": relevant})
    return new_queries, labels

def save_results(res):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(folder, exist_ok=True)
    rows = []
    summary = []
    metrics = ("P@5", "R@5", "P@10", "R@10")

    for method in ("tfidf", "bm25"):
        for i, item in enumerate(res[method], 1):
            if item["n_rel"] == 0:
                continue
            row = {"query_number": i, "query": item["q"], "method": method.upper() if method == "bm25" else "TF-IDF"}
            for key in metrics:
                row[key] = round(item[key], 3)
            rows.append(row)

    print("\nFINAL COMPARISON")
    for method in ("TF-IDF", "BM25"):
        data = [row for row in rows if row["method"] == method]
        if not data:
            continue
        result = {"method": method}
        for key in metrics:
            result[key] = round(sum(row[key] for row in data) / len(data), 3)
        summary.append(result)
        print(method, " ".join(f"{key}={result[key]:.3f}" for key in metrics))

    path = os.path.join(folder, "per_query.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_number", "query", "method"] + list(metrics))
        writer.writeheader()
        writer.writerows(rows)

    path = os.path.join(folder, "summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"] + list(metrics))
        writer.writeheader()
        writer.writerows(summary)
    print("Results saved in evaluation/results/")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder with movies.csv/tags.csv (ml-latest-small)")
    ap.add_argument("--cleaned", help="path to movies_cleaned.csv")
    ap.add_argument("--min-ratings", type=int, default=0)
    ap.add_argument("--queries", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_queries.json"))
    ap.add_argument("--per-query", action="store_true", help="also print every query")
    ap.add_argument("--labels", help="CSV containing 0/1 relevance judgments")
    a = ap.parse_args()
    if not (a.data or a.cleaned):
        ap.error("give --data or --cleaned")
    with open(a.queries, "r", encoding="utf-8") as f:
        queries = json.load(f)["queries"]
    ks = (5, 10)

    if a.labels:
        queries, labels = read_labels(a.labels, queries)
        docs = load_cleaned(a.cleaned, a.min_ratings, False) if a.cleaned else load_docs(a.data, False)
        res = run(docs, queries, ks, labels)
        save_results(res)
        return

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

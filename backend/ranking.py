"""Workstream 3: Information Retrieval & Ranking (TF-IDF and BM25 baselines).

Pure standard library, no installs needed.

MovieLens files used (from the ml-latest or ml-32m folder):
  movies.csv  : movieId,title,genres        (genres split by '|')
  tags.csv    : userId,movieId,tag,timestamp (optional, user tags)

Usage:
  python ranking.py --data path/to/ml-latest "funny family movie with a plot twist"
  python ranking.py --selftest
"""
import argparse
import csv
import math
import os
import re
from collections import Counter, defaultdict

STOP = set("a an and the of for with in on to is are be it that this movie film movies "
           "films want something looking show me i some very really".split())


def tokenize(text):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if t not in STOP]


def load_docs(data_dir, use_title=True):
    """Return {movieId: (title, text)} with title + genres + user tags as text.
    use_title=False keeps the title for display but leaves it out of the searchable text."""
    titles, genres, tags = {}, {}, defaultdict(list)
    with open(os.path.join(data_dir, "movies.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[row["movieId"]] = row["title"]
            genres[row["movieId"]] = row["genres"].replace("|", " ")
    tags_path = os.path.join(data_dir, "tags.csv")
    if os.path.exists(tags_path):
        with open(tags_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tags[row["movieId"]].append(row["tag"])
    return {m: (titles[m], " ".join(([titles[m]] if use_title else []) + [genres[m]] + tags.get(m, [])))
            for m in titles}


def load_cleaned(path, min_ratings=0, use_title=True):
    """Load the team's data/movies_cleaned.csv (from pre-process.py).
    Columns: movieId,title,genres,year,tag,rating_count,average_rating,imdbId,tmdbId.
    Text = title + genres + year + merged user tags."""
    docs = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if float(row.get("rating_count") or 0) < min_ratings:
                continue  # skip movies with too few ratings (obscure titles dominate otherwise)
            genres = (row.get("genres") or "").replace("|", " ")
            text = " ".join(([row["title"]] if use_title else []) +
                            [genres, row.get("year") or "", row.get("tag") or ""])
            title = row["title"] + (f" ({row['year']})" if row.get("year") else "")
            docs[row["movieId"]] = (title, text)
    return docs


class Index:
    def __init__(self, docs):
        self.titles = {m: t for m, (t, _) in docs.items()}
        self.tf = {m: Counter(tokenize(txt)) for m, (_, txt) in docs.items()}
        self.len = {m: sum(c.values()) for m, c in self.tf.items()}
        self.avgdl = sum(self.len.values()) / max(len(self.len), 1)
        self.N = len(self.tf)
        self.df = Counter()
        for c in self.tf.values():
            self.df.update(c.keys())
        self.postings = defaultdict(list)
        for m, c in self.tf.items():
            for t in c:
                self.postings[t].append(m)
        # precompute tf-idf doc norms
        self.norm = {}
        for m, c in self.tf.items():
            s = sum((self._tfidf(tf, t)) ** 2 for t, tf in c.items())
            self.norm[m] = math.sqrt(s) or 1.0

    def idf(self, t):
        return math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))

    def _tfidf(self, tf, t):
        return (1 + math.log(tf)) * math.log((self.N + 1) / (self.df[t] + 1))

    def search_tfidf(self, query, k=10):
        q = Counter(tokenize(query))
        qw = {t: self._tfidf(tf, t) for t, tf in q.items() if t in self.df}
        qn = math.sqrt(sum(w * w for w in qw.values())) or 1.0
        scores = defaultdict(float)
        for t, w in qw.items():
            for m in self.postings[t]:
                scores[m] += w * self._tfidf(self.tf[m][t], t)
        ranked = sorted(((s / (self.norm[m] * qn), m) for m, s in scores.items()), reverse=True)
        return [(m, s) for s, m in ranked[:k]]

    def search_bm25(self, query, k=10, k1=1.5, b=0.75):
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            if t not in self.df:
                continue
            idf = self.idf(t)
            for m in self.postings[t]:
                tf = self.tf[m][t]
                denom = tf + k1 * (1 - b + b * self.len[m] / self.avgdl)
                scores[m] += idf * tf * (k1 + 1) / denom
        ranked = sorted(((s, m) for m, s in scores.items()), reverse=True)
        return [(m, s) for s, m in ranked[:k]]


def precision_recall_at_k(ranked_ids, relevant, k):
    top = ranked_ids[:k]
    hits = sum(1 for m in top if m in relevant)
    return hits / k, (hits / len(relevant) if relevant else 0.0)


def evaluate(index, queries, ks=(5, 10)):
    """queries: {query_text: set(relevant movieIds)}. Prints a comparison table."""
    print(f"{'method':8} " + " ".join(f"P@{k:<3} R@{k:<3}" for k in ks))
    for name, fn in (("tfidf", index.search_tfidf), ("bm25", index.search_bm25)):
        agg = {k: [0.0, 0.0] for k in ks}
        for q, rel in queries.items():
            ids = [m for m, _ in fn(q, max(ks))]
            for k in ks:
                p, r = precision_recall_at_k(ids, rel, k)
                agg[k][0] += p
                agg[k][1] += r
        n = len(queries)
        print(f"{name:8} " + " ".join(f"{agg[k][0]/n:.3f} {agg[k][1]/n:.3f}" for k in ks))


def title_match_rate(index, queries, k=10, method="bm25"):
    """Share of top-k results whose TITLE shares a word with the query.
    A high rate means results come from title wording, not from genres or tags."""
    fn = index.search_bm25 if method == "bm25" else index.search_tfidf
    hits = total = 0
    for q in queries:
        qt = set(tokenize(q))
        for m, _ in fn(q, k):
            total += 1
            hits += bool(qt & set(tokenize(index.titles[m])))
    return hits / total if total else 0.0


DEMO_QUERIES = ["funny family movie", "dark mystery with a plot twist",
                "action movie with a big plot twist", "feel good romantic comedy",
                "scary haunted house", "animated adventure for kids"]


def title_report(load, **kw):
    """Print title-match rate with and without the title in the searchable text."""
    for use_title in (True, False):
        idx = Index(load(use_title=use_title, **kw))
        row = {mth: title_match_rate(idx, DEMO_QUERIES, 10, mth) for mth in ("bm25", "tfidf")}
        label = "title in text    " if use_title else "title NOT in text"
        print(f"{label}  bm25 {row['bm25']:.0%}   tfidf {row['tfidf']:.0%}   (top 10, {len(DEMO_QUERIES)} queries)")


def selftest():
    docs = {
        "1": ("Toy Story (1995)", "Toy Story (1995) Animation Children Comedy family funny"),
        "2": ("Se7en (1995)", "Se7en (1995) Crime Mystery Thriller dark twist"),
        "3": ("The Sixth Sense (1999)", "The Sixth Sense (1999) Drama Mystery Thriller twist ending"),
        "4": ("Shrek (2001)", "Shrek (2001) Animation Comedy Children family funny"),
        "5": ("Heat (1995)", "Heat (1995) Action Crime Thriller"),
    }
    idx = Index(docs)
    print("bm25 'funny family movie':", [idx.titles[m] for m, _ in idx.search_bm25("funny family movie", 3)])
    print("tfidf 'dark mystery with a plot twist':", [idx.titles[m] for m, _ in idx.search_tfidf("dark mystery with a plot twist", 3)])
    evaluate(idx, {"funny family movie": {"1", "4"}, "mystery thriller twist": {"2", "3"}}, ks=(2, 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--data")
    ap.add_argument("--cleaned", help="path to the team's movies_cleaned.csv")
    ap.add_argument("--min-ratings", type=int, default=0,
                    help="with --cleaned, ignore movies with fewer ratings than this")
    ap.add_argument("--method", choices=["tfidf", "bm25"], default="bm25")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--no-title", action="store_true",
                    help="leave the title out of the searchable text (it is still shown in results)")
    ap.add_argument("--title-report", action="store_true",
                    help="print how often top results match the query only through the title")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.title_report and (a.data or a.cleaned):
        if a.cleaned:
            title_report(lambda use_title: load_cleaned(a.cleaned, a.min_ratings, use_title))
        else:
            title_report(lambda use_title: load_docs(a.data, use_title))
    elif (a.data or a.cleaned) and a.query:
        use_title = not a.no_title
        idx = Index(load_cleaned(a.cleaned, a.min_ratings, use_title) if a.cleaned
                    else load_docs(a.data, use_title))
        fn = idx.search_bm25 if a.method == "bm25" else idx.search_tfidf
        for m, s in fn(a.query, a.k):
            print(f"{s:7.3f}  {idx.titles[m]}")
    else:
        ap.print_help()

"""User-based collaborative filtering for the movie recommender, plus a hybrid reranker.

Pure standard library. Works with MovieLens ratings.csv (userId,movieId,rating,timestamp).
Uses Index from ranking.py for the text-search side.

Idea:
  1. Represent each user as a dict {movieId: rating}, mean-centered.
  2. Similarity between two users = cosine on their mean-centered ratings
     over movies both rated (shrunk when they share few movies).
  3. To score a movie for a user, average the neighbors' centered ratings
     weighted by similarity, then add the user's own mean.
  4. Hybrid: take text-search candidates (BM25), then blend in the CF score.

Usage:
  python cf.py --selftest
  python cf.py --data path/to/ml-latest-small --user 1 "funny family movie"
  python cf.py --data path/to/ml-latest-small --holdout   # simple RMSE check
"""
import argparse
import csv
import math
import os
import random
from collections import defaultdict


def load_ratings(data_dir):
    ratings = defaultdict(dict)
    with open(os.path.join(data_dir, "ratings.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings[row["userId"]][row["movieId"]] = float(row["rating"])
    return ratings


class UserCF:
    def __init__(self, ratings, k=30, min_overlap=3, shrink=5):
        self.r = ratings
        self.k, self.min_overlap, self.shrink = k, min_overlap, shrink
        self.mean = {u: sum(d.values()) / len(d) for u, d in ratings.items() if d}
        self.centered = {u: {m: v - self.mean[u] for m, v in d.items()} for u, d in ratings.items() if d}
        self.norm = {u: math.sqrt(sum(x * x for x in d.values())) or 1.0 for u, d in self.centered.items()}
        self.by_movie = defaultdict(list)
        for u, d in self.centered.items():
            for m in d:
                self.by_movie[m].append(u)

    def sim(self, a, b):
        da, db = self.centered[a], self.centered[b]
        if len(da) > len(db):
            da, db = db, da
        common = [m for m in da if m in db]
        if len(common) < self.min_overlap:
            return 0.0
        dot = sum(da[m] * db[m] for m in common)
        s = dot / (self.norm[a] * self.norm[b])
        return s * len(common) / (len(common) + self.shrink)  # shrink when overlap is small

    def neighbors(self, user, movie=None):
        pool = self.by_movie[movie] if movie is not None else self.centered.keys()
        cands = [(self.sim(user, v), v) for v in pool if v != user]
        cands = [(s, v) for s, v in cands if s > 0]
        cands.sort(reverse=True)
        return cands[: self.k]

    def predict(self, user, movie):
        """Predicted rating, or None if no neighbors rated the movie."""
        if user not in self.centered:
            return None
        num = den = 0.0
        for s, v in self.neighbors(user, movie):
            num += s * self.centered[v][movie]
            den += abs(s)
        if den == 0:
            return None
        return min(5.0, max(0.5, self.mean[user] + num / den))

    def recommend(self, user, n=10, candidates=None):
        seen = set(self.r.get(user, {}))
        if candidates is None:
            candidates = {m for v in (x for _, x in self.neighbors(user)) for m in self.r[v]}
        out = []
        for m in candidates:
            if m in seen:
                continue
            p = self.predict(user, m)
            if p is not None:
                out.append((m, p))
        out.sort(key=lambda x: -x[1])
        return out[:n]


def hybrid_search(index, cf, user, query, k=10, pool=50, alpha=0.7):
    """Blend text score (BM25) with CF predicted rating. alpha = weight on text.
    Text scores and ratings are each min-max scaled to 0..1 over the candidate pool."""
    text = index.search_bm25(query, pool)
    if not text:
        return []
    ts = [s for _, s in text]
    lo, hi = min(ts), max(ts)
    scored = []
    for m, s in text:
        t = (s - lo) / (hi - lo) if hi > lo else 1.0
        p = cf.predict(user, m)
        c = (p - 0.5) / 4.5 if p is not None else 0.5  # neutral when CF has no opinion
        scored.append((alpha * t + (1 - alpha) * c, m))
    scored.sort(reverse=True)
    return [(m, s) for s, m in scored[:k]]


def holdout_rmse(ratings, frac=0.2, seed=0, **kw):
    """Hide a fraction of each user's ratings, predict them, report RMSE and coverage."""
    rnd = random.Random(seed)
    train, test = defaultdict(dict), []
    for u, d in ratings.items():
        for m, v in d.items():
            if rnd.random() < frac:
                test.append((u, m, v))
            else:
                train[u][m] = v
    cf = UserCF(train, **kw)
    se, n = 0.0, 0
    for u, m, v in test:
        p = cf.predict(u, m)
        if p is not None:
            se += (p - v) ** 2
            n += 1
    rmse = math.sqrt(se / n) if n else float("nan")
    return rmse, n, len(test)


def selftest():
    from ranking import Index
    ratings = {
        "u1": {"1": 5, "4": 5, "2": 1, "5": 2},
        "u2": {"1": 5, "4": 4, "3": 1, "5": 2},
        "u3": {"2": 5, "3": 5, "1": 1, "4": 2},
        "u4": {"2": 4, "3": 5, "5": 3, "1": 2},
        "me": {"1": 5, "2": 1, "5": 2},  # likes family comedies, dislikes dark ones
    }
    cf = UserCF(ratings, k=3, min_overlap=2, shrink=1)
    print("neighbors of me:", [(round(s, 2), v) for s, v in cf.neighbors("me")])
    print("predict me, movie 4 (Shrek-like):", round(cf.predict("me", "4"), 2))
    print("predict me, movie 3 (dark):", round(cf.predict("me", "3"), 2))
    docs = {
        "1": ("Toy Story (1995)", "Toy Story Animation Children Comedy family funny"),
        "2": ("Se7en (1995)", "Se7en Crime Mystery Thriller dark twist"),
        "3": ("The Sixth Sense (1999)", "The Sixth Sense Drama Mystery Thriller twist ending"),
        "4": ("Shrek (2001)", "Shrek Animation Comedy Children family funny"),
        "5": ("Heat (1995)", "Heat Action Crime Thriller"),
    }
    idx = Index(docs)
    q = "twist mystery"
    print("text only  :", [idx.titles[m] for m, _ in idx.search_bm25(q, 3)])
    print("hybrid(me) :", [idx.titles[m] for m, _ in hybrid_search(idx, cf, "me", q, 3, alpha=0.5)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--data")
    ap.add_argument("--user")
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.data and a.holdout:
        rmse, n, total = holdout_rmse(load_ratings(a.data))
        print(f"RMSE {rmse:.3f} on {n} of {total} held-out ratings ({n/total:.0%} coverage)")
    elif a.data and a.user and a.query:
        from ranking import Index, load_docs
        idx = Index(load_docs(a.data))
        cf = UserCF(load_ratings(a.data))
        for m, s in hybrid_search(idx, cf, a.user, a.query, alpha=a.alpha):
            print(f"{s:6.3f}  {idx.titles[m]}")
    else:
        ap.print_help()

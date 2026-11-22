
import os
import sys
import json
import csv

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(root, "backend"))

from ranking import Index, load_cleaned

path = os.path.join(root, "data", "movies_cleaned.csv")
query_path = os.path.join(root, "evaluation", "pooled_queries.json")
output = os.path.join(root, "evaluation", "relevance_labels.csv")

docs = load_cleaned(path, use_title=False)
idx = Index(docs)

with open(query_path, "r") as f:
    queries = json.load(f)["queries"]

if os.path.exists(output):
    print("Label file already exists. Rename or move it before generating a new one.")
    sys.exit()

with open(output, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "query_number",
        "query",
        "movie_id",
        "title",
        "movie_text",
        "relevant"
    ])

    for i, item in enumerate(queries):
        query = item["q"]
        movies = []

        results1 = idx.search_tfidf(query, 10)
        results2 = idx.search_bm25(query, 10)

        for movie_id, score in results1 + results2:
            if movie_id not in movies:
                movies.append(movie_id)

        for movie_id in item["relevant"]:
            if movie_id in docs and movie_id not in movies:
                movies.append(movie_id)

        for movie_id in movies:
            title, movie_text = docs[movie_id]

            writer.writerow([
                i + 1,
                query,
                movie_id,
                title,
                movie_text[:200],
                ""
            ])

        print("Query", i + 1, "-", len(movies), "movies collected")

print("\nSaved:", output)

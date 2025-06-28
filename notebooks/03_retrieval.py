import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/cases.csv")

tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["text_full"])

def retrieve(query: str, k: int = 5):
    query_vec = tfidf.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_k_idx = np.argsort(sim_scores)[::-1][:k]
    return df.iloc[top_k_idx][["case_id", "ringkasan_fakta"]]

X_train, X_test, y_train, y_test = train_test_split(
    tfidf_matrix, df["case_id"], test_size=0.2, random_state=42
)

with open("data/eval/queries.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

topk = 5
hits = 0
for item in queries:
    pred = retrieve(item["query"], k=topk)["case_id"].tolist()
    found = item["ground_truth"] in pred
    hits += found
    print(f"Q: {item['query']}\n  → {pred}\n  → Match: {found}")

print(f"\nAkurasi Top-{topk}: {hits}/{len(queries)} ({(hits/len(queries))*100:.1f}%)")

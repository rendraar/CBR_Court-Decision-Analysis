import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset kasus
df = pd.read_csv("data/processed/cases.csv")

# Gunakan kolom solusi: ringkasan_fakta (atau ganti dengan amar_putusan jika tersedia)
case_solutions = dict(zip(df["case_id"], df["ringkasan_fakta"]))

# Representasi TF-IDF dari text_full
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["text_full"])

def predict_outcome(query: str, k: int = 5):
    # 1) Hitung TF-IDF query
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    # 2) Ambil top-k kasus berdasarkan similarity
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_case_ids = df.iloc[top_k_idx]['case_id'].tolist()

    # 3) Weighted voting berdasarkan similarity
    weighted = {}
    for idx in top_k_idx:
        case_id = df.iloc[idx]['case_id']
        solusi = case_solutions.get(case_id, "").strip()
        weighted[solusi] = weighted.get(solusi, 0) + scores[idx]

    if not weighted:
        return "Tidak ditemukan", []

    # 4) Solusi dengan skor tertinggi
    predicted_solution = max(weighted, key=weighted.get)
    return predicted_solution, top_k_case_ids

# Contoh 5 query kasus baru
queries = [
    {"query_id": "Q001", "query": "kepemilikan senjata api rakitan tanpa izin"},
    {"query_id": "Q002", "query": "tersangka membawa pistol ilegal saat razia"},
    {"query_id": "Q003", "query": "penyimpanan senjata tajam di rumah"},
    {"query_id": "Q004", "query": "ditemukan senjata api dalam mobil saat patroli"},
    {"query_id": "Q005", "query": "tersangka menodongkan senjata rakitan ke warga"},
]

results = []

for q in queries:
    predicted, top5 = predict_outcome(q["query"], k=5)
    results.append({
        "query_id": q["query_id"],
        "predicted_solution": predicted,
        "top_5_case_ids": ";".join(top5)
    })

df_result = pd.DataFrame(results)
os.makedirs("data/results", exist_ok=True)
df_result.to_csv("data/results/predictions.csv", index=False)

print("✅ Disimpan ke data/results/predictions.csv")
df_result


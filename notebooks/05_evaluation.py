import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# Load data kasus dan solusi
df = pd.read_csv("data/processed/cases.csv")
case_solutions = dict(zip(df["case_id"], df["ringkasan_fakta"]))

# Buat ulang TF-IDF vectorizer dan representasi dokumen
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["text_full"])

# Fungsi retrieve + predict dari Tahap 4
def predict_outcome(query: str, k: int = 5):
    query_vec = tfidf.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_case_ids = df.iloc[top_k_idx]['case_id'].tolist()

    weighted = {}
    for idx in top_k_idx:
        case_id = df.iloc[idx]['case_id']
        solusi = case_solutions.get(case_id, "").strip()
        weighted[solusi] = weighted.get(solusi, 0) + scores[idx]

    predicted_solution = max(weighted, key=weighted.get) if weighted else "Tidak ditemukan"
    return predicted_solution, top_k_case_ids

def eval_retrieval(queries, ground_truth_dict, k=5):
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.preprocessing import MultiLabelBinarizer

    y_true = []
    y_pred = []

    for q in queries:
        query_id = q["query_id"]
        query_text = q["query"]
        ground_truth = ground_truth_dict[query_id]
        
        # retrieve k teratas
        _, top_k_case_ids = predict_outcome(query_text, k=k)
        y_true.append([ground_truth])
        y_pred.append(top_k_case_ids)

    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    precision = precision_score(y_true_bin, y_pred_bin, average="micro")
    recall = recall_score(y_true_bin, y_pred_bin, average="micro")
    f1 = f1_score(y_true_bin, y_pred_bin, average="micro")

    result = {
        "top_k": k,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    return result

queries = [
    {"query_id": "Q001", "query": "kepemilikan senjata api rakitan"},
    {"query_id": "Q002", "query": "tersangka membawa pistol ilegal"},
    {"query_id": "Q003", "query": "senjata tajam ditemukan dalam tas"}
]

ground_truth = {
    "Q001": "case_003",
    "Q002": "case_007",
    "Q003": "case_002"
}

metrics = eval_retrieval(queries, ground_truth, k=5)
pd.DataFrame([metrics]).to_csv("data/eval/retrieval_metrics.csv", index=False)

def eval_prediction(predictions_df, solution_truth_dict):
    y_true = []
    y_pred = []

    for _, row in predictions_df.iterrows():
        qid = row['query_id']
        y_true.append(solution_truth_dict.get(qid, ""))
        y_pred.append(row['predicted_solution'])

    acc = accuracy_score(y_true, y_pred)
    return {"accuracy": acc}

# Load hasil prediksi dari tahap 4
df_pred = pd.read_csv("data/results/predictions.csv")

# Ground truth solusi
solution_truth = {
    "Q001": "dipenjara 5 tahun",
    "Q002": "pidana penjara",
    "Q003": "ditahan"
}

result = eval_prediction(df_pred, solution_truth)
pd.DataFrame([result]).to_csv("data/eval/prediction_metrics.csv", index=False)

sns.barplot(data=pd.DataFrame([metrics]), palette="pastel")
plt.title("Evaluasi Retrieval")
plt.show()

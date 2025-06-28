# 🔍 Case-Based Reasoning untuk Analisis Putusan Mahkamah Agung (Senjata Api)

Sistem ini merupakan implementasi sederhana dari pendekatan **Case-Based Reasoning (CBR)** menggunakan data putusan pidana dari kategori _Senjata Api_ pada situs [Direktori Putusan Mahkamah Agung Republik Indonesia](https://putusan3.mahkamahagung.go.id). Pipeline ini dibangun dengan Python dan menggunakan TF-IDF serta model retrieval sederhana.

---

## 📦 Struktur Folder

project/
├── data/

│ ├── pdf/ # File PDF asli hasil scraping

│ ├── raw/ # File teks hasil ekstraksi & cleaning

│ ├── processed/ # File structured (CSV, XLSX)

│ ├── results/ # Hasil prediksi

│ └── eval/ # Query dan hasil evaluasi

├── logs/

│ └── cleaning.log # Log pembersihan PDF

├── scripts/

│ ├── main.ipynb

│ ├── 03_retrieval.py

│ ├── 04_retrieval.py

│ ├── 05_evaluation.py

├── requirements.txt

└── README.md


---

## 🛠️ Instalasi

### 1. Instal dependency

pip install -r requirements.txt

🧪 Dependency Utama

    beautifulsoup4

    requests

    PyMuPDF

    pandas

    scikit-learn

    numpy

    matplotlib, seaborn (visualisasi)

    transformers (opsional untuk embedding BERT)

🚀 Jalankan Pipeline End-to-End
Tahap 1 & 2 – Scraping, Download PDF, Ekstraksi dan Cleaning

jupyter nbconvert --to notebook --execute --inplace notebooks/main.ipynb

Tahap 3 – Representasi Kasus

python notebooks/03_retrieval.py

Tahap 4 – Retrieval Kasus Mirip

python notebooks/04_retrieval.py

Tahap 5 – Evaluasi Model & Akurasi Prediksi

python notebooks/05_evaluation.py


💡 Contoh Query Manual

from scripts.predict import predict_outcome
query = "tersangka membawa pistol ilegal tanpa izin"
predicted, top_k = predict_outcome(query)
print(predicted)

📌 Catatan

    Dataset terbatas pada kategori Senjata Api, sebanyak ≥30 putusan.

    Sistem dapat diperluas ke domain lain seperti Narkotika, Pidana Umum, atau Perdata.

    Untuk representasi lebih kuat, dapat menggunakan IndoBERT atau RoBERTa pretrained.

📚 Lisensi

Proyek ini menggunakan lisensi MIT. Data yang digunakan bersumber dari publikasi resmi Mahkamah Agung dan hanya untuk tujuan edukatif dan riset.

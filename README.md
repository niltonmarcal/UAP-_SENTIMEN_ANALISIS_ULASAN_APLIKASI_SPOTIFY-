# 🎧 Klasifikasi Sentimen Review Spotify Menggunakan BiLSTM, IndoBERT, dan DistilBERT

## 📌 Deskripsi Proyek
Proyek ini merupakan tugas **Ujian Akhir Praktikum (UAP)** mata kuliah **Pembelajaran Mesin**.  
Tujuan utama proyek ini adalah melakukan **klasifikasi sentimen teks** terhadap ulasan pengguna aplikasi **Spotify** berbahasa Indonesia menggunakan pendekatan **Deep Learning**.

Model yang digunakan terdiri dari:
- **Neural Network dasar (Non-Pretrained)** berupa **BiLSTM**
- **Transfer Learning** menggunakan **IndoBERT**
- **Transfer Learning** menggunakan **DistilBERT Multilingual**

Selain proses pelatihan dan evaluasi model, proyek ini juga menyediakan **dashboard interaktif berbasis Streamlit** yang memungkinkan pengguna melakukan prediksi sentimen teks secara langsung.

---

## 🎯 Tujuan Proyek
- Mengimplementasikan **model RNN berbasis BiLSTM** yang dilatih dari awal (scratch)
- Menerapkan **Transfer Learning** menggunakan **IndoBERT**
- Menerapkan **Transfer Learning** menggunakan **DistilBERT Multilingual**
- Membandingkan performa model menggunakan metrik evaluasi
- Menyediakan **dashboard interaktif Streamlit** sebagai media demonstrasi hasil model

---

## 📂 Dataset

### 🔹 Sumber Dataset
Dataset berupa **ulasan aplikasi Spotify** berbahasa Indonesia dengan total sekitar **6000 data** yang telah melalui proses preprocessing dan penyeimbangan kelas (balanced).

Dataset disimpan dalam folder:

### 🔹 Struktur Dataset
Dataset berbentuk file CSV dengan kolom utama:
- `text_clean` : teks ulasan yang telah diproses
- `label` : kelas sentimen (positif, netral, negatif)

📌 Dataset telah diseimbangkan untuk mengurangi bias antar kelas.

---

## 🧹 Preprocessing Dataset
Tahapan preprocessing teks yang dilakukan meliputi:
- Case folding (huruf kecil)
- Penghapusan URL, simbol, dan karakter non-alfanumerik
- Normalisasi spasi
- Tokenisasi teks
- Padding dan truncation untuk input model

Preprocessing dilakukan untuk memastikan data siap digunakan oleh model **BiLSTM** maupun **Transformer-based models**.

---

## 🧠 Model yang Digunakan

### 1️⃣ BiLSTM (Base Model / Non-Pretrained)
- Model **Bidirectional LSTM** yang dibangun dan dilatih dari awal
- Digunakan sebagai **baseline model**
- Embedding dan bobot dilatih tanpa pretrained weight
- Performa cukup baik, namun masih kalah dibandingkan model pretrained

---

### 2️⃣ IndoBERT (Transfer Learning)
- Menggunakan **IndoBERT pretrained**
- Dilakukan fine-tuning untuk tugas klasifikasi sentimen
- Memberikan performa terbaik pada teks Bahasa Indonesia
- Mampu menangkap konteks dan makna kalimat dengan lebih baik

---

### 3️⃣ DistilBERT Multilingual (Transfer Learning)
- Model transformer hasil distilasi BERT
- Mendukung teks multibahasa
- Lebih ringan dan cepat dibandingkan BERT standar
- Performa mendekati IndoBERT dengan efisiensi lebih tinggi

---

## 📊 Evaluasi Model
Evaluasi performa model dilakukan menggunakan:
- **Accuracy**
- **F1-Score (Macro)**
- **Confusion Matrix**
- **Classification Report**

---

## 📈 Hasil dan Perbandingan Model

### 🔹 Tabel Perbandingan Performa Model

| Model                    | Accuracy | F1-Score (Macro) |
|--------------------------|----------|------------------|
| BiLSTM (Base Model)      | 0.78     | 0.77             |
| IndoBERT                 | 0.88     | 0.87             |
| DistilBERT Multilingual  | 0.85     | 0.84             |

📌 *Catatan:*  
- Nilai Accuracy dan F1-Score diperoleh dari data uji (test set).  
- F1-Score Macro digunakan untuk memberikan bobot yang seimbang pada setiap kelas sentimen.

### 🔹 Analisis Hasil
- **BiLSTM** mampu menangkap pola sekuensial teks, namun performanya masih terbatas karena tidak memanfaatkan pengetahuan linguistik dari data besar.
- **IndoBERT** memberikan performa terbaik karena dilatih khusus pada korpus Bahasa Indonesia.
- **DistilBERT** menawarkan kompromi antara performa dan efisiensi komputasi.

Hasil perbandingan model juga disimpan dalam file:


---

## 🌐 Dashboard Streamlit

### 🔹 Fitur Dashboard
- Input teks ulasan Spotify
- Pilihan model (BiLSTM / IndoBERT / DistilBERT)
- Prediksi kelas sentimen
- Probabilitas tiap kelas sentimen

📌 Dashboard digunakan sebagai media demonstrasi hasil model.

---

## ▶️ Cara Menjalankan Dashboard

Install dependency:
bash
pip install streamlit torch transformers
streamlit run app.py

UAP_ML_PROJECT/
├── data/
│   └── spotify_uap_6000_balanced.csv
├── models/
│   ├── indobert/
│   ├── distilbert/
│   ├── base_bilstm.keras
│   └── label_classes.json
├── src/
│   └── app.py
├── UAP_ML.ipynb
├── model_comparison.csv
└── README.md

## 📝 Catatan Notebook
Notebook dibuat ulang menggunakan **VS Code (Jupyter)** untuk memastikan kompatibilitas penuh dengan **GitHub Notebook Renderer**.  
Notebook dapat dijalankan secara normal di **VS Code**, **Jupyter Notebook**, maupun **Google Colab**.

## ✅ Kesimpulan
Proyek ini berhasil mengimplementasikan klasifikasi sentimen teks menggunakan:
- satu model neural network dasar (**BiLSTM**)
- dua model pretrained berbasis Transformer (**IndoBERT** dan **DistilBERT**)

Hasil eksperimen menunjukkan bahwa **transfer learning memberikan peningkatan performa yang signifikan** dibandingkan model dasar.


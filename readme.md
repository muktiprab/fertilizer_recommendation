# 🌱 Fertilizer Recommendation System

Aplikasi Streamlit untuk rekomendasi pupuk berbasis Machine Learning menggunakan **Random Forest** dan **XGBoost**.

## Fitur
- Form input data pertanian (distrik, tanah, tanaman, kondisi lingkungan, kandungan nutrisi)
- Prediksi pupuk dari dua model sekaligus dengan confidence score
- Top-5 probabilitas prediksi per model
- Perbandingan performa model (F1-Macro, ROC-AUC, Cross-Validation)
- F1-Score per kelas pupuk
- Analisis distribusi dataset

## Cara Deploy ke Streamlit Community Cloud

1. Push semua file ke GitHub repository
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repository → pilih `app.py` sebagai main file
4. Deploy

## Struktur File
```
├── app.py
├── requirements.txt
└── README.md
```

> Model dilatih otomatis saat pertama kali app dijalankan dan di-cache oleh Streamlit.
> Tidak perlu menyertakan file `.pkl` ke repository.
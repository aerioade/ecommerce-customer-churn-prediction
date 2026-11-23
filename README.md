# Sistem Prediksi *E-Commerce Customer Churn* Berbasis Komparasi Algoritma Klasifikasi & *Explainable AI* (SHAP)

**Team-Based Project (TBP) Data Mining** — *End-to-End Machine Learning Pipeline* dan *Web Dashboard* Interaktif (Streamlit) untuk memprediksi risiko berhentinya pelanggan (*customer churn*) pada platform *e-commerce*, dilengkapi interpretabilitas **SHAP (*SHapley Additive exPlanations*)** serta sistem **3-Tier Risk Stratification** untuk rekomendasi aksi retensi otomatis bagi tim CRM.

---

## Ringkasan Proyek

* **Jenis Tugas Data Mining:** Klasifikasi Biner (*Binary Classification* — `Churn` = `1` vs `Non-Churn` = `0`)
* **Dataset:** `data/raw/data_ecommerce_customer_churn.csv` ($3.941$ baris mentah $\times$ $11$ kolom: $10$ fitur prediktor + $1$ target `Churn`; $3.270$ baris unik setelah pembersihan $671$ duplikat)
* **3 Algoritma Pembanding:**
  1. **Logistic Regression** (*Linear Baseline*)
  2. **Random Forest** (*Ensemble Bagging*)
  3. **XGBoost** (*Ensemble Boosting*)
* **Validasi & Optimasi:** *5-Fold Stratified Cross-Validation* (`RANDOM_STATE = 42`) dengan metrik optimasi utama **F1-Score**, serta evaluasi komprehensif menggunakan Accuracy, Precision, Recall, Specificity, F1-Score, FPR, dan ROC-AUC.
* **Explainable AI (XAI):** Penjelasan global & lokal berbasis **SHAP** serta validasi ganda (*Global SHAP Importance* vs *Tree Gain/Gini Importance*).

---

## Protokol Metodologi & Aturan Teknis

1. **Skema Kolom Aktual (11 Kolom):** Menggunakan tepat 10 fitur prediktor (`Tenure`, `WarehouseToHome`, `NumberOfDeviceRegistered`, `PreferedOrderCat`, `SatisfactionScore`, `MaritalStatus`, `NumberOfAddress`, `Complain`, `DaySinceLastOrder`, `CashbackAmount`) dan 1 target (`Churn`).
2. **Protokol Anti-*Data Leakage*:**
   * Standarisasi kategori deterministik pada `PreferedOrderCat` (`'Mobile'` $\rightarrow$ `'Mobile Phone'`).
   * Penghapusan $671$ baris duplikat (`drop_duplicates()`) **sebelum** pemisahan data agar sampel identik tidak bocor ke *train* dan *test*.
   * *Stratified Train-Test Split* ($80:20$, `random_state=42`) dilakukan **sebelum** proses *fitting* `SimpleImputer`, `OneHotEncoder`, dan `StandardScaler` melalui `scikit-learn` `Pipeline` & `ColumnTransformer`.
3. **Tanpa SMOTE & Tanpa PCA:**
   * Penanganan *class imbalance* menggunakan pembobotan kelas internal (`class_weight='balanced'` untuk Logistic Regression & Random Forest; `scale_pos_weight` untuk XGBoost) agar distribusi data tetap alami.
   * Tanpa reduksi dimensi PCA untuk menjaga performa F1-score serta mempertahankan nama fitur asli pada visualisasi SHAP.

---

## Sistem *3-Tier Risk Stratification*

Probabilitas *churn* $p = P(y = 1 \mid X)$ dipetakan menjadi tiga tingkat risiko bisnis:

| Tingkat Risiko | Ambang Probabilitas ($p$) | Rekomendasi Aksi Retensi Otomatis |
| :--- | :--- | :--- |
| **High Risk** | $p \ge 0.65$ | **Prioritas *Service Recovery*:** Hubungi pelanggan secara proaktif via tim CS/CRM dalam 24 jam, selesaikan komplain, dan berikan voucher retensi personal. |
| **Medium Risk** | $0.35 \le p < 0.65$ | **Kampanye *Re-Engagement*:** Kirim notifikasi personalisasi produk sesuai kategori favorit dan tawarkan promo *cashback* terarah. |
| **Low Risk** | $p < 0.35$ | ***Loyalty Reinforcement*:** Pertahankan loyalitas melalui program poin / VIP *tier* dan *early access* produk baru tanpa diskon agresif. |

---

## Struktur Repositori

```text
ecommerce-customer-churn-prediction/
├── data/
│   ├── raw/
│   │   └── data_ecommerce_customer_churn.csv
│   └── processed/
│       ├── train_data.csv
│       └── test_data.csv
├── src/
│   ├── __init__.py
│   ├── config.py             # Konstanta global, path, skema 10 fitur, threshold risiko
│   ├── data_loader.py        # Pembacaan CSV/Excel, validasi skema, standarisasi kategori
│   ├── eda.py                # Statistik deskriptif, duplikat, & deteksi outlier IQR
│   ├── preprocessing.py      # ColumnTransformer & Pipeline anti-data leakage
│   ├── train.py              # Training 3 model, 5-Fold CV, tuning, evaluasi, simpan artefak
│   ├── explainability.py     # Komputasi SHAP global & lokal + validasi ganda
│   └── risk_engine.py        # Logika 3-Tier Risk Stratification & rekomendasi retensi
├── models/                   # Pipeline terlatih (.joblib) & shap_background.joblib
├── artifacts/                # Ringkasan EDA, metrik evaluasi, & feature importance (.json)
├── app/
│   └── streamlit_app.py      # Web Dashboard Interaktif (4 Modul Utama)
├── notebooks/
│   └── full_experiment.ipynb # Dokumentasi eksperimen end-to-end
├── docs/
│   └── md/                   # Blueprint, context, dan rencana task proyek
├── requirements.txt
└── README.md
```

---

## Fitur *Web Dashboard* (`app/streamlit_app.py`)

1. **Modul 1 — Prediksi Individual & Penjelasan SHAP Lokal:** Form interaktif 10 fitur, probabilitas *churn* (%), *Risk Tier Badge*, rekomendasi retensi otomatis, *SHAP Waterfall Plot*, serta Top 3 faktor pendorong & penahan risiko.
2. **Modul 2 — *Batch Prediction* (Unggah CSV / Excel):** Unduh template CSV 10 kolom, proses prediksi massal, grafik distribusi *Risk Tier*, tabel hasil, dan unduh hasil prediksi (CSV).
3. **Modul 3 — Analitik & Komparasi 3 Model:** Tabel perbandingan metrik, *Grouped Bar Chart*, kurva ROC-AUC gabungan, *Confusion Matrix* per model, dan validasi ganda *Global SHAP* vs *Tree Gain/Gini Importance*.
4. **Modul 4 — Eksplorasi Data (EDA) & Ringkasan Proyek:** Statistik pembersihan data ($3.941 \rightarrow 3.270$ baris unik), *churn rate* per kategori, tabel deteksi *outlier* IQR, ringkasan metodologi, dan referensi ilmiah.

---

## Panduan Instalasi & Menjalankan Proyek

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Menjalankan EDA & Pelatihan Model
```bash
python -m src.eda
python -m src.train
```

### 3. Menjalankan *Web Dashboard* Streamlit
```bash
streamlit run app/streamlit_app.py
```

---

## Referensi Ilmiah Acuan (Scopus 2026)

1. **Inkesar, A. (2026).** *An Explainable Machine Learning Approach With Density-Regularized Oversampling for Industrial Customer Churn Prediction.* IEEE Access, Vol. 14, pp. 119355–119383.
2. **Hantash, L. A., et al. (2026).** *AI-Driven Customer Churn Prediction and Retention Intelligence from Behavioral Signals.* Journal of Intelligent Decision Making and Information Science (JIDMIS), Vol. 3(7s), pp. 1023–1041.
3. **Khan, M. I., et al. (2026).** *WPIS: Machine Learning-Driven E-Commerce Churn Prediction Integrating Technical Metrics and Information Quality for Strategic MIS Decision Support.* MDPI Information, Vol. 17(9), 886.

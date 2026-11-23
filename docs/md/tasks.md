# RENCANA PENGERJAAN PROYEK (TASK BREAKDOWN)
## Sistem Prediksi *E-Commerce Customer Churn* Menggunakan Komparasi Algoritma Klasifikasi Berbasis *Explainable AI* (SHAP)

Dokumen ini memecah spesifikasi teknis pada [context.md](file:///D:/Data%20Mining%20Project/ecommerce-customer-churn-prediction/docs/md/context.md) dan [blueprints.md](file:///D:/Data%20Mining%20Project/ecommerce-customer-churn-prediction/docs/md/blueprints.md) menjadi **10 Task berurutan**. Setiap task dirancang agar dapat dieksekusi satu per satu secara terstruktur (misal: *"kerjakan Task 1"*, *"lanjut Task 2"*, dst.).

---

### Aturan Ketat (*Hard Constraints*) di Seluruh Task
1. **Skema Kolom Aktual (11 Kolom):** Hanya gunakan 10 fitur prediktor + 1 target (`Churn`) dari `data/raw/data_ecommerce_customer_churn.csv`. Dilarang mereferensikan kolom fiktif (`CustomerID`, `Gender`, `CityTier`, `PreferredPaymentMode`).
2. **Tanpa SMOTE / *Oversampling* Sintetis:** Penanganan *class imbalance* hanya menggunakan `class_weight='balanced'` (Logistic Regression & Random Forest) dan `scale_pos_weight` (XGBoost).
3. **Tanpa PCA / Reduksi Dimensi:** Pertahankan nama fitur asli untuk keterbacaan SHAP.
4. **Protokol Anti-*Data Leakage* Mutlak:** `drop_duplicates()` wajib dijalankan **sebelum** `train_test_split`, dan *imputer/encoder/scaler* wajib di-*fit* hanya pada data *train* melalui `Pipeline` & `ColumnTransformer`.
5. **Hanya 3 Model Pembanding:** Logistic Regression, Random Forest, dan XGBoost.
6. **Reproducibility:** Gunakan `RANDOM_STATE = 42` secara konsisten.

---

## Ringkasan Urutan Task

| Task | Fokus Utama | Output / Target File |
| :--- | :--- | :--- |
| **Task 1** | Setup Struktur, Dependensi, & Konfigurasi Global | `requirements.txt`, `src/__init__.py`, `src/config.py`, struktur folder |
| **Task 2** | Data Loader & Exploratory Data Analysis (EDA + IQR) | `src/data_loader.py`, `src/eda.py`, `artifacts/eda_summary.json` |
| **Task 3** | Preprocessing Pipeline (Anti-*Data Leakage*) | `src/preprocessing.py`, `data/processed/train_data.csv`, `test_data.csv` |
| **Task 4** | Pelatihan 3 Model, 5-Fold CV, Tuning, & Evaluasi | `src/train.py`, `models/*.joblib`, `artifacts/evaluation_metrics.json` |
| **Task 5** | Explainable AI (SHAP) & Validasi Ganda | `src/explainability.py`, `models/shap_background.joblib`, `artifacts/feature_importance.json` |
| **Task 6** | Logika *3-Tier Risk Stratification* | `src/risk_engine.py` |
| **Task 7** | Web Dashboard: Kerangka Utama & Modul 1 (Individual + SHAP) | `app/streamlit_app.py` (Sidebar/Tabs + Modul 1) |
| **Task 8** | Web Dashboard: Modul 2 (*Batch Prediction* CSV/Excel) | `app/streamlit_app.py` (Modul 2) |
| **Task 9** | Web Dashboard: Modul 3 (Komparasi Model) & Modul 4 (EDA) | `app/streamlit_app.py` (Modul 3 & Modul 4) |
| **Task 10** | Notebook Eksperimen, `README.md`, & Verifikasi End-to-End | `notebooks/full_experiment.ipynb`, `README.md`, uji integrasi |

---

## Detail Spesifikasi Per Task

### Task 1: Setup Struktur Repository, Dependensi, & Konfigurasi Global
**Tujuan:** Menyiapkan struktur folder standar, daftar library Python, dan konstanta global agar seluruh modul saling terhubung secara konsisten.
* **Ruang Lingkup Kerja:**
  1. Buat struktur direktori: `data/processed/`, `src/`, `models/`, `artifacts/`, `app/`, `notebooks/`.
  2. Buat `requirements.txt` berisi: `pandas`, `numpy`, `openpyxl`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `plotly`, `joblib`, `streamlit`.
  3. Buat `src/__init__.py` dan `src/config.py` yang mendefinisikan:
     * Path direktori & file (`RAW_DATA_PATH`, `TRAIN_DATA_PATH`, `TEST_DATA_PATH`, `MODELS_DIR`, `ARTIFACTS_DIR`).
     * Konstanta `RANDOM_STATE = 42`, `TEST_SIZE = 0.20`, `N_SPLITS = 5`.
     * Daftar 10 fitur prediktor: 8 fitur numerik (`Tenure`, `WarehouseToHome`, `NumberOfDeviceRegistered`, `SatisfactionScore`, `NumberOfAddress`, `Complain`, `DaySinceLastOrder`, `CashbackAmount`) dan 2 fitur kategorikal (`PreferedOrderCat`, `MaritalStatus`), serta kolom target (`Churn`).
     * Pemetaan standarisasi kategori (`{'Mobile': 'Mobile Phone'}`) dan ambang batas risiko (`HIGH_RISK_THRESHOLD = 0.65`, `MEDIUM_RISK_THRESHOLD = 0.35`).
* **Kriteria Selesai:** Struktur folder lengkap tersedia dan `python -c "import src.config"` berjalan tanpa error.

---

### Task 2: Data Loader & Exploratory Data Analysis (EDA + Deteksi Outlier IQR)
**Tujuan:** Membangun utilitas pembacaan data yang aman serta modul EDA untuk mendokumentasikan kondisi dataset mentah vs bersih.
* **Ruang Lingkup Kerja:**
  1. Implementasikan `src/data_loader.py`:
     * Fungsi load file `.csv` dan `.xlsx`.
     * Validasi skema kolom wajib (10 fitur prediktor, dan opsional `Churn` untuk data *training*).
     * Standarisasi deterministik kolom `PreferedOrderCat`: `'Mobile'` $\rightarrow$ `'Mobile Phone'`.
  2. Implementasikan `src/eda.py`:
     * Statistik sebelum & sesudah `drop_duplicates()` (target: `3.941` baris mentah $\rightarrow$ `3.270` baris unik, `671` duplikat).
     * Distribusi kelas target `Churn` (`0` vs `1`) dan rasio imbalansi.
     * Jumlah & persentase *missing values* per kolom (`Tenure`, `WarehouseToHome`, `DaySinceLastOrder`).
     * *Churn rate* per kategori pada `PreferedOrderCat` dan `MaritalStatus`.
     * Deteksi *outlier* dengan metode **IQR** ($Q_1 - 1.5 \times \text{IQR}$ dan $Q_3 + 1.5 \times \text{IQR}$) pada fitur numerik kontinu (**hanya didokumentasikan, tidak dihapus**).
     * Simpan seluruh ringkasan EDA ke `artifacts/eda_summary.json`.
* **Kriteria Selesai:** Menjalankan `python -m src.eda` menghasilkan `artifacts/eda_summary.json` yang valid dan angka sesuai spesifikasi dataset.

---

### Task 3: Data Preprocessing Pipeline (Protokol Anti-*Data Leakage*)
**Tujuan:** Membangun alur pembagian data dan transformasi fitur yang menjamin nol kebocoran data (*zero data leakage*).
* **Ruang Lingkup Kerja:**
  1. Implementasikan `src/preprocessing.py`:
     * Fungsi `prepare_and_split_data()`: standarisasi kategori $\rightarrow$ `drop_duplicates().reset_index(drop=True)` $\rightarrow$ pisah $X$ dan $y$ $\rightarrow$ `train_test_split(..., test_size=0.20, stratify=y, random_state=42)`.
     * Simpan hasil split ke `data/processed/train_data.csv` (2.616 baris) dan `data/processed/test_data.csv` (654 baris).
     * Fungsi pembuat `ColumnTransformer`:
       * **Numerik (8 fitur):** `SimpleImputer(strategy='median')` + `StandardScaler()` (**khusus Logistic Regression**); tanpa `StandardScaler` untuk Random Forest & XGBoost.
       * **Kategorikal (2 fitur):** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
     * Pastikan `get_feature_names_out()` dapat dipanggil dengan rapi untuk pemetaan fitur SHAP.
     * Fungsi hitung `scale_pos_weight = N_negatif_train / N_positif_train` khusus dari `y_train`.
* **Kriteria Selesai:** `train_data.csv` dan `test_data.csv` terbentuk tanpa baris duplikat lintas partisi, serta `ColumnTransformer` dapat di-*fit* pada `X_train` dan mengekstrak nama fitur hasil *encoding*.

---

### Task 4: Pelatihan 3 Model, 5-Fold Stratified CV, Hyperparameter Tuning, & Evaluasi
**Tujuan:** Melatih, mengoptimasi, dan mengevaluasi Logistic Regression, Random Forest, dan XGBoost secara adil pada *test set* yang sama.
* **Ruang Lingkup Kerja:**
  1. Implementasikan `src/train.py`:
     * Gunakan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` dengan `scoring='f1'` pada `X_train`, `y_train`.
     * **Model 1 (Logistic Regression):** `class_weight='balanced'`, `max_iter=1000`, `random_state=42`, tuning `C` (`[0.01, 0.1, 1.0, 10.0]`), `solver=['lbfgs']`.
     * **Model 2 (Random Forest):** `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`, tuning `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
     * **Model 3 (XGBoost):** `scale_pos_weight=computed_ratio`, `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`, tuning `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.
     * Evaluasi ketiga model pada `X_test`, `y_test` untuk seluruh metrik wajib: `TP`, `TN`, `FP`, `FN`, Accuracy, Precision, Recall, Specificity, F1-Score, FPR, ROC-AUC, serta array `fpr`, `tpr`, `thresholds`.
     * Pilih 1 model terbaik berdasarkan **F1-Score tertinggi** (ROC-AUC sebagai *tie-breaker*).
     * Simpan `models/logistic_regression_pipeline.joblib`, `models/random_forest_pipeline.joblib`, `models/xgboost_pipeline.joblib`, `models/best_model_pipeline.joblib`, dan `artifacts/evaluation_metrics.json`.
* **Kriteria Selesai:** `python -m src.train` selesai tanpa error dan seluruh artefak `.joblib` serta `evaluation_metrics.json` terbentuk.

---

### Task 5: Explainable AI (SHAP) & Validasi Ganda (*Dual Validation*)
**Tujuan:** Menyediakan interpretabilitas global dan lokal menggunakan SHAP serta memvalidasi konsistensi peringkat fitur dengan *feature importance* bawaan model.
* **Ruang Lingkup Kerja:**
  1. Implementasikan `src/explainability.py`:
     * Ekstrak `preprocessor` dan `classifier` dari `models/best_model_pipeline.joblib`.
     * Bangun `shap.TreeExplainer(classifier)` untuk model berbasis pohon (tangani perbedaan bentuk output 2D pada XGBoost vs 3D `(n_samples, n_features, 2)` pada Random Forest agar selalu mengambil kontribusi kelas positif `1` secara aman).
     * Hitung **Global SHAP Importance** ($\text{FI}_{\text{SHAP}}$) pada data uji yang telah ditransformasi.
     * Bandingkan dengan `classifier.feature_importances_` (Gain/Gini Importance) dan simpan hasil **Dual Validation** ke `artifacts/feature_importance.json`.
     * Simpan metadata / background SHAP ke `models/shap_background.joblib`.
     * Sediakan fungsi inferensi **SHAP Lokal** untuk 1 observasi (mengembalikan objek `shap.Explanation` untuk *Waterfall Plot* + Top 3 faktor pendorong & Top 3 faktor penahan risiko *churn*).
* **Kriteria Selesai:** `artifacts/feature_importance.json` dan `models/shap_background.joblib` terbentuk, serta fungsi penjelasan lokal berjalan tanpa *shape mismatch error*.

---

### Task 6: Logika *3-Tier Risk Stratification* (`src/risk_engine.py`)
**Tujuan:** Menerjemahkan probabilitas *churn* menjadi tingkat risiko bisnis dan rekomendasi aksi retensi otomatis bagi tim CRM.
* **Ruang Lingkup Kerja:**
  1. Implementasikan `src/risk_engine.py`:
     * Pemetaan probabilitas $p = P(y = 1 \mid X)$:
       * **High Risk** ($p \ge 0.65$, warna `#DC2626`) $\rightarrow$ Rekomendasi *Service Recovery* proaktif dalam 24 jam + voucher retensi.
       * **Medium Risk** ($0.35 \le p < 0.65$, warna `#D97706`) $\rightarrow$ Rekomendasi *Re-Engagement* (notifikasi kategori favorit + promo *cashback* terarah).
       * **Low Risk** ($p < 0.35$, warna `#16A34A`) $\rightarrow$ Rekomendasi *Loyalty Reinforcement* (poin loyalitas / VIP tier tanpa diskon agresif).
     * Sediakan fungsi untuk prediksi tunggal (mengembalikan dictionary terstruktur: `tier`, `color`, `indication`, `action`) dan fungsi vektor untuk *batch DataFrame* (menambahkan kolom `Churn_Probability`, `Predicted_Churn`, `Risk_Tier`, `Recommended_Action`).
* **Kriteria Selesai:** Fungsi tunggal dan *batch* teruji untuk nilai batas ($0.0, 0.34, 0.35, 0.64, 0.65, 1.0$).

---

### Task 7: Web Dashboard — Kerangka Utama & Modul 1 (Prediksi Individual + SHAP Lokal)
**Tujuan:** Membangun kerangka aplikasi Streamlit serta fitur simulasi prediksi pelanggan tunggal lengkap dengan penjelasan SHAP lokal.
* **Ruang Lingkup Kerja:**
  1. Buat `app/streamlit_app.py` dengan desain bersih, navigasi *sidebar* / *tabs* untuk 4 modul utama, serta caching model & artefak (`@st.cache_resource`, `@st.cache_data`).
  2. Implementasikan **Modul 1 (Prediksi Individual & Penjelasan SHAP Lokal)**:
     * Form interaktif untuk seluruh **10 fitur prediktor** (`st.number_input`, `st.slider`, `st.selectbox`) dengan nilai default realistis (median/modus dataset).
     * Tampilkan **Probabilitas Churn (%)** (KPI card / gauge chart Plotly), status prediksi (`Berisiko Churn` / `Bertahan`), **Badge Risk Tier** berwarna, dan kotak **Rekomendasi Aksi Retensi** dari `risk_engine.py`.
     * Render **SHAP Waterfall Plot** untuk input pelanggan tersebut tanpa terpotong, disertai ringkasan teks otomatis **Top 3 Faktor Pendorong** dan **Top 3 Faktor Penahan** risiko *churn*.
* **Kriteria Selesai:** `streamlit run app/streamlit_app.py` berjalan lancar, Modul 1 menerima 10 input dan menampilkan probabilitas, *risk tier*, rekomendasi, serta *SHAP Waterfall Plot*.

---

### Task 8: Web Dashboard — Modul 2 (*Batch Prediction* Upload CSV/Excel)
**Tujuan:** Memungkinkan pengguna memprediksi risiko *churn* ratusan hingga ribuan pelanggan sekaligus melalui unggah berkas.
* **Ruang Lingkup Kerja:**
  1. Implementasikan **Modul 2** pada `app/streamlit_app.py`:
     * Tombol **"Download Template CSV"** berisi header 10 kolom prediktor.
     * Komponen `st.file_uploader` yang menerima file `.csv` dan `.xlsx`.
     * Validasi kelengkapan 10 kolom wajib + standarisasi `'Mobile'` $\rightarrow$ `'Mobile Phone'` sebelum inferensi.
     * Jalankan `best_model_pipeline.predict_proba()` + `risk_engine` untuk menghasilkan kolom `Churn_Probability`, `Predicted_Churn`, `Risk_Tier`, dan `Recommended_Action`.
     * Tampilkan KPI ringkasan *batch* (Total Pelanggan, jumlah High/Medium/Low Risk, rata-rata probabilitas *churn*), grafik distribusi *Risk Tier* (Plotly Bar/Pie Chart), tabel interaktif hasil prediksi, dan tombol **"Download Hasil Prediksi (CSV)"**.
* **Kriteria Selesai:** Unggah file CSV/Excel berjalan tanpa error, grafik distribusi tampil, dan hasil prediksi dapat diunduh sebagai CSV.

---

### Task 9: Web Dashboard — Modul 3 (Analitik & Komparasi 3 Model) & Modul 4 (EDA & Ringkasan Proyek)
**Tujuan:** Menyajikan transparansi performa komparatif ketiga algoritma, validasi ganda SHAP, serta temuan EDA dan metodologi proyek pada dashboard.
* **Ruang Lingkup Kerja:**
  1. Implementasikan **Modul 3 (Analitik & Komparasi 3 Model)**:
     * Tabel perbandingan metrik (Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC) untuk Logistic Regression, Random Forest, dan XGBoost dengan sorotan pada model terbaik.
     * Visualisasi Plotly: Grouped Bar Chart perbandingan metrik, Kurva ROC-AUC gabungan (3 model + garis *random baseline*), *Confusion Matrix* interaktif per model (via dropdown/tabs), dan grafik berdampingan **Validasi Ganda** (*Global SHAP Importance* vs *Tree Gain/Gini Importance*).
  2. Implementasikan **Modul 4 (Eksplorasi Data & Ringkasan Proyek)**:
     * Statistik pembersihan data (`3.941` mentah $\rightarrow$ `3.270` unik) & distribusi kelas `Churn` vs `Non-Churn`.
     * Grafik *Churn Rate* berdasarkan `PreferedOrderCat` dan `MaritalStatus`.
     * Tabel ringkasan deteksi *outlier* IQR dari `artifacts/eda_summary.json`.
     * Penjelasan singkat protokol anti-*data leakage* dan daftar 3 referensi ilmiah Scopus (2026).
* **Kriteria Selesai:** Seluruh grafik Plotly di Modul 3 dan Modul 4 tampil responsif dan data bersumber akurat dari `artifacts/*.json`.

---

### Task 10: Notebook Eksperimen End-to-End, Dokumentasi `README.md`, & Verifikasi Akhir
**Tujuan:** Melengkapi dokumentasi akademis-teknis proyek dan memastikan seluruh *deliverables* terverifikasi dari hulu ke hilir.
* **Ruang Lingkup Kerja:**
  1. Buat `notebooks/full_experiment.ipynb` yang mendokumentasikan alur eksperimen *end-to-end* (EDA, deteksi *outlier* IQR, *preprocessing* anti-*data leakage*, 5-Fold CV & *tuning* 3 model, evaluasi metrik, analisis SHAP global/lokal, dan *3-Tier Risk Stratification*).
  2. Lengkapi `README.md` dengan identitas proyek, struktur repositori, panduan instalasi & cara menjalankan training maupun dashboard Streamlit, ringkasan hasil model, serta 3 referensi ilmiah Scopus 2026 (Inkesar 2026, Hantash et al. 2026, Khan et al. 2026).
  3. Lakukan verifikasi akhir menyeluruh: pastikan seluruh file pada `data/processed/`, `src/`, `models/`, `artifacts/`, `app/`, `notebooks/` lengkap dan dapat dijalankan ulang secara reproduksibel.
* **Kriteria Selesai:** Seluruh 6 *deliverables* utama proyek (dataset bersih, codebase + notebook, model `.joblib`, aplikasi Streamlit 4 modul, dokumentasi lengkap) siap digunakan untuk presentasi dan *live demo*.

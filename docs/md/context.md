# PROJECT SPECIFICATION & CONTEXT FOR AI CODING AGENT

## Sistem Prediksi E-Commerce Customer Churn Menggunakan Komparasi Algoritma Klasifikasi Berbasis Explainable AI (SHAP)

---

## 1. RINGKASAN PROYEK & TUJUAN UTAMA

Dokumen ini adalah *Single Source of Truth* (panduan teknis utama) untuk membangun proyek **Team-Based Project (TBP) Data Mining** secara *end-to-end* berdasarkan dataset aktual `data_ecommerce_customer_churn.csv`.

### Tujuan Program
Membangun *pipeline* Data Mining dan aplikasi *web dashboard* interaktif untuk:
1. Melakukan eksplorasi data (EDA), penghapusan duplikat, dan *preprocessing* anti-*data leakage* pada dataset `data_ecommerce_customer_churn.csv`.
2. Melatih, mengoptimasi (*hyperparameter tuning*), dan membandingkan performa **3 algoritma klasifikasi** dari 3 paradigma berbeda:
   * **Logistic Regression** (Linear Baseline)
   * **Random Forest** (Ensemble Bagging)
   * **XGBoost** (Ensemble Boosting)
3. Mengintegrasikan **Explainable AI (SHAP)** secara global dan lokal agar keputusan model transparan (tidak *black-box*).
4. Menyediakan *web dashboard* interaktif dengan fitur prediksi individual (10 input fitur), penjelasan SHAP lokal, *batch prediction* (upload CSV/Excel), analitik komparasi 3 model, serta **3-Tier Risk Stratification** dengan rekomendasi aksi retensi otomatis.

### Aturan Ketat (HARD CONSTRAINTS UNTUK AI AGENT)
AI Agent **WAJIB** mematuhi aturan berikut saat menulis kode:
1. **SKEMA KOLOM AKTUAL (11 KOLOM):** Gunakan hanya 10 fitur prediktor dan 1 kolom target (`Churn`) yang tersedia di `data_ecommerce_customer_churn.csv`. **Jangan** mereferensikan kolom yang tidak ada seperti `CustomerID`, `Gender`, `CityTier`, atau `PreferredPaymentMode`.
2. **TANPA SMOTE / Oversampling Sintetis:** Jangan gunakan `SMOTE`, `ADASYN`, atau metode *resampling* lainnya. Tangani ketidakseimbangan kelas (*class imbalance*) hanya menggunakan pembobotan kelas bawaan algoritma (`class_weight='balanced'` untuk Logistic Regression & Random Forest, serta `scale_pos_weight` untuk XGBoost).
3. **TANPA PCA / Reduksi Dimensi:** Jangan gunakan PCA atau teknik ekstraksi fitur yang mengubah nama fitur asli. Berdasarkan acuan literatur (Inkesar, 2026), PCA menurunkan performa F1-score pada dataset ini dan merusak keterbacaan grafik SHAP.
4. **PROTOKOL ANTI-DATA LEAKAGE MUTLAK:**
   * Lakukan `drop_duplicates()` terlebih dahulu (karena dataset memiliki 671 baris duplikat akibat ketiadaan `CustomerID` yang dapat menyebabkan kebocoran sampel identik antara *train* dan *test*).
   * Lakukan `train_test_split` (80:20 *stratified*) **SEBELUM** melakukan `fit` pada *imputer*, *encoder*, maupun *scaler*. Semua transformasi harus dibungkus menggunakan `scikit-learn` `Pipeline` dan `ColumnTransformer`.
5. **HANYA 3 MODEL PEMBANDING:** Logistic Regression, Random Forest, dan XGBoost.
6. **REPRODUCIBILITY:** Gunakan `RANDOM_STATE = 42` secara konsisten di seluruh proses *split*, *cross-validation*, dan inisialisasi model.

---

## 2. TECH STACK & STRUKTUR REPOSITORY

### Tech Stack Utama
* **Bahasa Pemrograman:** Python 3.10+
* **Manipulasi Data:** `pandas`, `numpy`, `openpyxl`
* **Machine Learning:** `scikit-learn`, `xgboost`
* **Explainable AI:** `shap`
* **Visualisasi:** `matplotlib`, `seaborn`, `plotly`
* **Serialisasi Model:** `joblib`
* **Web Dashboard / Prototype:** `streamlit` (prioritas utama untuk integrasi cepat ML + SHAP + UI)

### Struktur Folder Proyek yang Harus Dibangun
```text
ecommerce-churn-xai/
├── data/
│   ├── raw/
│   │   └── data_ecommerce_customer_churn.csv  # Dataset aktual (3.941 baris, 11 kolom)
│   └── processed/
│       ├── train_data.csv                     # Split train setelah deduplikasi (80%)
│       └── test_data.csv                      # Split test setelah deduplikasi (20%)
├── src/
│   ├── __init__.py
│   ├── config.py                              # Konstanta global (RANDOM_STATE, path, daftar 10 fitur)
│   ├── data_loader.py                         # Fungsi load dataset (.csv / .xlsx) & validasi skema
│   ├── eda.py                                 # Fungsi analisis EDA, deteksi duplikat & outlier IQR
│   ├── preprocessing.py                       # Pembuatan ColumnTransformer & Pipeline anti-leakage
│   ├── train.py                               # Training 3 model, 5-Fold CV, tuning, evaluasi, simpan artefak
│   ├── explainability.py                      # Komputasi SHAP (TreeExplainer) global & lokal
│   └── risk_engine.py                         # Logika 3-Tier Risk Stratification & rekomendasi retensi
├── models/
│   ├── logistic_regression_pipeline.joblib    # Full pipeline (Preprocessor + LR)
│   ├── random_forest_pipeline.joblib          # Full pipeline (Preprocessor + RF)
│   ├── xgboost_pipeline.joblib                # Full pipeline (Preprocessor + XGBoost)
│   ├── best_model_pipeline.joblib             # Salinan pipeline model terbaik
│   └── shap_background.joblib                 # Background data / metadata untuk SHAP di app
├── artifacts/
│   ├── evaluation_metrics.json                # Hasil metrik lengkap 3 model + confusion matrix + ROC data
│   ├── eda_summary.json                       # Ringkasan EDA, statistik duplikat, & outlier IQR
│   └── feature_importance.json                # Dual validation: Gain/Gini vs SHAP Global Importance
├── app/
│   └── streamlit_app.py                       # Aplikasi Web Dashboard Interaktif (4 Modul Utama)
├── notebooks/
│   └── full_experiment.ipynb                  # Notebook dokumentasi eksperimen end-to-end
├── requirements.txt
└── README.md
```

---

## 3. SPESIFIKASI DATASET AKTUAL & KAMUS DATA

* **File Dataset:** `data/raw/data_ecommerce_customer_churn.csv`
* **Dimensi Mentah (*Raw*):** 3.941 baris $\times$ 11 kolom (10 fitur prediktor + 1 kolom target `Churn`).
* **Kualitas Data Awal:**
  * **Baris Duplikat:** Terdapat 671 baris duplikat (menyisakan **3.270 baris unik** setelah `drop_duplicates()`).
  * **Missing Values:** Terdapat pada 3 kolom numerik (`Tenure`: 194 baris kosong, `WarehouseToHome`: 169 baris kosong, `DaySinceLastOrder`: 213 baris kosong).
* **Distribusi Target (`Churn`) pada Data Mentah (3.941 baris):**
  * `0` (Non-Churn / Bertahan): 3.267 baris ($82.90\%$)
  * `1` (Churn / Berhenti): 674 baris ($17.10\%$)

### Kamus Data Aktual (11 Atribut)

| No | Nama Kolom | Tipe Data | Peran | Missing Values (Raw) | Deskripsi & Nilai Unik |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Tenure` | Float (`float64`) | Numerik | 194 | Lama waktu berlangganan/menjadi pelanggan (dalam bulan). |
| 2 | `WarehouseToHome` | Float (`float64`) | Numerik | 169 | Jarak dari gudang pengiriman ke rumah pelanggan (km/mil). |
| 3 | `NumberOfDeviceRegistered` | Integer (`int64`) | Numerik | 0 | Jumlah total perangkat yang terdaftar pada akun pelanggan (1–6). |
| 4 | `PreferedOrderCat` | String (`object`) | Kategorikal | 0 | Kategori pesanan favorit: `Laptop & Accessory`, `Mobile`, `Mobile Phone`, `Fashion`, `Grocery`, `Others`. |
| 5 | `SatisfactionScore` | Integer (`int64`) | Numerik | 0 | Skor kepuasan pelanggan terhadap layanan (skala 1–5). |
| 6 | `MaritalStatus` | String (`object`) | Kategorikal | 0 | Status pernikahan pelanggan: `Single`, `Married`, `Divorced`. |
| 7 | `NumberOfAddress` | Integer (`int64`) | Numerik | 0 | Jumlah alamat pengiriman yang terdaftar pada akun pelanggan. |
| 8 | `Complain` | Binary (`int64`) | Numerik | 0 | Riwayat mengajukan komplain pada bulan terakhir (`0` = Tidak, `1` = Ya). |
| 9 | `DaySinceLastOrder` | Float (`float64`) | Numerik | 213 | Jumlah hari sejak pesanan terakhir hingga periode observasi. |
| 10 | `CashbackAmount` | Float (`float64`) | Numerik | 0 | Rata-rata nilai *cashback* yang diterima pelanggan. |
| 11 | `Churn` | Binary (`int64`) | **Target** | 0 | `0` = Bertahan (Non-Churn), `1` = Berhenti berlangganan (Churn). |

### Catatan Standarisasi Kategori (`data_loader.py`)
Pada kolom `PreferedOrderCat`, terdapat dua label kategori yang merujuk pada produk yang sama (`'Mobile'` dan `'Mobile Phone'`). Lakukan pemetaan deterministik berikut pada `data_loader.py` (aman dilakukan sebelum *split* karena tidak bergantung pada statistik data):
* `PreferedOrderCat`: `'Mobile'` $\rightarrow$ `'Mobile Phone'`

---

## 4. TAHAP 1: EXPLORATORY DATA ANALYSIS (EDA) & DETEKSI OUTLIER

Implementasikan modul `src/eda.py` untuk menghasilkan statistik deskriptif sebelum dan sesudah pembersihan duplikat, lalu menyimpannya ke `artifacts/eda_summary.json`:

1. **Distribusi Target & Duplikasi:** Hitung jumlah/persentase kelas `0` dan `1` serta jumlah baris duplikat.
2. **Analisis Missing Values:** Hitung jumlah dan persentase nilai kosong per kolom (`Tenure`, `WarehouseToHome`, `DaySinceLastOrder`).
3. **Churn Rate per Kategori:** Hitung persentase *churn* untuk setiap kategori pada `PreferedOrderCat` dan `MaritalStatus`.
4. **Deteksi Outlier dengan Metode IQR (Interquartile Range):**
   Hitung untuk fitur numerik kontinu menggunakan rumus:
   $$\text{IQR} = Q_3 - Q_1$$
   $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}$$
   $$\text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
   Suatu nilai $x$ dikategorikan sebagai *outlier* jika $x < \text{Lower Bound}$ atau $x > \text{Upper Bound}$.
   *(Catatan: Outlier cukup didokumentasikan pada EDA dan **tidak dihapus** karena Random Forest dan XGBoost robust terhadap outlier serta mencerminkan variasi perilaku pelanggan nyata).*

---

## 5. TAHAP 2: DATA PREPROCESSING (PROTOKOL ANTI-DATA LEAKAGE)

Implementasikan modul `src/preprocessing.py` dengan urutan eksekusi ketat berikut:

### Langkah 1: Standarisasi Label & Deduplikasi
1. Standarisasi `'Mobile'` $\rightarrow$ `'Mobile Phone'` pada `PreferedOrderCat`.
2. Hapus baris duplikat menggunakan `df = df.drop_duplicates().reset_index(drop=True)`. Langkah ini wajib dilakukan **sebelum** pemisahan data latih dan uji agar baris identik tidak masuk ke kedua partisi secara bersamaan (*data leakage*).
3. Pisahkan matriks fitur $X$ (10 fitur) dan vektor target $y$ (`Churn`).

### Langkah 2: Stratified Train-Test Split (80:20)
Lakukan pemisahan data **sebelum** komputasi imputasi atau *scaling* apa pun:
$$P(y = 1)_{\text{train}} \approx P(y = 1)_{\text{test}} \approx P(y = 1)_{\text{clean}}$$
Gunakan `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)`.

### Langkah 3: Penanganan Missing Values (Di dalam Pipeline)
* **Fitur Numerik (8 fitur: `Tenure`, `WarehouseToHome`, `NumberOfDeviceRegistered`, `SatisfactionScore`, `NumberOfAddress`, `Complain`, `DaySinceLastOrder`, `CashbackAmount`):**
  Gunakan `SimpleImputer(strategy='median')`. Nilai median dihitung hanya dari *training set*:
  $$x_{ij} = \text{median}(x_{\cdot j}^{\text{train}}), \quad \text{jika } x_{ij} \text{ kosong}$$
* **Fitur Kategorikal (2 fitur: `PreferedOrderCat`, `MaritalStatus`):**
  Gunakan `SimpleImputer(strategy='most_frequent')` sebagai pengaman pada data inferensi baru:
  $$x_{ij} = \arg\max_v P(x_{\cdot j}^{\text{train}} = v), \quad \text{jika } x_{ij} \text{ kosong}$$

### Langkah 4: Encoding Fitur Kategorikal
Gunakan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` pada 2 fitur kategorikal (`PreferedOrderCat`, `MaritalStatus`).
Pastikan `get_feature_names_out()` dapat dipanggil untuk memetakan nama fitur hasil *encoding* ke dalam visualisasi SHAP.

### Langkah 5: Feature Scaling (Khusus Logistic Regression)
Terapkan `StandardScaler()` pada 8 fitur numerik **hanya untuk pipeline Logistic Regression** menggunakan rumus Z-Score Normalization:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
di mana $\mu_{\text{train}}$ adalah rata-rata dan $\sigma_{\text{train}}$ adalah standar deviasi yang dihitung eksklusif dari data *training*.
*(Untuk Random Forest dan XGBoost, pipeline numerik cukup menggunakan `SimpleImputer(strategy='median')` tanpa scaling).*

### Langkah 6: Perhitungan Bobot Kelas (Tanpa SMOTE)
Hitung `scale_pos_weight` untuk XGBoost secara dinamis **hanya dari `y_train`**:
$$\text{scale\_pos\_weight} = \frac{N_{\text{negatif, train}}}{N_{\text{positif, train}}}$$

---

## 6. TAHAP 3: PEMODELAN, RUMUS MATEMATIS, & HYPERPARAMETER TUNING

Implementasikan modul `src/train.py` untuk melatih dan membandingkan 3 algoritma berikut:

### Model 1: Logistic Regression (Linear Baseline)
* **Kombinasi Linear:**
  $$z = w_0 + \sum_{j=1}^{n} w_j x_j = w^T X + b$$
* **Fungsi Sigmoid (Probabilitas Kelas Positif):**
  $$P(y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Fungsi Kerugian (Binary Cross-Entropy Loss):**
  $$L(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
* **Konfigurasi Dasar:** `class_weight='balanced'`, `max_iter=1000`, `random_state=42`.
* **Ruang Tuning (`GridSearchCV`):**
  * `classifier__C`: `[0.01, 0.1, 1.0, 10.0]`
  * `classifier__solver`: `['lbfgs']`

### Model 2: Random Forest Classifier (Ensemble Bagging)
* **Gini Impurity (Kriteria Pemisahan Node):**
  $$\text{Gini}(D) = 1 - \sum_{k=1}^{K} (p_k)^2$$
* **Entropy (Alternatif Kriteria Pemisahan):**
  $$\text{Entropy}(D) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$
* **Agregasi Prediksi (Soft Voting dari $B$ Pohon):**
  $$P(y = 1 \mid X) = \frac{1}{B} \sum_{b=1}^{B} P_b(y = 1 \mid X)$$
* **Konfigurasi Dasar:** `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`.
* **Ruang Tuning (`RandomizedSearchCV` / `GridSearchCV`):**
  * `classifier__n_estimators`: `[100, 200, 300]`
  * `classifier__max_depth`: `[None, 10, 20, 30]`
  * `classifier__min_samples_split`: `[2, 5, 10]`
  * `classifier__min_samples_leaf`: `[1, 2, 4]`

### Model 3: XGBoost Classifier (Ensemble Boosting)
* **Prediksi Aditif dari $K$ Pohon:**
  $$\hat{y}_i = \sum_{t=1}^{K} f_t(X_i), \quad f_t \in \mathcal{F}$$
* **Pembaruan Iteratif dengan Learning Rate ($\eta$):**
  $$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(X_i)$$
* **Fungsi Objektif Teregulasi:**
  $$\text{Obj}^{(t)} = \sum_{i=1}^{N} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(X_i)\right) + \Omega(f_t)$$
* **Penalti Kompleksitas Pohon:**
  $$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
* **Probabilitas Akhir (Sigmoid):**
  $$P(y = 1 \mid X_i) = \frac{1}{1 + e^{-\hat{y}_i}}$$
* **Konfigurasi Dasar:** `scale_pos_weight=computed_ratio`, `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`.
* **Ruang Tuning (`RandomizedSearchCV` / `GridSearchCV`):**
  * `classifier__n_estimators`: `[100, 200, 300]`
  * `classifier__max_depth`: `[3, 5, 7, 9]`
  * `classifier__learning_rate`: `[0.01, 0.05, 0.1, 0.2]`
  * `classifier__subsample`: `[0.8, 1.0]`
  * `classifier__colsample_bytree`: `[0.8, 1.0]`

### Strategi Cross-Validation & Pemilihan Model Terbaik
* Gunakan **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** pada *training set*.
* Gunakan `scoring='f1'` sebagai metrik optimasi utama saat *hyperparameter tuning*.
* Evaluasi ketiga model hasil tuning pada *held-out test set* (20%).
* Pilih **1 model terbaik secara keseluruhan** berdasarkan **F1-Score tertinggi** (dengan ROC-AUC sebagai penentu jika seri), lalu simpan sebagai `models/best_model_pipeline.joblib`.

---

## 7. TAHAP 4: METRIK EVALUASI & RUMUS MATEMATIS

Hitung seluruh metrik berikut pada *test set* untuk ketiga model dan simpan ke `artifacts/evaluation_metrics.json`:

1. **Confusion Matrix:** Ekstrak nilai $\text{TP}$ (True Positive), $\text{TN}$ (True Negative), $\text{FP}$ (False Positive), dan $\text{FN}$ (False Negative).
2. **Accuracy:**
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
3. **Precision:**
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
4. **Recall (Sensitivity / TPR):**
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
5. **Specificity (TNR):**
   $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$
6. **F1-Score:**
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
7. **False Positive Rate (FPR):**
   $$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$$
8. **ROC-AUC:**
   $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$
   *(Simpan array `fpr`, `tpr`, dan `thresholds` ke JSON agar dapat digambar sebagai kurva ROC interaktif di dashboard).*

---

## 8. TAHAP 5: EXPLAINABLE AI (SHAP) & VALIDASI GANDA

Implementasikan modul `src/explainability.py`:

### Rumus Matematis SHAP
1. **Nilai SHAP Lokal (Kontribusi Fitur ke-$j$ pada Observasi $i$):**
   $$\phi_j(i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \cdot (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$
2. **Sifat Aditif Prediksi:**
   $$\hat{y}_i = \phi_0 + \sum_{j=1}^{|F|} \phi_j(i)$$
3. **Kepentingan Fitur Global (Global SHAP Importance):**
   $$\text{FI}_{\text{SHAP}}(j) = \frac{1}{N} \sum_{i=1}^{N} |\phi_j(i)|$$

### Spesifikasi Implementasi Teknis SHAP
* Ekstrak tahap `preprocessor` dan `classifier` dari `best_model_pipeline.joblib`.
* Jika model terbaik adalah **XGBoost** atau **Random Forest**, gunakan `shap.TreeExplainer(classifier)`.
* **Penanganan Dimensi Output SHAP:**
  * Untuk `XGBClassifier`, `explainer(X_transformed)` menghasilkan array 2D `(n_samples, n_features)` dalam skala *log-odds*.
  * Untuk `RandomForestClassifier`, output `shap_values` bisa berbentuk 3D `(n_samples, n_features, 2)`. Pastikan kode mengambil irisan kelas positif (`[:, :, 1]` atau `shap_values[1]`) secara aman agar tidak terjadi *shape mismatch error*.
* **Dual Validation (Acuan: Khan et al., 2026):**
  Hitung dan simpan kedua peringkat fitur ke `artifacts/feature_importance.json`:
  1. `model_importance`: `classifier.feature_importances_` (Gain / Gini Importance).
  2. `shap_importance`: $\text{FI}_{\text{SHAP}}(j)$ (rata-rata nilai mutlak SHAP pada *test set*).

---

## 9. TAHAP 6: LOGIKA 3-TIER RISK STRATIFICATION (`src/risk_engine.py`)

Implementasikan fungsi pemetaan probabilitas *churn* $p = P(y = 1 \mid X)$ ke dalam 3 tingkat risiko bisnis (diadaptasi dari Khan et al., 2026):

| Risk Tier | Kondisi Probabilitas | Warna Badge UI | Rekomendasi Aksi Retensi Otomatis |
| :--- | :--- | :--- | :--- |
| **High Risk** | $p \ge 0.65$ | Merah (`#DC2626`) | **Prioritas Service Recovery:** Hubungi pelanggan secara proaktif melalui tim CS/CRM, selesaikan komplain dalam 24 jam, dan berikan penawaran retensi personal / voucher khusus. |
| **Medium Risk** | $0.35 \le p < 0.65$ | Kuning/Oranye (`#D97706`) | **Kampanye Re-Engagement:** Kirim notifikasi personalisasi produk berdasarkan kategori favorit, tawarkan promo *cashback* terarah, atau pengingat kupon aktif. |
| **Low Risk** | $p < 0.35$ | Hijau (`#16A34A`) | **Loyalty Reinforcement:** Pertahankan kepuasan melalui program poin loyalitas / VIP tier dan *early access* produk baru tanpa perlu diskon agresif. |

---

## 10. TAHAP 7: SPESIFIKASI WEB DASHBOARD (`app/streamlit_app.py`)

Bangun aplikasi web interaktif dengan desain bersih, modern, dan navigasi *sidebar* / *tabs* yang terdiri dari **4 Modul Utama**:

### Modul 1: Prediksi Individual & Penjelasan SHAP Lokal
* **Input:** Form interaktif untuk seluruh **10 fitur prediktor** (`Tenure`, `WarehouseToHome`, `NumberOfDeviceRegistered`, `PreferedOrderCat`, `SatisfactionScore`, `MaritalStatus`, `NumberOfAddress`, `Complain`, `DaySinceLastOrder`, `CashbackAmount`) menggunakan `st.number_input`, `st.slider`, dan `st.selectbox` dengan nilai default yang realistis dari median/modus dataset.
* **Output Prediksi:**
  * Gauge chart / KPI card menampilkan **Probabilitas Churn (%)**.
  * Status prediksi (`Berisiko Churn` / `Bertahan`) dan **Badge Risk Tier** (High / Medium / Low).
  * Kotak **Rekomendasi Aksi Retensi** otomatis dari `risk_engine.py`.
* **Output Explainable AI (SHAP Lokal):**
  * Tampilkan **SHAP Waterfall Plot** untuk input pelanggan tersebut.
  * Tampilkan ringkasan teks otomatis: **Top 3 Faktor Pendorong Risiko Churn** (nilai SHAP positif tertinggi) dan **Top 3 Faktor Penahan Churn** (nilai SHAP negatif terbesar).

### Modul 2: Batch Prediction (Upload CSV / Excel)
* **Input:** Komponen `st.file_uploader` menerima file `.csv` atau `.xlsx` berisi data banyak pelanggan sekaligus (sertakan tombol **"Download Template CSV"** yang memuat 10 kolom fitur prediktor).
* **Proses:** Standarisasi label kategori $\rightarrow$ validasi 10 kolom wajib $\rightarrow$ jalankan `best_model_pipeline.predict_proba()` $\rightarrow$ tambahkan kolom `Churn_Probability`, `Predicted_Churn`, `Risk_Tier`, dan `Recommended_Action`.
* **Output:**
  * KPI ringkasan *batch* (Total Pelanggan, Jumlah High/Medium/Low Risk, Rata-rata Probabilitas Churn).
  * Grafik distribusi *Risk Tier* (Bar/Pie Chart Plotly).
  * Tabel interaktif hasil prediksi + tombol **Download Hasil Prediksi (CSV)**.

### Modul 3: Dashboard Analitik & Komparasi 3 Model
* **Tabel Perbandingan Metrik:** Menampilkan Accuracy, Precision, Recall, Specificity, F1-Score, dan ROC-AUC untuk Logistic Regression, Random Forest, dan XGBoost secara berdampingan (sorot model terbaik).
* **Visualisasi Evaluasi Interaktif (Plotly):**
  * Grouped Bar Chart perbandingan metrik 3 model.
  * Kurva ROC-AUC gabungan (3 garis model + garis *random baseline*).
  * Confusion Matrix untuk masing-masing model (dapat dipilih lewat dropdown/tabs).
* **Validasi Ganda Feature Importance:**
  * Grafik berdampingan antara **Global SHAP Feature Importance** dan **Tree Gain/Gini Importance**.

### Modul 4: Eksplorasi Data (EDA) & Ringkasan Proyek
* Menampilkan statistik pembersihan data (3.941 baris mentah $\rightarrow$ 3.270 baris unik setelah deduplikasi) serta distribusi kelas target (`Churn` vs `Non-Churn`).
* Grafik *Churn Rate* berdasarkan kategori (`PreferedOrderCat` dan `MaritalStatus`).
* Tabel ringkasan deteksi *outlier* (IQR) dari `artifacts/eda_summary.json`.
* Penjelasan singkat arsitektur metodologi anti-*data leakage* dan referensi ilmiah.

---

## 11. PANDUAN EKSEKUSI & VERIFIKASI UNTUK AI AGENT

Jalankan urutan kerja berikut saat membangun program:
1. **Setup Environment:** Buat `requirements.txt` dan struktur folder sesuai Bagian 2. Letakkan dataset `data_ecommerce_customer_churn.csv` di `data/raw/`.
2. **Build Pipeline (`src/`):** Implementasikan `config.py`, `data_loader.py`, `eda.py`, `preprocessing.py`, `train.py`, `explainability.py`, dan `risk_engine.py`.
3. **Run Training Script:** Eksekusi `python -m src.train` dan pastikan seluruh artefak di `models/` (`.joblib`) serta `artifacts/` (`.json`) terbentuk tanpa error.
4. **Verify Metrics:** Pastikan tidak ada *data leakage* (duplikat dihapus sebelum split; *imputer* dan *scaler* hanya di-*fit* pada `X_train`), cek bahwa F1-score & ROC-AUC XGBoost/Random Forest melampaui Logistic Regression.
5. **Build & Test Dashboard:** Implementasikan `app/streamlit_app.py`, lalu uji dengan `streamlit run app/streamlit_app.py`. Pastikan:
   * Form 10 input pada prediksi individual berjalan dan SHAP Waterfall Plot ter-render dengan benar tanpa terpotong.
   * Upload *batch* CSV berjalan dan file hasil bisa diunduh.
   * Semua grafik di halaman komparasi model dan EDA tampil responsif.

---

## 12. REFERENSI ILMIAH ACUAN (SCOPUS 2026)

Sertakan referensi berikut pada `README.md` dan tab informasi aplikasi:
1. **Inkesar, A. (2026).** *An Explainable Machine Learning Approach With Density-Regularized Oversampling for Industrial Customer Churn Prediction.* IEEE Access, Vol. 14, pp. 119355–119383. *(Acuan karakteristik dataset e-commerce churn, deteksi outlier IQR, keputusan tanpa PCA, dan faktor SHAP dominan).*
2. **Hantash, L. A., et al. (2026).** *AI-Driven Customer Churn Prediction and Retention Intelligence from Behavioral Signals.* Journal of Intelligent Decision Making and Information Science (JIDMIS), Vol. 3(7s), pp. 1023–1041. *(Acuan keunggulan model ensemble pada sinyal perilaku pelanggan dan kebutuhan transparansi XAI).*
3. **Khan, M. I., et al. (2026).** *WPIS: Machine Learning-Driven E-Commerce Churn Prediction Integrating Technical Metrics and Information Quality for Strategic MIS Decision Support.* MDPI Information, Vol. 17(9), 886. *(Acuan protokol anti-data leakage, validasi ganda Gain vs SHAP, dan 3-Tier Risk Stratification).*
# BLUEPRINT LENGKAP TEAM-BASED PROJECT (TBP) DATA MINING
## Sistem Prediksi *E-Commerce Customer Churn* Menggunakan Komparasi Algoritma Klasifikasi Berbasis *Explainable AI* (SHAP)

---

## 1. RINGKASAN EKSEKUTIF & RUANG LINGKUP PROYEK

### Identitas Proyek
* **Judul Proyek:** Sistem Prediksi *E-Commerce Customer Churn* Menggunakan Komparasi Algoritma Klasifikasi Berbasis *Explainable AI* (SHAP)
* **Jenis Tugas Data Mining:** Prediksi Berbasis Klasifikasi Biner (*Binary Classification*)
* **Target Produk Akhir:** *Pipeline Machine Learning End-to-End* dan Aplikasi *Web Dashboard* Interaktif

### Apa yang Kita Lakukan
Proyek ini merancang dan membangun sistem Data Mining terapan dari hulu ke hilir (*end-to-end*) untuk memprediksi risiko berhentinya pelanggan (*customer churn*) pada platform *e-commerce*. Alur pengerjaan meliputi eksplorasi data (EDA), pembersihan data duplikat dan penerapan protokol *preprocessing* anti-*data leakage*, serta pelatihan dan evaluasi komparatif **3 algoritma klasifikasi** dari paradigma yang berbeda:
1. **Logistic Regression** (Paradigma Linear *Baseline*)
2. **Random Forest** (Paradigma *Ensemble Bagging*)
3. **XGBoost** (Paradigma *Ensemble Boosting*)

Untuk menjembatani hasil model kompleks (*black-box*) dengan kebutuhan pengambilan keputusan bisnis, sistem mengintegrasikan **Explainable AI (XAI)** menggunakan **SHAP (*SHapley Additive exPlanations*)** serta memetakan probabilitas risiko ke dalam skema **3-Tier Risk Stratification** (*High, Medium, Low Risk*) yang disajikan pada *web dashboard* interaktif.

### Urgensi Bisnis & Kebaruan Terapan (*Applied Novelty*)
* **Urgensi Ekonomi:** Biaya mengakuisisi pelanggan baru pada pasar digital berkisar **5 hingga 25 kali lebih mahal** dibandingkan mempertahankan pelanggan yang sudah ada (*existing customers*).
* **Celah Penelitian (*Research Gap*):** Literatur prediksi *churn* masih didominasi sektor telekomunikasi dan perbankan, sedangkan studi *e-commerce* yang menggabungkan validasi anti-*data leakage*, interpretabilitas SHAP, dan implementasi langsung ke dalam *dashboard* pendukung keputusan masih terbatas.
* **Kebaruan Terapan:** Mengoperasionalkan model terbaik dan penjelasan SHAP per individu ke dalam antarmuka web siap pakai lengkap dengan rekomendasi aksi retensi otomatis bagi tim *Customer Relationship Management* (CRM).

---

## 2. TARGET OUTPUT (*DELIVERABLES*) WAJIB TBP

| No | Artefak *Deliverable* | Deskripsi & Spesifikasi Teknis | Format Output |
| :--- | :--- | :--- | :--- |
| 1 | **Dataset Bersih & Terdokumentasi** | Dataset mentah (`3.941` baris) dan dataset hasil pembersihan/pembagian (*train-test split*) tanpa kebocoran data, disertai kamus data 11 atribut. | `.csv` |
| 2 | **Kode Program & Notebook Eksperimen** | *Codebase* Python modular dan notebook dokumentasi yang memuat EDA, deteksi *outlier* IQR, *preprocessing*, *5-Fold Cross-Validation*, *tuning*, evaluasi, dan analisis SHAP. | `.py`, `.ipynb` |
| 3 | **Artefak Model Terlatih** | File serialisasi *pipeline* lengkap (`Preprocessor + Classifier`) untuk ketiga algoritma serta model terbaik yang siap dipanggil untuk inferensi. | `.joblib` / `.pkl` |
| 4 | ***Prototype / Web Dashboard* Interaktif** | Aplikasi web berbasis Streamlit yang memuat form simulasi 10 input, grafik SHAP lokal, *batch prediction* (unggah CSV/Excel), dan analitik komparasi 3 model. | Web App (`streamlit`) |
| 5 | **Laporan Proyek Akademis** | Dokumen laporan ilmiah terstruktur (Bab 1 s.d. Bab 5) yang dilengkapi landasan matematis dan tinjauan pustaka 3 jurnal Scopus (2026). | `.pdf` / `.docx` |
| 6 | **Dek Presentasi & Skenario Demo** | Slide presentasi tim yang ringkas serta skenario demonstrasi langsung (*live demo*) aplikasi web di depan dosen penguji. | `.pptx` / `.pdf` |

---

## 3. SPESIFIKASI DATASET AKTUAL & TEMUAN EDA

### Dimensi & Distribusi Kelas (`data_ecommerce_customer_churn.csv`)
* **Dimensi Mentah (*Raw*):** $3.941$ baris pelanggan $\times$ $11$ kolom ($10$ fitur prediktor + $1$ target `Churn`).
* **Kualitas Data Awal:**
  * Ditemukan **$671$ baris duplikat** (karena kolom identitas seperti `CustomerID` tidak disertakan dalam dataset ini), menyisakan **$3.270$ baris unik** setelah pembersihan duplikat.
  * Ditemukan *missing values* alami pada 3 atribut numerik: `Tenure` ($194$ baris), `WarehouseToHome` ($169$ baris), dan `DaySinceLastOrder` ($213$ baris).
* **Distribusi Variabel Target (`Churn`) pada Data Mentah:**
  * **Kelas `0` (*Non-Churn* / Bertahan):** $3.267$ pelanggan ($82{,}90\%$)
  * **Kelas `1` (*Churn* / Berhenti):** $674$ pelanggan ($17{,}10\%$)
  * **Rasio Imbalansi:** $\approx 4{,}85 : 1$ (Imbalansi kelas moderat)

### Kamus Data Lengkap (11 Atribut)

| No | Nama Kolom | Tipe Data | Peran | *Missing Value* | Deskripsi & Nilai Unik |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Tenure` | Numerik (`float64`) | Prediktor | $194$ | Lama waktu berlangganan/menjadi anggota (dalam bulan). |
| 2 | `WarehouseToHome` | Numerik (`float64`) | Prediktor | $169$ | Jarak dari gudang pengiriman ke rumah pelanggan (km/mil). |
| 3 | `NumberOfDeviceRegistered` | Numerik (`int64`) | Prediktor | $0$ | Jumlah total perangkat yang terdaftar pada akun pelanggan ($1$–$6$). |
| 4 | `PreferedOrderCat` | Kategorikal (`object`) | Prediktor | $0$ | Kategori pesanan favorit: `Laptop & Accessory`, `Mobile`, `Mobile Phone`, `Fashion`, `Grocery`, `Others`. |
| 5 | `SatisfactionScore` | Numerik (`int64`) | Prediktor | $0$ | Skor kepuasan pelanggan terhadap layanan (skala $1$–$5$). |
| 6 | `MaritalStatus` | Kategorikal (`object`) | Prediktor | $0$ | Status pernikahan pelanggan: `Single`, `Married`, `Divorced`. |
| 7 | `NumberOfAddress` | Numerik (`int64`) | Prediktor | $0$ | Jumlah alamat pengiriman yang terdaftar pada akun pelanggan. |
| 8 | `Complain` | Numerik (`int64`) | Prediktor | $0$ | Riwayat mengajukan komplain dalam 1 bulan terakhir (`0` = Tidak, `1` = Ya). |
| 9 | `DaySinceLastOrder` | Numerik (`float64`) | Prediktor | $213$ | Jumlah hari sejak pesanan terakhir hingga periode observasi. |
| 10 | `CashbackAmount` | Numerik (`float64`) | Prediktor | $0$ | Rata-rata nilai *cashback* yang diterima pelanggan. |
| 11 | `Churn` | Numerik (`int64`) | **Target** | $0$ | `0` = Bertahan (*Non-Churn*), `1` = Berhenti (*Churn*). |

### Analisis EDA & Deteksi *Outlier* (Metode IQR)
1. **Konsolidasi Label Kategori:** Pada kolom `PreferedOrderCat`, label `'Mobile'` dan `'Mobile Phone'` merujuk pada kategori produk yang sama sehingga distandarisasi menjadi `'Mobile Phone'`.
2. **Pola Perilaku Utama:** Pelanggan yang mengajukan komplain (`Complain = 1`), memiliki masa langganan singkat (`Tenure` rendah), berstatus `Single`, atau memesan kategori `Mobile Phone` menunjukkan tingkat *churn* yang jauh lebih tinggi dibandingkan segmen lainnya.
3. **Rumus Deteksi *Outlier* dengan *Interquartile Range* (IQR):**
   $$\text{IQR} = Q_3 - Q_1$$
   $$\text{Lower Bound} = Q_1 - 1{,}5 \times \text{IQR}$$
   $$\text{Upper Bound} = Q_3 + 1{,}5 \times \text{IQR}$$
   Nilai $x$ dikategorikan sebagai *outlier* jika $x < \text{Lower Bound}$ atau $x > \text{Upper Bound}$. *Outlier* didokumentasikan pada tahap EDA dan **tidak dihapus** karena mencerminkan variasi transaksi pelanggan nyata serta dapat ditangani dengan baik oleh algoritma berbasis pohon (*Random Forest* dan *XGBoost*).

---

## 4. TAHAPAN *DATA PREPROCESSING* (PROTOKOL ANTI-*DATA LEAKAGE*)

Untuk mencegah kebocoran informasi dari data uji ke data latih (*data leakage*), tahapan penyiapan data dijalankan dengan urutan ketat berikut:

```text
[1. Standarisasi Kategori ('Mobile' -> 'Mobile Phone') & Deduplikasi (drop_duplicates)]
                                        |
                                        v
[2. Pemisahan Data Terlebih Dahulu: Stratified Train-Test Split (80% Train : 20% Test)]
                   /                                         \
                  v                                           v
   [Data Latih (80% = 2.616 baris)]              [Data Uji (20% = 654 baris)]
                  |                                           |
                  v                                           v
   [Fit Imputer, OneHotEncoder, &                [Transform Data Uji menggunakan
    StandardScaler hanya pada Train]              parameter statistik dari Train]
```

### Rincian Tahapan & Rumus *Preprocessing*
1. **Standarisasi & Penghapusan Duplikat:**
   Menghapus $671$ baris duplikat menggunakan `drop_duplicates()` **sebelum** pembagian data agar sampel identik tidak terpecah masuk ke *training set* dan *test set* secara bersamaan.
2. **Stratified Train-Test Split ($80:20$):**
   Membagi data bersih ($3.270$ baris) dengan `random_state=42` sambil menjaga proporsi kelas target tetap seimbang di kedua partisi:
   $$P(y = 1)_{\text{train}} \approx P(y = 1)_{\text{test}} \approx P(y = 1)_{\text{clean}}$$
3. **Penanganan *Missing Values* (`SimpleImputer`):**
   * **8 Fitur Numerik:** Diimputasi dengan nilai **Median** yang dihitung eksklusif dari data latih karena lebih tahan (*robust*) terhadap *outlier* dibandingkan rata-rata:
     $$x_{ij} = \text{median}\left(x_{\cdot j}^{\text{train}}\right), \quad \text{jika } x_{ij} \text{ kosong}$$
   * **2 Fitur Kategorikal:** Diimputasi dengan nilai **Modus (*Most Frequent*)** dari data latih sebagai pengaman saat inferensi data baru:
     $$x_{ij} = \arg\max_v P\left(x_{\cdot j}^{\text{train}} = v\right), \quad \text{jika } x_{ij} \text{ kosong}$$
4. ***Encoding* Fitur Kategorikal (`OneHotEncoder`):**
   Mengubah `PreferedOrderCat` dan `MaritalStatus` menjadi vektor biner menggunakan `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
5. ***Feature Scaling* (`StandardScaler` — Khusus Logistic Regression):**
   Membakukan skala 8 fitur numerik menggunakan parameter rata-rata ($\mu_{\text{train}}$) dan standar deviasi ($\sigma_{\text{train}}$) dari data latih:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
   *(Model berbasis pohon yaitu Random Forest dan XGBoost tidak memerlukan feature scaling).*
6. **Keputusan Teknis Penting:**
   * **Tanpa SMOTE / *Oversampling* Sintetis:** Penanganan ketidakseimbangan kelas dilakukan melalui pembobotan kelas internal (`class_weight='balanced'` pada LR & RF, serta `scale_pos_weight` pada XGBoost) agar distribusi data tetap alami:
     $$\text{scale\_pos\_weight} = \frac{N_{\text{negatif, train}}}{N_{\text{positif, train}}}$$
   * **Tanpa PCA:** Mengacu pada temuan empiris Inkesar (2026), reduksi dimensi PCA menurunkan performa F1-score pada dataset *churn* sekaligus menghilangkan nama fitur asli yang dibutuhkan untuk interpretabilitas SHAP.

---

## 5. METODOLOGI PEMODELAN & LANDASAN RUMUS MATEMATIS

### 1. Logistic Regression (*Linear Baseline*)
Memetakan kombinasi linear fitur ke dalam probabilitas kelas menggunakan fungsi aktivasi Sigmoid.
* **Kombinasi Linear:**
  $$z = w_0 + \sum_{j=1}^{n} w_j x_j = w^T X + b$$
* **Fungsi Sigmoid:**
  $$P(y = 1 \mid X) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Fungsi Kerugian (*Binary Cross-Entropy Loss*):**
  $$L(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$

### 2. Random Forest (*Ensemble Bagging*)
Membangun $B$ pohon keputusan secara paralel melalui *bootstrap sampling* dan pemilihan subset fitur acak (*feature subspacing*).
* **Ukuran Impuritas (*Gini Impurity*):**
  $$\text{Gini}(D) = 1 - \sum_{k=1}^{K} (p_k)^2$$
* **Ukuran Entropi (*Entropy*):**
  $$\text{Entropy}(D) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$
  di mana $p_k$ adalah proporsi sampel kelas $k$ pada simpul (*node*) $D$.
* **Agregasi Prediksi (*Soft Voting*):**
  $$P(y = 1 \mid X) = \frac{1}{B} \sum_{b=1}^{B} P_b(y = 1 \mid X)$$

### 3. XGBoost — *Extreme Gradient Boosting* (*Ensemble Boosting*)
Membangun $K$ pohon regresi secara sekuensial untuk meminimalkan galat residual dari iterasi sebelumnya dengan penalti regularisasi.
* **Prediksi Aditif:**
  $$\hat{y}_i = \sum_{t=1}^{K} f_t(X_i), \quad f_t \in \mathcal{F}$$
* **Pembaruan Iteratif dengan *Learning Rate* ($\eta$):**
  $$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot f_t(X_i)$$
* **Fungsi Objektif Teregulasi:**
  $$\text{Obj}^{(t)} = \sum_{i=1}^{N} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(X_i)\right) + \Omega(f_t)$$
* **Penalti Kompleksitas Pohon ($\Omega$):**
  $$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
  di mana $T$ adalah jumlah daun (*leaves*), $w_j$ adalah bobot daun ke-$j$, $\gamma$ adalah penalti jumlah daun, dan $\lambda$ adalah parameter regularisasi $L_2$.

### Strategi Validasi Silang & *Hyperparameter Tuning*
* Menggunakan **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** pada *training set* dengan metrik optimasi utama **`scoring='f1'`**.
* **Parameter yang Dioptimasi:**
  * **Logistic Regression:** Regularisasi $C \in \{0{,}01; 0{,}1; 1{,}0; 10{,}0\}$.
  * **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
  * **XGBoost:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

---

## 6. METRIK EVALUASI MODEL & RUMUS MATEMATIS

Evaluasi model diukur berdasarkan komponen *Confusion Matrix* ($\text{TP}$, $\text{TN}$, $\text{FP}$, $\text{FN}$) pada *test set* ($20\%$):

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall (Sensitivity)} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$\text{Specificity (TNR)} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{False Positive Rate (FPR)} = \frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{Specificity}$$

$$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}) \, d(\text{FPR})$$

### Prioritas Evaluasi Bisnis
Dalam kasus *customer churn*, kesalahan **False Negative ($\text{FN}$)**—pelanggan yang sebenarnya akan *churn* tetapi gagal terdeteksi—menimbulkan kerugian jauh lebih besar daripada **False Positive ($\text{FP}$)** karena biaya akuisisi pelanggan baru sangat tinggi. Oleh karena itu, penentuan model terbaik memprioritaskan **F1-Score**, **Recall**, dan **ROC-AUC**.

---

## 7. *EXPLAINABLE AI* (SHAP) & RUMUS *SHAPLEY VALUES*

### Landasan Matematis SHAP
1. **Kontribusi Fitur Individual (*Local SHAP Value*):**
   $$\phi_j(i) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|! \cdot (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$
   di mana $F$ adalah himpunan seluruh fitur, $S$ adalah subset fitur tanpa fitur ke-$j$, dan $f(S)$ adalah ekspektasi prediksi model pada subset $S$.
2. **Sifat Aditif Prediksi:**
   $$\hat{y}_i = \phi_0 + \sum_{j=1}^{|F|} \phi_j(i)$$
3. **Kepentingan Fitur Global (*Global SHAP Importance*):**
   $$\text{FI}_{\text{SHAP}}(j) = \frac{1}{N} \sum_{i=1}^{N} |\phi_j(i)|$$

### Validasi Ganda (*Dual Validation*)
Mengadopsi metodologi Khan et al. (2026), peringkat kepentingan fitur global dari **SHAP ($\text{FI}_{\text{SHAP}}$)** dibandingkan langsung dengan peringkat kepentingan fitur bawaan pohon (**Gain / Gini Importance**) untuk memvalidasi konsistensi faktor pendorong *churn* utama (`Tenure`, `Complain`, `SatisfactionScore`, `DaySinceLastOrder`, `NumberOfAddress`).

---

## 8. SPESIFIKASI *PROTOTYPE* / *DASHBOARD* WEB & STRATEGI RETENSI

### 4 Modul Utama *Web Dashboard* (`Streamlit`)
1. **Modul 1 — Prediksi Individual & Penjelasan SHAP Lokal:**
   * Form input interaktif untuk **10 fitur prediktor**.
   * Menampilkan probabilitas *churn* (%), status prediksi, *badge* **Risk Tier**, rekomendasi aksi retensi otomatis, serta grafik **SHAP Waterfall Plot** beserta 3 faktor pendorong dan penahan risiko utama.
2. **Modul 2 — *Batch Prediction* (Unggah CSV / Excel):**
   * Menyediakan unduhan *template* CSV 10 kolom, memproses ratusan/ribuan data pelanggan sekaligus, menampilkan grafik distribusi tingkat risiko, dan menyediakan tombol unduh hasil prediksi lengkap.
3. **Modul 3 — Analitik & Komparasi 3 Model:**
   * Tabel metrik evaluasi berdampingan (Logistic Regression vs Random Forest vs XGBoost), kurva ROC-AUC gabungan, *Confusion Matrix* per model, dan grafik perbandingan *Global SHAP* vs *Tree Gain Importance*.
4. **Modul 4 — Eksplorasi Data (EDA) & Ringkasan Proyek:**
   * Visualisasi statistik pembersihan data ($3.941 \rightarrow 3.270$ baris unik), distribusi *churn* per kategori, dan ringkasan deteksi *outlier* IQR.

### Sistem *3-Tier Risk Stratification* (Diadaptasi dari Khan et al., 2026)

| Tingkat Risiko | Ambang Probabilitas ($p$) | Indikasi Perilaku Utama | Rekomendasi Aksi Retensi Otomatis |
| :--- | :--- | :--- | :--- |
| **High Risk** | $p \ge 0{,}65$ | Risiko tinggi berpindah (misal: ada riwayat komplain, masa langganan baru, skor kepuasan rendah). | **Prioritas *Service Recovery*:** Hubungi pelanggan secara proaktif via tim CS/CRM dalam 24 jam, selesaikan keluhan, dan berikan voucher retensi personal. |
| **Medium Risk** | $0{,}35 \le p < 0{,}65$ | Keterlibatan (*engagement*) mulai menurun, jarak dari transaksi terakhir semakin lama. | **Kampanye *Re-Engagement*:** Kirim notifikasi personalisasi produk sesuai kategori favorit dan penawaran *cashback* terarah. |
| **Low Risk** | $p < 0{,}35$ | Pelanggan loyal, transaksi rutin, kepuasan terjaga. | ***Loyalty Reinforcement*:** Pertahankan dengan program poin loyalitas / VIP *tier* dan akses produk baru tanpa diskon agresif. |

---

## 9. SINTESIS TINJAUAN PUSTAKA (3 JURNAL SCOPUS 2026)

| Penulis & Jurnal | Fokus & Metodologi Utama | Kontribusi Spesifik untuk Proyek Ini |
| :--- | :--- | :--- |
| **Inkesar, A. (2026)**<br>*IEEE Access*, Vol. 14, pp. 119355–119383 | Prediksi *churn* berbasis XAI (SHAP), analisis *outlier* IQR, dan evaluasi dampak PCA serta *hyperparameter tuning* pada data ritel/*e-commerce*. | Menjadi acuan analisis *outlier* IQR, alasan tidak menggunakan PCA pada data *churn*, serta konfirmasi fitur perilaku dominan (`Tenure`, `Complain`, `DaySinceLastOrder`). |
| **Hantash, L. A., et al. (2026)**<br>*JIDMIS*, Vol. 3(7s), pp. 1023–1041 | Prediksi *churn* berbasis sinyal perilaku pelanggan dengan komparasi model *ensemble* dan SHAP untuk strategi retensi. | Menjadi landasan urgensi celah penelitian sektor *e-commerce*, keunggulan algoritma *tree-based ensemble* atas model linear, dan pentingnya interpretabilitas bagi tim bisnis. |
| **Khan, M. I., et al. (2026)**<br>*MDPI Information*, Vol. 17(9), 886 | Kerangka kerja prediksi *churn e-commerce* dengan XGBoost, protokol *preprocessing* anti-*data leakage*, validasi ganda SHAP, dan *MIS Decision Support*. | Menjadi rujukan utama alur pemisahan data sebelum *fitting*, metode validasi ganda (*Gain* vs *SHAP*), dan rancangan **3-Tier Risk Stratification** pada *dashboard*. |

---

## 10. STRUKTUR *REPOSITORY*, PEMBAGIAN PERAN TIM, & LINIMASA

### Struktur Folder *Repository*
```text
ecommerce-churn-xai/
├── data/
│   ├── raw/                  # data_ecommerce_customer_churn.csv (3.941 baris, 11 kolom)
│   └── processed/            # train_data.csv & test_data.csv (hasil deduplikasi & split)
├── src/
│   ├── config.py             # Konstanta, path, daftar 10 fitur prediktor
│   ├── data_loader.py        # Pembacaan CSV/Excel & standarisasi kategori
│   ├── eda.py                # Statistik deskriptif, duplikat, & deteksi outlier IQR
│   ├── preprocessing.py      # ColumnTransformer & Pipeline anti-data leakage
│   ├── train.py              # Pelatihan 3 model, 5-Fold CV, tuning, & evaluasi
│   ├── explainability.py     # Komputasi SHAP (TreeExplainer) global & lokal
│   └── risk_engine.py        # Logika 3-Tier Risk Stratification & rekomendasi aksi
├── models/                   # Artefak pipeline terlatih (.joblib)
├── artifacts/                # Hasil metrik evaluasi, EDA, & feature importance (.json)
├── app/
│   └── streamlit_app.py      # Aplikasi Web Dashboard Interaktif (4 Modul)
├── notebooks/
│   └── full_experiment.ipynb # Notebook dokumentasi eksperimen end-to-end
├── requirements.txt
└── README.md
```
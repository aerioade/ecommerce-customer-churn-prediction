# AGENTS.md — Panduan Kerja AI Coding Agent

Dokumen ini adalah instruksi resmi bagi setiap **AI Coding Agent** yang bekerja pada repositori **Sistem Prediksi *E-Commerce Customer Churn* Menggunakan Komparasi Algoritma Klasifikasi Berbasis *Explainable AI* (SHAP)**.

---

## 1. Langkah Wajib Sebelum Memulai (*Onboarding*)

Sebelum menulis, mengubah, atau mengeksekusi kode apa pun, AI Agent **WAJIB** membaca, menganalisis, dan memahami dokumen berikut:
1. **[docs/md/context.md](docs/md/context.md)** — *Single Source of Truth* spesifikasi teknis, skema data aktual, rumus matematis, struktur folder, dan *hard constraints*.
2. **[docs/md/blueprints.md](docs/md/blueprints.md)** — Blueprint lengkap proyek TBP Data Mining, target *deliverables*, landasan teori, dan tinjauan pustaka 3 jurnal Scopus (2026).
3. **[docs/md/tasks.md](docs/md/tasks.md)** — Pemecahan proyek menjadi **10 Task berurutan** lengkap dengan ruang lingkup dan kriteria selesai.
4. **[src/config.py](src/config.py)** — Konstanta global, path file, skema 10 fitur prediktor + 1 target, dan ambang batas risiko (*risk thresholds*).
5. **[data/raw/data_ecommerce_customer_churn.csv](data/raw/data_ecommerce_customer_churn.csv)** — Dataset mentah aktual proyek (`3.941` baris $\times$ `11` kolom); AI Agent wajib memeriksa dan memahami struktur, tipe data, nilai unik, *missing values*, dan duplikat pada dataset ini sebelum mengolah atau memodelkan data.

---

## 2. Alur Pengerjaan Berbasis Task

1. **Ikuti Urutan Task pada `docs/md/tasks.md`:** Kerjakan task sesuai instruksi pengguna (misalnya: *"Lakukan Task 2"*, *"Lanjut Task 3"*).
2. **Jaga Ketergantungan Antar Task:** Jangan melompati prasyarat task sebelumnya. Jika sebuah task membutuhkan artefak dari task sebelumnya yang belum ada (misalnya `best_model_pipeline.joblib` sebelum membangun dashboard), beri tahu pengguna terlebih dahulu.
3. **Penuhi Kriteria Selesai:** Setiap task baru dianggap selesai setelah seluruh file target dibuat/diperbarui dan kriteria verifikasinya diuji secara nyata tanpa error.
4. **Cek Status Environment:** Pastikan dependensi pada `requirements.txt` tersedia saat menjalankan modul terkait (`pandas`, `numpy`, `openpyxl`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `plotly`, `joblib`, `streamlit`).

---

## 3. Aturan Ketat (*Hard Constraints*) Proyek

AI Agent **DILARANG** melanggar ketentuan teknis berikut di seluruh tahapan kode:

1. **Skema Kolom Aktual (11 Kolom):**
   * Gunakan **hanya** 10 fitur prediktor (`Tenure`, `WarehouseToHome`, `NumberOfDeviceRegistered`, `PreferedOrderCat`, `SatisfactionScore`, `MaritalStatus`, `NumberOfAddress`, `Complain`, `DaySinceLastOrder`, `CashbackAmount`) dan 1 target (`Churn`) dari `data/raw/data_ecommerce_customer_churn.csv`.
   * **Jangan** mereferensikan kolom yang tidak ada di dataset seperti `CustomerID`, `Gender`, `CityTier`, atau `PreferredPaymentMode`.
2. **Protokol Anti-*Data Leakage* Mutlak:**
   * Lakukan standarisasi kategori (`'Mobile'` $\rightarrow$ `'Mobile Phone'` pada `PreferedOrderCat`) dan penghapusan duplikat (`drop_duplicates().reset_index(drop=True)`) **sebelum** `train_test_split`.
   * Lakukan `train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)` **sebelum** `fit` pada `SimpleImputer`, `OneHotEncoder`, atau `StandardScaler`.
   * Bungkus seluruh transformasi fitur menggunakan `scikit-learn` `Pipeline` dan `ColumnTransformer`.
3. **Tanpa SMOTE / *Oversampling* Sintetis:**
   * Jangan gunakan `SMOTE`, `ADASYN`, atau teknik *resampling* lainnya.
   * Tangani *class imbalance* hanya melalui `class_weight='balanced'` (Logistic Regression & Random Forest) dan `scale_pos_weight` yang dihitung eksklusif dari `y_train` (XGBoost).
4. **Tanpa PCA / Reduksi Dimensi:**
   * Jangan gunakan PCA atau transformasi yang menghilangkan nama fitur asli karena menurunkan F1-Score dan merusak interpretabilitas grafik SHAP.
5. **Penanganan *Outlier* IQR:**
   * *Outlier* yang terdeteksi metode IQR hanya didokumentasikan pada EDA (`artifacts/eda_summary.json`) dan **tidak dihapus**.
6. **Hanya 3 Model Pembanding & Reproducibility:**
   * Hanya latih dan bandingkan **Logistic Regression**, **Random Forest**, dan **XGBoost** dengan `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` dan `scoring='f1'`.
   * Gunakan `RANDOM_STATE = 42` secara konsisten di seluruh proses *split*, *cross-validation*, dan inisialisasi model.

---

## 4. Ketentuan Implementasi Teknis & Kualitas Kode

1. **Gunakan Konfigurasi Terpusat:** Selalu impor path, daftar fitur, `RANDOM_STATE`, dan threshold risiko dari `src/config.py`. Jangan menuliskan ulang konstanta atau path secara *hardcode* di modul lain.
2. **Kode Modular & Bersih:**
   * Pisahkan tanggung jawab sesuai struktur `src/` (`data_loader.py`, `eda.py`, `preprocessing.py`, `train.py`, `explainability.py`, `risk_engine.py`).
   * Hindari duplikasi logika; buat fungsi helper yang ringkas, jelas, dan memiliki *docstring*.
3. **Kompatibilitas SHAP:**
   * Pastikan `OneHotEncoder` dan `ColumnTransformer` mendukung `get_feature_names_out()`.
   * Tangani perbedaan dimensi output SHAP secara aman: 2D `(n_samples, n_features)` pada XGBoost vs 3D `(n_samples, n_features, 2)` pada Random Forest (ambil irisan kelas positif `1`).
4. **Konsistensi *3-Tier Risk Stratification*:**
   * **High Risk:** $p \ge 0.65$ (`#DC2626`)
   * **Medium Risk:** $0.35 \le p < 0.65$ (`#D97706`)
   * **Low Risk:** $p < 0.35$ (`#16A34A`)

---

## 5. Verifikasi & Pelaporan

1. **Uji Nyata Setelah Perubahan:** Setelah menulis atau mengubah kode, jalankan verifikasi yang relevan (misalnya `python -c "import ..."`, `python -m src.eda`, `python -m src.train`, atau uji aplikasi Streamlit) sebelum menyatakan task selesai.
2. **Kejujuran Hasil (*No Fabrication*):** Laporkan hanya file, simbol, metrik, dan output perintah yang benar-benar diamati dari eksekusi alat pada sesi berjalan. Jangan mengarang angka metrik evaluasi atau status kelulusan uji yang belum dijalankan.
3. **Jaga Dokumentasi Tetap Sinkron:** Jika ada penambahan dependensi atau perubahan struktur/cara pakai, perbarui `requirements.txt` dan `README.md` agar tetap selaras dengan kondisi kode aktual.

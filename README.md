# Segmentasi Produk E-Commerce untuk Manajemen Inventori menggunakan K-Medoids

Case Method — mengelompokkan produk e-commerce berdasarkan pola penjualannya
(Recency, Frequency, Quantity) menggunakan algoritma K-Medoids (PAM) yang
diimplementasikan dari nol, untuk mendukung keputusan manajemen stok/inventori.

## Dataset

"Online Retail" — UCI Machine Learning Repository (ID 352):
https://archive.ics.uci.edu/dataset/352/online+retail

File mentah (`Online Retail.xlsx`) tidak disertakan di repo ini (ukuran besar);
silakan unduh langsung dari tautan di atas dan letakkan di `data/extracted/Online Retail.xlsx`
sebelum menjalankan `code/02_preprocessing_rfm.py`.

## Struktur Folder

```
├── code/                               Source code Python, urut sesuai tahapan
│   ├── 02_preprocessing_rfm.py         Cleaning level baris transaksi (umum)
│   ├── 02b_preprocessing_product_rfm.py Cleaning & agregasi RFM khusus produk
│   ├── kmedoids.py                     Implementasi K-Medoids (PAM) dari nol
│   ├── test_kmedoids.py                Unit test (16 test, termasuk vs brute-force optimal)
│   ├── 04b_clustering_evaluation_product.py  Evaluasi & pemilihan k optimal
│   ├── 04c_regenerate_figures_remapped.py    Pengurutan ulang ID cluster + regenerasi figure
│   └── 05b_interpretasi_cluster_product.py   Interpretasi hasil & rekomendasi bisnis
├── notebook/
│   └── Preprocessing_Segmentasi_Produk_KMedoids.ipynb  Notebook siap-jalan di Google Colab
├── data/
│   └── product_rfm.csv                 Tabel RFM final per produk (3.659 produk)
└── output/
    ├── product_rfm_final_labeled.csv   Hasil akhir clustering + label segmen
    └── figures_product/                6 figure hasil evaluasi & interpretasi
```

## Cara Menjalankan

1. Unduh dataset dari UCI, letakkan di `data/extracted/Online Retail.xlsx`.
2. Jalankan berurutan dari folder `code/`:
   ```bash
   pip install pandas numpy matplotlib scikit-learn openpyxl
   python3 02_preprocessing_rfm.py
   python3 02b_preprocessing_product_rfm.py
   python3 04b_clustering_evaluation_product.py
   python3 04c_regenerate_figures_remapped.py
   python3 05b_interpretasi_cluster_product.py
   ```
   Atau jalankan `notebook/Preprocessing_Segmentasi_Produk_KMedoids.ipynb` langsung di Google Colab
   untuk tahap preprocessing (upload dataset saat diminta).
3. Verifikasi implementasi algoritma:
   ```bash
   python3 test_kmedoids.py
   ```

## Ringkasan Hasil

- **3.659 produk** berhasil diagregasi dari 541.909 baris transaksi mentah.
- **k=3** dipilih sebagai jumlah cluster optimal (Silhouette Score = 0,5820):
  - Cluster 0 — *Slow-moving / Dead-stock* (16,0%)
  - Cluster 1 — *Medium* (63,7%)
  - Cluster 2 — *Fast-moving* (20,3%)
- Ditemukan 37 produk dengan pola pembelian borongan (bulk purchase) yang memerlukan
  peninjauan manual sebelum keputusan restock besar diambil.

## Referensi

- Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. John Wiley & Sons.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53-65.

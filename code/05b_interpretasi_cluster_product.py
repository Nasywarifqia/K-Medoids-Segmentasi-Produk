"""
05b_interpretasi_cluster_product.py
Tahap 6 - Analisis Hasil (VERSI PRODUK)
Case Method: Segmentasi Produk E-Commerce untuk Manajemen Inventori
menggunakan Algoritma K-Medoids

Langkah:
1. Karakteristik tiap cluster (statistik lengkap + medoid representatif).
2. Interpretasi bisnis tiap cluster (untuk keputusan restock/diskon/discontinue).
3. Identifikasi kasus khusus (produk borongan 1x beli / outlier).
4. Kesimpulan.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

OUT_DIR = "../output"
FIG_DIR = "../output/figures_product"
PALETTE = ["#C44E52", "#DD8452", "#55A868"]
CLUSTER_NAMES = {0: "Slow-moving / Dead-stock", 1: "Medium", 2: "Fast-moving"}

print("=" * 70)
print("TAHAP 6 - ANALISIS HASIL (INTERPRETASI CLUSTER PRODUK)")
print("=" * 70)

prfm = pd.read_csv(f"{OUT_DIR}/product_rfm_with_clusters.csv")
X_scaled = np.load(f"{OUT_DIR}/X_scaled_product.npy")
features = ["Recency", "Frequency", "Quantity"]
best_k = int(prfm["Cluster"].nunique())

# -------------------------------------------------------------------
# 1. Karakteristik lengkap tiap cluster
# -------------------------------------------------------------------
print("\n[1] Statistik lengkap R-F-Q per cluster")
summary = prfm.groupby("Cluster")[features].agg(["mean", "median", "min", "max"]).round(1)
print(summary)

counts = prfm["Cluster"].value_counts().sort_index()
pct = (counts / len(prfm) * 100).round(1)

# Cari medoid representatif (produk paling dekat ke rata-rata cluster)
print("\n    Produk 'paling representatif' (terdekat ke pusat) tiap cluster:")
medoid_products = {}
for c in range(best_k):
    idx_in_cluster = np.where(prfm["Cluster"].values == c)[0]
    cluster_mean = X_scaled[idx_in_cluster].mean(axis=0)
    dist = np.sqrt(((X_scaled[idx_in_cluster] - cluster_mean) ** 2).sum(axis=1))
    rep_idx = idx_in_cluster[np.argmin(dist)]
    rep_row = prfm.iloc[rep_idx]
    medoid_products[c] = rep_row
    print(f"      Cluster {c} ({CLUSTER_NAMES[c]}): {rep_row['ProductName']} "
          f"(R={rep_row['Recency']:.0f}, F={rep_row['Frequency']:.0f}, Q={rep_row['Quantity']:.0f})")

# -------------------------------------------------------------------
# 2. Interpretasi bisnis & rekomendasi tiap cluster
# -------------------------------------------------------------------
print("\n[2] Interpretasi bisnis dan rekomendasi aksi per cluster")

recommendations = {
    0: {
        "nama": "Slow-moving / Dead-stock",
        "ciri": "Sudah lama tidak terjual (rata-rata 236.7 hari sejak transaksi terakhir), "
                "jarang dipesan (rata-rata 12.9 invoice), volume terjual kecil (rata-rata 136 unit).",
        "risiko": "Modal tertahan di gudang (holding cost tinggi), risiko produk usang/rusak, "
                  "menghabiskan ruang simpan yang bisa dipakai produk lain.",
        "aksi": [
            "Hentikan restock otomatis untuk produk di segmen ini.",
            "Lakukan clearance sale / diskon besar untuk melepas stok yang masih ada.",
            "Evaluasi apakah produk perlu di-discontinue permanen dari katalog.",
            "Jika masih ingin dipertahankan, coba re-listing dengan foto/deskripsi baru atau bundling dengan produk laris.",
        ],
    },
    1: {
        "nama": "Medium",
        "ciri": "Masih aktif terjual (rata-rata 15.5 hari sejak transaksi terakhir), frekuensi "
                "pemesanan moderat (rata-rata 53.8 invoice), volume terjual menengah (rata-rata 552 unit).",
        "risiko": "Berisiko 'tergelincir' jadi dead-stock jika tidak dipantau, tapi juga berpotensi "
                  "naik kelas jadi fast-moving jika didorong dengan promosi yang tepat.",
        "aksi": [
            "Pantau tren Recency & Frequency secara berkala (misal bulanan) untuk mendeteksi dini "
            "produk yang mulai melambat.",
            "Pertimbangkan promosi ringan (cross-sell, bundling) untuk mendorong produk promising "
            "naik ke kategori fast-moving.",
            "Stok direplenish secukupnya (just-in-time), tidak perlu stok besar seperti fast-moving.",
        ],
    },
    2: {
        "nama": "Fast-moving",
        "ciri": "Paling baru terjual (rata-rata 3.0 hari sejak transaksi terakhir), paling sering "
                "dipesan (rata-rata 340.7 invoice), volume terjual sangat besar (rata-rata 5,074 unit).",
        "risiko": "Risiko stockout (kehabisan stok) jika tidak direplenish tepat waktu, berpotensi "
                  "kehilangan penjualan dan pelanggan jika produk ini sering kosong.",
        "aksi": [
            "Prioritaskan restock rutin dan pastikan buffer stock aman (safety stock) untuk mencegah stockout.",
            "Jadikan prioritas dalam negosiasi harga/termin dengan supplier karena volume order tinggi.",
            "Tempatkan di lokasi gudang yang mudah diakses (fast picking area) untuk efisiensi fulfillment.",
        ],
    },
}

for c in range(best_k):
    r = recommendations[c]
    print(f"\n    --- Cluster {c}: {r['nama']} ({counts[c]:,} produk, {pct[c]}%) ---")
    print(f"    Ciri    : {r['ciri']}")
    print(f"    Risiko  : {r['risiko']}")
    print(f"    Aksi    :")
    for a in r["aksi"]:
        print(f"      - {a}")

# -------------------------------------------------------------------
# 3. Identifikasi kasus khusus: produk borongan 1x beli (outlier)
# -------------------------------------------------------------------
print("\n[3] Identifikasi kasus khusus: produk dengan Quantity besar tapi Frequency sangat rendah")
print("    (indikasi pembelian borongan/wholesale sekali besar, BUKAN pola fast-moving asli)")

# Rasio Quantity per Frequency -- kalau sangat tinggi dibanding median cluster, ini kandidat anomali
prfm["qty_per_invoice"] = prfm["Quantity"] / prfm["Frequency"]
threshold = prfm["qty_per_invoice"].quantile(0.99)
special_cases = prfm[prfm["qty_per_invoice"] >= threshold].sort_values("qty_per_invoice", ascending=False)
print(f"\n    Ambang batas (persentil 99 dari Quantity/Frequency): {threshold:.1f} unit per invoice")
print(f"    Jumlah produk yang masuk kategori ini: {len(special_cases)}")
print(special_cases[["StockCode", "ProductName", "Cluster", "Recency", "Frequency", "Quantity", "qty_per_invoice"]].head(10).to_string(index=False))

print(
    "\n    Catatan: produk-produk ini sebagian besar tetap masuk Cluster 2 (Fast-moving) karena "
    "Quantity totalnya besar, padahal pola pemesanannya tidak konsisten (cuma 1-beberapa kali "
    "beli borongan). Untuk kebutuhan OPERASIONAL (bukan hanya clustering statistik), tim gudang "
    "sebaiknya tetap meninjau manual produk dengan qty_per_invoice sangat tinggi ini, karena "
    "pola 'fast-moving'-nya bisa jadi semu (didorong oleh satu pembeli besar, bukan permintaan "
    "pasar yang luas/konsisten)."
)

prfm.to_csv(f"{OUT_DIR}/product_rfm_final_labeled.csv", index=False)

# -------------------------------------------------------------------
# Visualisasi: perbandingan karakteristik antar cluster (bar chart ternormalisasi)
# -------------------------------------------------------------------
mean_table = prfm.groupby("Cluster")[features].mean()
norm_table = (mean_table - mean_table.min()) / (mean_table.max() - mean_table.min() + 1e-9)

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(features))
width = 0.8 / best_k
for i in range(best_k):
    ax.bar(x + i * width, norm_table.loc[i].values, width=width,
           label=f"Cluster {i} - {CLUSTER_NAMES[i]}", color=PALETTE[i])
ax.set_xticks(x + width * (best_k - 1) / 2)
ax.set_xticklabels(features)
ax.set_ylabel("Nilai ternormalisasi (0-1)")
ax.set_title("Perbandingan Karakteristik R-F-Q antar Cluster Produk")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_perbandingan_karakteristik_cluster.png", dpi=150)
plt.close()
print(f"\n    Grafik disimpan: {FIG_DIR}/06_perbandingan_karakteristik_cluster.png")

# -------------------------------------------------------------------
# 4. Kesimpulan
# -------------------------------------------------------------------
print("\n[4] Kesimpulan hasil pengelompokan produk")
print(f"""
    - Dari {len(prfm):,} produk, teridentifikasi 3 segmen dengan karakteristik berbeda:
        * {counts[0]:,} produk ({pct[0]}%) adalah Slow-moving/Dead-stock -- kandidat utama
          untuk diskon/clearance/discontinue.
        * {counts[1]:,} produk ({pct[1]}%) adalah Medium -- perlu dipantau berkala, berpotensi
          naik atau turun kelas.
        * {counts[2]:,} produk ({pct[2]}%) adalah Fast-moving -- prioritas utama ketersediaan
          stok (safety stock & restock rutin).
    - Catatan penting: {len(special_cases)} produk (sebagian besar masuk Fast-moving) polanya
      didorong oleh pembelian borongan 1x beli dalam jumlah besar, bukan permintaan pasar yang
      konsisten. Tim operasional disarankan meninjau produk-produk ini secara manual sebelum
      mengambil keputusan restock besar hanya berdasarkan label cluster.
    - Segmentasi ini memberikan dasar OBJEKTIF dan DATA-DRIVEN bagi tim gudang/purchasing untuk
      memprioritaskan alokasi modal dan ruang simpan, dibanding mengandalkan intuisi semata.
""")

print("=" * 70)
print("TAHAP 6 SELESAI -> lanjut ke Tahap 7 (Penyusunan Laporan)")
print("=" * 70)

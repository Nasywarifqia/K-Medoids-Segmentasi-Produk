"""
02b_preprocessing_product_rfm.py
Tahap 3 - Data Preprocessing (VERSI PRODUK)
Case Method: Segmentasi Produk E-Commerce untuk Manajemen Inventori
menggunakan Algoritma K-Medoids

Melanjutkan dari data transaksi yang SUDAH bersih di level baris
(online_retail_clean.csv, hasil 02_preprocessing_rfm.py: sudah bebas
missing CustomerID, transaksi pembatalan, duplikat, dan harga <= 0).

Langkah tambahan khusus level PRODUK:
1. Menghapus kode non-produk (biaya administratif: ongkir, manual, dll).
2. Menentukan nama (Description) representatif per StockCode.
3. Mengubah data transaksi (per baris item) menjadi data per PRODUK.
4. Membentuk atribut Recency, Frequency, Quantity per produk.
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

CLEAN_CSV = "../data/online_retail_clean.csv"
PRODUCT_CLEAN_CSV = "../data/online_retail_clean_product.csv"
PRODUCT_RFM_CSV = "../data/product_rfm.csv"

print("=" * 70)
print("TAHAP 3 (VERSI PRODUK) - DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(CLEAN_CSV, parse_dates=["InvoiceDate"])
n_awal = len(df)
print(f"\nJumlah baris data bersih (level transaksi, dari tahap sebelumnya): {n_awal:,}")
print(f"Jumlah produk unik (StockCode) sebelum dibersihkan: {df['StockCode'].nunique():,}")

# -------------------------------------------------------------------
# 1. Menghapus kode non-produk (biaya administratif)
# -------------------------------------------------------------------
print("\n[1] Menghapus kode non-produk (biaya administratif, bukan barang fisik)")
NON_PRODUCT_CODES = ["POST", "DOT", "M", "BANK CHARGES", "PADS", "CRUK", "C2", "S", "AMAZONFEE"]
before = len(df)
is_non_product = df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
removed_codes = df.loc[is_non_product].groupby("StockCode")["Description"].first()
print("    Kode yang dihapus beserta deskripsinya:")
for code, desc in removed_codes.items():
    n_rows = (df["StockCode"] == code).sum()
    print(f"      - {code:15s}: {desc:35s} ({n_rows:,} baris)")

df = df[~is_non_product].copy()
print(f"\n    Total baris dihapus: {before - len(df):,} -> sisa {len(df):,} baris")
print(f"    Jumlah produk unik setelah dibersihkan: {df['StockCode'].nunique():,}")

# -------------------------------------------------------------------
# 2. Menentukan Description representatif per StockCode (modus)
# -------------------------------------------------------------------
print("\n[2] Menentukan nama produk (Description) representatif per StockCode")
print("    Metode: menggunakan Description dengan frekuensi kemunculan TERBANYAK (modus)")

desc_variation = df.groupby("StockCode")["Description"].nunique()
n_multi_desc = (desc_variation > 1).sum()
print(f"    Jumlah StockCode dengan >1 variasi Description: {n_multi_desc:,}")

# Modus per StockCode (Description paling sering muncul)
mode_desc = (
    df.groupby("StockCode")["Description"]
    .agg(lambda x: x.value_counts().idxmax())
    .rename("ProductName")
)

print("\n    Contoh hasil pemilihan nama representatif (untuk StockCode dengan variasi nama):")
example_codes = desc_variation[desc_variation > 1].index[:3]
for code in example_codes:
    variations = df.loc[df["StockCode"] == code, "Description"].value_counts()
    print(f"      StockCode {code}: variasi = {dict(variations)} -> dipilih: '{mode_desc[code]}'")

df.to_csv(PRODUCT_CLEAN_CSV, index=False)
print(f"\n    Data transaksi bersih (level produk) disimpan ke: {PRODUCT_CLEAN_CSV}")

# -------------------------------------------------------------------
# 3 & 4. Mengubah data transaksi menjadi data per PRODUK + bentuk atribut R-F-Q
# -------------------------------------------------------------------
print("\n[3-4] Agregasi ke level produk & membentuk atribut Recency-Frequency-Quantity")

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"    Snapshot date (acuan hitung Recency): {snapshot_date}")

product_rfm = df.groupby("StockCode").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Quantity=("Quantity", "sum"),
).reset_index()

product_rfm = product_rfm.merge(mode_desc.reset_index(), on="StockCode", how="left")
product_rfm = product_rfm[["StockCode", "ProductName", "Recency", "Frequency", "Quantity"]]

print("\n    Definisi atribut per produk:")
print("    - Recency   : jumlah hari sejak produk ini TERAKHIR terjual hingga snapshot_date")
print("                  (makin kecil = produk masih aktif terjual baru-baru ini)")
print("    - Frequency : jumlah invoice/transaksi BERBEDA yang memuat produk ini")
print("                  (makin besar = makin sering dipesan pelanggan)")
print("    - Quantity  : total UNIT produk ini yang terjual sepanjang periode data")
print("                  (makin besar = makin cepat pergerakan stoknya/fast-moving)")

print(f"\n    Jumlah produk hasil agregasi: {len(product_rfm):,}")
print("\n    Statistik deskriptif atribut Recency-Frequency-Quantity:")
print(product_rfm[["Recency", "Frequency", "Quantity"]].describe())

product_rfm.to_csv(PRODUCT_RFM_CSV, index=False)
print(f"\n    Data RFM per produk disimpan ke: {PRODUCT_RFM_CSV}")

print("\n    10 produk dengan Quantity terjual tertinggi:")
print(product_rfm.sort_values("Quantity", ascending=False).head(10).to_string(index=False))

print("\n    10 produk dengan Recency terbesar (paling lama tidak terjual / berpotensi dead-stock):")
print(product_rfm.sort_values("Recency", ascending=False).head(10).to_string(index=False))

print("\n" + "=" * 70)
print("TAHAP 3 (VERSI PRODUK) SELESAI -> lanjut ke Tahap 4 (Implementasi K-Medoids)")
print("=" * 70)

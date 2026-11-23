"""
02_preprocessing_rfm.py
Tahap 3 - Data Preprocessing
Case Method: Segmentasi Pelanggan E-Commerce dengan Algoritma K-Medoids

Langkah:
1. Membersihkan data kosong (Description, CustomerID).
2. Menghapus/menangani transaksi pembatalan (InvoiceNo berawalan 'C', Quantity negatif).
3. Menghapus data duplikat & transaksi tidak valid (UnitPrice <= 0).
4. Menghitung total nilai transaksi (Quantity x UnitPrice).
5. Mengubah data transaksi (per baris item) menjadi data per pelanggan.
6. Membentuk atribut RFM (Recency, Frequency, Monetary).
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

RAW_CSV = "../data/online_retail_raw.csv"
CLEAN_CSV = "../data/online_retail_clean.csv"
RFM_CSV = "../data/rfm_customers.csv"

print("=" * 70)
print("TAHAP 3 - DATA PREPROCESSING")
print("=" * 70)

df = pd.read_csv(RAW_CSV, parse_dates=["InvoiceDate"])
n_awal = len(df)
print(f"\nJumlah baris awal: {n_awal:,}")

# -------------------------------------------------------------------
# 1. Membersihkan data kosong
# -------------------------------------------------------------------
print("\n[1] Membersihkan data kosong")
before = len(df)
df = df.dropna(subset=["CustomerID"])
print(f"    Hapus baris tanpa CustomerID : {before - len(df):,} baris dihapus -> sisa {len(df):,}")

before = len(df)
df["Description"] = df["Description"].fillna("UNKNOWN")
print(f"    Isi Description kosong dengan 'UNKNOWN' (tidak menghapus baris): {df['Description'].eq('UNKNOWN').sum():,} baris")

df["CustomerID"] = df["CustomerID"].astype(int)

# -------------------------------------------------------------------
# 2. Menghapus / menangani transaksi pembatalan
# -------------------------------------------------------------------
print("\n[2] Menangani transaksi pembatalan")
before = len(df)
is_cancel = df["InvoiceNo"].astype(str).str.startswith("C")
df = df[~is_cancel]
print(f"    Hapus transaksi pembatalan (InvoiceNo awalan 'C'): {before - len(df):,} baris dihapus -> sisa {len(df):,}")

before = len(df)
df = df[df["Quantity"] > 0]
print(f"    Hapus baris dengan Quantity <= 0 (retur/anomali)  : {before - len(df):,} baris dihapus -> sisa {len(df):,}")

# -------------------------------------------------------------------
# 3. Menghapus duplikat & transaksi tidak valid
# -------------------------------------------------------------------
print("\n[3] Menghapus duplikat & transaksi tidak valid")
before = len(df)
df = df.drop_duplicates()
print(f"    Hapus baris duplikat         : {before - len(df):,} baris dihapus -> sisa {len(df):,}")

before = len(df)
df = df[df["UnitPrice"] > 0]
print(f"    Hapus baris UnitPrice <= 0   : {before - len(df):,} baris dihapus -> sisa {len(df):,}")

print(f"\n    Total baris dihapus selama cleaning: {n_awal - len(df):,} ({(n_awal-len(df))/n_awal*100:.2f}%)")
print(f"    Baris data bersih tersisa           : {len(df):,}")
print(f"    Jumlah pelanggan unik tersisa       : {df['CustomerID'].nunique():,}")

# -------------------------------------------------------------------
# 4. Menghitung total nilai transaksi
# -------------------------------------------------------------------
print("\n[4] Menghitung total nilai transaksi (TotalPrice = Quantity x UnitPrice)")
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
print(df[["InvoiceNo", "Quantity", "UnitPrice", "TotalPrice"]].head())

df.to_csv(CLEAN_CSV, index=False)
print(f"\n    Data bersih disimpan ke: {CLEAN_CSV}")

# -------------------------------------------------------------------
# 5 & 6. Mengubah data transaksi menjadi data per pelanggan + bentuk RFM
# -------------------------------------------------------------------
print("\n[5-6] Agregasi ke level pelanggan & membentuk atribut RFM")

# Snapshot date = 1 hari setelah transaksi terakhir dalam dataset
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
print(f"    Snapshot date (acuan hitung Recency): {snapshot_date}")

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum"),
).reset_index()

print("\n    Definisi atribut RFM:")
print("    - Recency   : jumlah hari sejak transaksi terakhir pelanggan hingga snapshot_date (makin kecil = makin baru belanja)")
print("    - Frequency : jumlah invoice/transaksi unik yang pernah dilakukan pelanggan")
print("    - Monetary  : total nilai (GBP) seluruh transaksi yang pernah dilakukan pelanggan")

print(f"\n    Jumlah pelanggan hasil agregasi: {len(rfm):,}")
print("\n    Statistik deskriptif RFM:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())

# Buang pelanggan dengan Monetary <= 0 (edge case sisa setelah cleaning, jika ada)
before = len(rfm)
rfm = rfm[rfm["Monetary"] > 0]
if before - len(rfm) > 0:
    print(f"\n    Hapus {before-len(rfm)} pelanggan dengan Monetary <= 0 (sisa anomali)")

rfm.to_csv(RFM_CSV, index=False)
print(f"\n    Data RFM per pelanggan disimpan ke: {RFM_CSV}")
print(rfm.head(10))

print("\n" + "=" * 70)
print("TAHAP 3 SELESAI -> lanjut ke Tahap 4 (Implementasi K-Medoids)")
print("=" * 70)

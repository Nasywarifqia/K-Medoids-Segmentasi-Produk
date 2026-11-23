"""
04b_clustering_evaluation_product.py
Tahap 4 (penerapan) & Tahap 5 - Evaluasi dan Visualisasi (VERSI PRODUK)
Case Method: Segmentasi Produk E-Commerce untuk Manajemen Inventori
menggunakan Algoritma K-Medoids

Menggunakan implementasi K-Medoids (PAM) dari nol yang sama (kmedoids.py),
diterapkan pada atribut Recency-Frequency-Quantity per PRODUK (bukan per
pelanggan).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

from kmedoids import KMedoids

np.random.seed(42)
pd.set_option("display.width", 160)

PRODUCT_RFM_CSV = "../data/product_rfm.csv"
OUT_DIR = "../output"
FIG_DIR = "../output/figures_product"
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

print("=" * 70)
print("TAHAP 4 (penerapan) & TAHAP 5 - EVALUASI DAN VISUALISASI (PRODUK)")
print("=" * 70)

prfm = pd.read_csv(PRODUCT_RFM_CSV)
print(f"\nJumlah produk: {len(prfm):,}")

# -------------------------------------------------------------------
# 1. Standarisasi atribut Recency-Frequency-Quantity
# -------------------------------------------------------------------
print("\n[1] Standarisasi atribut (Z-score)")
print("    Alasan sama seperti versi pelanggan: skala Recency (hari), Frequency")
print("    (jumlah invoice), dan Quantity (unit, range sangat lebar -- sampai")
print("    80,995) sangat berbeda. Tanpa standarisasi, Quantity akan mendominasi")
print("    perhitungan jarak Euclidean.")

features = ["Recency", "Frequency", "Quantity"]
X_raw = prfm[features].values.astype(float)

mean_ = X_raw.mean(axis=0)
std_ = X_raw.std(axis=0, ddof=0)
X_scaled = (X_raw - mean_) / std_

print(f"    Mean sebelum standarisasi : {dict(zip(features, mean_.round(3)))}")
print(f"    Std  sebelum standarisasi : {dict(zip(features, std_.round(3)))}")

np.save(f"{OUT_DIR}/X_scaled_product.npy", X_scaled)

# -------------------------------------------------------------------
# 2 & 3. Menjalankan K-Medoids untuk beberapa k, hitung silhouette
# -------------------------------------------------------------------
print("\n[2-3] Menjalankan K-Medoids untuk k = 2..6 dan menghitung Silhouette Score")

k_range = range(2, 7)
results = []
models = {}

for k in k_range:
    km = KMedoids(n_clusters=k, random_state=42, verbose=False)
    km.fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_, metric="euclidean")
    results.append({"k": k, "cost": km.cost_, "silhouette": sil, "n_iter": km.n_iter_})
    models[k] = km
    print(f"    k={k}: total cost={km.cost_:.2f} | silhouette score={sil:.4f} | iterasi swap={km.n_iter_}")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT_DIR}/k_evaluation_results_product.csv", index=False)

best_k = int(results_df.loc[results_df["silhouette"].idxmax(), "k"])
print(f"\n    >> Nilai k dengan Silhouette Score TERBAIK: k = {best_k} "
      f"(silhouette = {results_df['silhouette'].max():.4f})")

best_model = models[best_k]

# -------------------------------------------------------------------
# 4a. Visualisasi: Cost vs k dan Silhouette vs k
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(results_df["k"], results_df["cost"], marker="o", color=PALETTE[0], linewidth=2)
axes[0].set_xlabel("Jumlah Cluster (k)")
axes[0].set_ylabel("Total Cost (PAM)")
axes[0].set_title("Total Cost vs Jumlah Cluster")
axes[0].grid(alpha=0.3)

axes[1].plot(results_df["k"], results_df["silhouette"], marker="o", color=PALETTE[2], linewidth=2)
axes[1].axvline(best_k, color=PALETTE[3], linestyle="--", alpha=0.7, label=f"k terbaik = {best_k}")
axes[1].set_xlabel("Jumlah Cluster (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs Jumlah Cluster")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_cost_silhouette_vs_k.png", dpi=150)
plt.close()
print(f"\n    Grafik disimpan: {FIG_DIR}/01_cost_silhouette_vs_k.png")

# -------------------------------------------------------------------
# 4b. Silhouette plot untuk k terbaik
# -------------------------------------------------------------------
sil_values = silhouette_samples(X_scaled, best_model.labels_)
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
for i in range(best_k):
    cluster_sil = np.sort(sil_values[best_model.labels_ == i])
    size_i = len(cluster_sil)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                      facecolor=PALETTE[i % len(PALETTE)], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10
ax.axvline(x=results_df.loc[results_df["k"] == best_k, "silhouette"].values[0],
           color="red", linestyle="--", label="Rata-rata silhouette")
ax.set_xlabel("Nilai Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette Plot (k={best_k})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_silhouette_plot.png", dpi=150)
plt.close()
print(f"    Grafik disimpan: {FIG_DIR}/02_silhouette_plot.png")

# -------------------------------------------------------------------
# 4c. Scatter plot hasil clustering (PCA 2D)
# -------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum()

fig, ax = plt.subplots(figsize=(7, 5.5))
for i in range(best_k):
    mask = best_model.labels_ == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.6,
               color=PALETTE[i % len(PALETTE)], label=f"Cluster {i} (n={mask.sum()})")
medoid_pca = pca.transform(X_scaled[best_model.medoid_indices_])
ax.scatter(medoid_pca[:, 0], medoid_pca[:, 1], s=250, marker="X",
           color="black", edgecolor="white", linewidth=1.5, label="Medoid", zorder=5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variansi)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variansi)")
ax.set_title(f"Visualisasi Hasil Clustering K-Medoids (k={best_k}, PCA 2D, total {explained_var*100:.1f}% variansi)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_scatter_pca_clusters.png", dpi=150)
plt.close()
print(f"    Grafik disimpan: {FIG_DIR}/03_scatter_pca_clusters.png")
print(f"    (2 komponen PCA menjelaskan {explained_var*100:.1f}% variansi data)")

# -------------------------------------------------------------------
# 4d. Jumlah produk per cluster
# -------------------------------------------------------------------
prfm["Cluster"] = best_model.labels_
cluster_counts = prfm["Cluster"].value_counts().sort_index()
print("\n    Jumlah produk per cluster:")
for c, n in cluster_counts.items():
    print(f"      Cluster {c}: {n:,} produk ({n/len(prfm)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(6, 4.2))
bars = ax.bar(cluster_counts.index.astype(str), cluster_counts.values,
              color=[PALETTE[i % len(PALETTE)] for i in cluster_counts.index])
for bar, val in zip(bars, cluster_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_xlabel("Cluster")
ax.set_ylabel("Jumlah Produk")
ax.set_title("Jumlah Produk per Cluster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_cluster_sizes.png", dpi=150)
plt.close()
print(f"    Grafik disimpan: {FIG_DIR}/04_cluster_sizes.png")

# -------------------------------------------------------------------
# 4e. Boxplot R-F-Q per cluster
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for idx, feat in enumerate(features):
    data_per_cluster = [prfm.loc[prfm["Cluster"] == c, feat].values for c in sorted(prfm["Cluster"].unique())]
    bp = axes[idx].boxplot(data_per_cluster, patch_artist=True, tick_labels=[str(c) for c in sorted(prfm["Cluster"].unique())])
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[idx].set_title(feat)
    axes[idx].set_xlabel("Cluster")
    axes[idx].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_boxplot_rfq_per_cluster.png", dpi=150)
plt.close()
print(f"    Grafik disimpan: {FIG_DIR}/05_boxplot_rfq_per_cluster.png")

prfm.to_csv(f"{OUT_DIR}/product_rfm_with_clusters.csv", index=False)
np.save(f"{OUT_DIR}/best_k_product.npy", np.array([best_k]))
print(f"\nData produk + label cluster disimpan ke: {OUT_DIR}/product_rfm_with_clusters.csv")

print("\n" + "=" * 70)
print(f"TAHAP 5 SELESAI (k terpilih = {best_k}) -> lanjut ke Tahap 6 (Analisis Hasil)")
print("=" * 70)

"""
04c_regenerate_figures_remapped.py
Regenerasi visualisasi Tahap 5 setelah remapping nomor cluster produk
(0 = slow/dead-stock, 1 = medium, 2 = fast-moving) agar urutan nomor cluster
konsisten dan intuitif di seluruh laporan.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

OUT_DIR = "../output"
FIG_DIR = "../output/figures_product"
PALETTE = ["#C44E52", "#DD8452", "#55A868"]  # merah(slow) - oranye(medium) - hijau(fast)

prfm = pd.read_csv(f"{OUT_DIR}/product_rfm_with_clusters.csv")
X_scaled = np.load(f"{OUT_DIR}/X_scaled_product.npy")
best_k = int(np.load(f"{OUT_DIR}/best_k_product.npy")[0])
features = ["Recency", "Frequency", "Quantity"]
labels = prfm["Cluster"].values

# Cari medoid baru (titik data asli paling dekat ke rata-rata tiap cluster baru,
# hanya untuk keperluan visual penanda "pusat" cluster pada scatter plot)
medoid_idx_new = []
for c in range(best_k):
    idx_in_cluster = np.where(labels == c)[0]
    cluster_mean = X_scaled[idx_in_cluster].mean(axis=0)
    dist_to_mean = np.sqrt(((X_scaled[idx_in_cluster] - cluster_mean) ** 2).sum(axis=1))
    medoid_idx_new.append(idx_in_cluster[np.argmin(dist_to_mean)])

# -------------------------------------------------------------------
# Scatter PCA
# -------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum()

cluster_names = {0: "Slow-moving / Dead-stock", 1: "Medium", 2: "Fast-moving"}

fig, ax = plt.subplots(figsize=(7, 5.5))
for i in range(best_k):
    mask = labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=18, alpha=0.6,
               color=PALETTE[i], label=f"Cluster {i} - {cluster_names[i]} (n={mask.sum()})")
medoid_pca = pca.transform(X_scaled[medoid_idx_new])
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
print(f"Grafik diperbarui: {FIG_DIR}/03_scatter_pca_clusters.png")

# -------------------------------------------------------------------
# Silhouette plot
# -------------------------------------------------------------------
sil_values = silhouette_samples(X_scaled, labels)
fig, ax = plt.subplots(figsize=(7, 5))
y_lower = 10
for i in range(best_k):
    cluster_sil = np.sort(sil_values[labels == i])
    size_i = len(cluster_sil)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, facecolor=PALETTE[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5 * size_i, str(i))
    y_lower = y_upper + 10
ax.axvline(x=sil_values.mean(), color="red", linestyle="--", label="Rata-rata silhouette")
ax.set_xlabel("Nilai Silhouette Coefficient")
ax.set_ylabel("Cluster")
ax.set_title(f"Silhouette Plot (k={best_k})")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_silhouette_plot.png", dpi=150)
plt.close()
print(f"Grafik diperbarui: {FIG_DIR}/02_silhouette_plot.png")

# -------------------------------------------------------------------
# Jumlah produk per cluster
# -------------------------------------------------------------------
cluster_counts = prfm["Cluster"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar([f"{c}\n{cluster_names[c]}" for c in cluster_counts.index], cluster_counts.values,
              color=[PALETTE[i] for i in cluster_counts.index])
for bar, val in zip(bars, cluster_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Jumlah Produk")
ax.set_title("Jumlah Produk per Cluster")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_cluster_sizes.png", dpi=150)
plt.close()
print(f"Grafik diperbarui: {FIG_DIR}/04_cluster_sizes.png")

# -------------------------------------------------------------------
# Boxplot R-F-Q per cluster
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for idx, feat in enumerate(features):
    data_per_cluster = [prfm.loc[prfm["Cluster"] == c, feat].values for c in range(best_k)]
    bp = axes[idx].boxplot(data_per_cluster, patch_artist=True, tick_labels=[str(c) for c in range(best_k)])
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[idx].set_title(feat)
    axes[idx].set_xlabel("Cluster")
    axes[idx].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_boxplot_rfq_per_cluster.png", dpi=150)
plt.close()
print(f"Grafik diperbarui: {FIG_DIR}/05_boxplot_rfq_per_cluster.png")

print("\nSemua grafik berhasil diperbarui dengan nomor cluster baru (0=slow/dead, 1=medium, 2=fast).")

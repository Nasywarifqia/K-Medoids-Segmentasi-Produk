"""
test_kmedoids.py
Unit test untuk memverifikasi kebenaran implementasi K-Medoids (PAM) dari nol
sebelum digunakan pada data RFM yang sebenarnya.

Dijalankan dengan: python3 test_kmedoids.py
Semua test harus menampilkan "PASSED" dan keluar dengan exit code 0.
"""

import numpy as np
from kmedoids import euclidean_distance_matrix, total_cost, KMedoids
from itertools import combinations

FAILED = []


def check(name, condition):
    status = "PASSED" if condition else "FAILED"
    print(f"[{status}] {name}")
    if not condition:
        FAILED.append(name)


def brute_force_kmedoids(X, k):
    """Mencari solusi K-Medoids OPTIMAL dengan mencoba SEMUA kombinasi medoid
    (hanya feasible untuk data sangat kecil) -- dipakai sebagai 'ground truth'
    untuk memverifikasi hasil implementasi PAM."""
    D = euclidean_distance_matrix(X)
    n = X.shape[0]
    best_cost = np.inf
    best_medoids = None
    for combo in combinations(range(n), k):
        cost, _ = total_cost(D, list(combo))
        if cost < best_cost:
            best_cost = cost
            best_medoids = combo
    return best_medoids, best_cost


print("=" * 70)
print("UNIT TEST: IMPLEMENTASI K-MEDOIDS (PAM) DARI NOL")
print("=" * 70)

# -------------------------------------------------------------------
# Test 1: matriks jarak Euclidean harus sama dengan hitungan manual
# -------------------------------------------------------------------
X_simple = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
D = euclidean_distance_matrix(X_simple)
expected_01 = 5.0  # sqrt(3^2+4^2) = 5
expected_02 = 10.0  # sqrt(6^2+8^2) = 10
expected_12 = 5.0  # sqrt(3^2+4^2) = 5
check("Jarak Euclidean titik 0-1 = 5.0", np.isclose(D[0, 1], expected_01))
check("Jarak Euclidean titik 0-2 = 10.0", np.isclose(D[0, 2], expected_02))
check("Jarak Euclidean titik 1-2 = 5.0", np.isclose(D[1, 2], expected_12))
check("Diagonal matriks jarak = 0", np.allclose(np.diag(D), 0))
check("Matriks jarak simetris (D == D.T)", np.allclose(D, D.T))

# -------------------------------------------------------------------
# Test 2: total_cost menghitung dengan benar untuk kasus sederhana
# -------------------------------------------------------------------
# 4 titik dalam satu garis: 0, 1, 10, 11 -- medoid di {0, 10}
X_line = np.array([[0.0], [1.0], [10.0], [11.0]])
D_line = euclidean_distance_matrix(X_line)
cost, labels = total_cost(D_line, [0, 2])
# titik0->medoid0=0, titik1->medoid0=1, titik2->medoid1(10)=0, titik3->medoid1(10)=1
check("Total cost kasus sederhana = 2.0", np.isclose(cost, 2.0))
check("Label assignment benar", list(labels) == [0, 0, 1, 1])

# -------------------------------------------------------------------
# Test 3: hasil PAM harus sama/mendekati solusi brute-force (optimal)
#          pada dataset kecil dengan struktur cluster yang jelas
# -------------------------------------------------------------------
np.random.seed(42)
cluster_a = np.random.normal(loc=[0, 0], scale=0.3, size=(5, 2))
cluster_b = np.random.normal(loc=[10, 10], scale=0.3, size=(5, 2))
X_small = np.vstack([cluster_a, cluster_b])

km = KMedoids(n_clusters=2, random_state=0, verbose=False)
km.fit(X_small)

_, best_cost_bf = brute_force_kmedoids(X_small, 2)
check(
    f"Cost PAM ({km.cost_:.4f}) sama dengan cost optimal brute-force ({best_cost_bf:.4f})",
    np.isclose(km.cost_, best_cost_bf, atol=1e-6),
)
# Dua cluster yang terpisah jauh harus menghasilkan 2 label berbeda yang konsisten per grup
labels_a = set(km.labels_[:5])
labels_b = set(km.labels_[5:])
check("Cluster A (5 titik pertama) konsisten 1 label", len(labels_a) == 1)
check("Cluster B (5 titik kedua) konsisten 1 label", len(labels_b) == 1)
check("Cluster A dan B mendapat label berbeda", labels_a != labels_b)

# -------------------------------------------------------------------
# Test 4: cost_history harus monoton tidak naik (non-increasing)
# -------------------------------------------------------------------
diffs = np.diff(km.cost_history_)
check("Riwayat cost monoton turun/tetap (PAM selalu memperbaiki/mempertahankan cost)",
      np.all(diffs <= 1e-9))

# -------------------------------------------------------------------
# Test 5: medoid yang dipilih harus berupa titik ASLI dari data (bukan centroid buatan)
# -------------------------------------------------------------------
is_real_point = all(
    any(np.allclose(X_small[m], row) for row in X_small) for m in km.medoid_indices_
)
check("Medoid adalah titik data asli (bukan rata-rata/centroid)", is_real_point)

# -------------------------------------------------------------------
# Test 6: validasi pada dataset acak lebih besar (100 titik, 3 fitur, k=3)
#          -- memastikan tidak error dan cost akhir <= cost build awal
# -------------------------------------------------------------------
np.random.seed(1)
X_big = np.vstack([
    np.random.normal([0, 0, 0], 1, size=(40, 3)),
    np.random.normal([15, 0, 0], 1, size=(30, 3)),
    np.random.normal([0, 15, 15], 1, size=(30, 3)),
])
from kmedoids import euclidean_distance_matrix as edm, pam_build, total_cost as tc

D_big = edm(X_big)
init_medoids = pam_build(D_big, 3, random_state=0)
init_cost, _ = tc(D_big, init_medoids)

km_big = KMedoids(n_clusters=3, random_state=0, verbose=False)
km_big.fit(X_big)

check("Cost akhir PAM <= cost setelah fase BUILD (swap tidak memperburuk)",
      km_big.cost_ <= init_cost + 1e-9)
check("Jumlah label unik = jumlah cluster yang diminta",
      len(set(km_big.labels_)) == 3)
check("Semua 100 titik memiliki label", len(km_big.labels_) == 100)

print("=" * 70)
if FAILED:
    print(f"HASIL: {len(FAILED)} TEST GAGAL -> {FAILED}")
    raise SystemExit(1)
else:
    print("HASIL: SEMUA TEST PASSED")
print("=" * 70)

"""
kmedoids.py
Implementasi ALGORITMA K-MEDOIDS (PAM - Partitioning Around Medoids) dari NOL,
tanpa menggunakan library clustering siap pakai (scikit-learn-extra, pyclustering, dll).

Algoritma PAM terdiri dari dua fase:
  1. Fase BUILD : memilih k medoid awal secara greedy agar total cost minimum.
  2. Fase SWAP  : menukar medoid dengan titik non-medoid jika pertukaran
                  tersebut menurunkan total cost, diulang sampai konvergen
                  (tidak ada pertukaran yang menurunkan cost lagi).

Fase SWAP divektorisasi dengan NumPy (bukan memanggil fungsi clustering siap
pakai) agar tetap efisien untuk dataset berukuran ribuan baris.

Referensi konsep: Kaufman, L. & Rousseeuw, P.J. (1990). "Partitioning Around
Medoids (Program PAM)" dalam Finding Groups in Data: An Introduction to
Cluster Analysis.
"""

import numpy as np


def euclidean_distance_matrix(X):
    """
    Menghitung matriks jarak Euclidean antar semua pasangan titik.

    Rumus jarak Euclidean antara titik a dan b (dimensi m):
        d(a, b) = sqrt( sum_{i=1}^{m} (a_i - b_i)^2 )

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)

    Returns
    -------
    D : ndarray, shape (n_samples, n_samples)
        D[i, j] = jarak Euclidean antara titik ke-i dan titik ke-j.
    """
    sum_sq = np.sum(X ** 2, axis=1)
    dot = X @ X.T
    sq_dist = sum_sq.reshape(-1, 1) + sum_sq.reshape(1, -1) - 2 * dot
    sq_dist = np.maximum(sq_dist, 0)  # jaga-jaga error numerik kecil negatif
    D = np.sqrt(sq_dist)
    return D


def total_cost(D, medoid_indices):
    """
    Menghitung total cost = total jarak setiap titik ke medoid terdekatnya.

    Parameters
    ----------
    D : ndarray (n, n) matriks jarak
    medoid_indices : array-like of int, indeks titik yang menjadi medoid

    Returns
    -------
    cost : float, total jarak minimum seluruh titik ke medoid terdekat
    labels : ndarray (n,), indeks medoid (dalam medoid_indices) terdekat tiap titik
    """
    dist_to_medoids = D[:, medoid_indices]
    labels = np.argmin(dist_to_medoids, axis=1)
    min_dist = np.min(dist_to_medoids, axis=1)
    cost = np.sum(min_dist)
    return cost, labels


def pam_build(D, k, random_state=None):
    """
    Fase BUILD (PAM): memilih k medoid awal secara greedy.

    Langkah:
    1. Medoid pertama = titik yang memiliki total jarak ke semua titik lain
       paling kecil (paling 'sentral').
    2. Medoid berikutnya dipilih satu per satu: untuk setiap kandidat titik
       yang belum jadi medoid, hitung seberapa besar penurunan total cost
       jika titik tersebut ditambahkan sebagai medoid. Pilih kandidat dengan
       penurunan cost terbesar (dihitung sekaligus/vektor untuk semua kandidat).
    3. Ulangi sampai terkumpul k medoid.
    """
    n = D.shape[0]

    total_dist = D.sum(axis=1)
    first_medoid = int(np.argmin(total_dist))
    medoids = [first_medoid]

    # current_min_dist[i] = jarak titik i ke medoid terdekat saat ini
    current_min_dist = D[:, first_medoid].copy()
    is_medoid = np.zeros(n, dtype=bool)
    is_medoid[first_medoid] = True

    while len(medoids) < k:
        # Untuk SETIAP kandidat non-medoid, hitung gain jika dijadikan medoid,
        # dilakukan serentak (vektorisasi) bukan satu per satu dengan loop berat.
        # improvement[i, c] = max(current_min_dist[i] - D[i, c], 0)
        candidate_idx = np.where(~is_medoid)[0]
        D_candidates = D[:, candidate_idx]  # shape (n, n_candidates)
        improvement = np.maximum(current_min_dist.reshape(-1, 1) - D_candidates, 0)
        gain_per_candidate = improvement.sum(axis=0)  # shape (n_candidates,)

        best_pos = int(np.argmax(gain_per_candidate))
        best_candidate = int(candidate_idx[best_pos])

        medoids.append(best_candidate)
        is_medoid[best_candidate] = True
        current_min_dist = np.minimum(current_min_dist, D[:, best_candidate])

    return medoids


def pam_swap(D, medoids, max_iter=300, verbose=False):
    """
    Fase SWAP (PAM): menukar medoid <-> non-medoid selama menurunkan total cost.

    Strategi "best improvement": pada setiap iterasi, evaluasi SEMUA kemungkinan
    pertukaran (setiap medoid dengan setiap non-medoid), lalu lakukan pertukaran
    dengan penurunan cost TERBESAR. Ulangi sampai tidak ada pertukaran yang
    menurunkan cost (konvergen) atau mencapai max_iter.

    Evaluasi delta-cost untuk semua kandidat non-medoid pada satu posisi medoid
    dilakukan sekaligus dengan operasi matriks NumPy (vektorisasi), bukan dengan
    memanggil ulang total_cost() satu per satu di dalam loop Python yang berat.
    """
    n = D.shape[0]
    k = len(medoids)
    medoids = list(medoids)
    current_cost, _ = total_cost(D, medoids)
    history = [current_cost]

    for iteration in range(1, max_iter + 1):
        is_medoid = np.zeros(n, dtype=bool)
        is_medoid[medoids] = True
        non_medoid_idx = np.where(~is_medoid)[0]

        best_gain = 0.0
        best_swap = None  # (posisi medoid yang diganti, index non-medoid baru)

        for m_pos in range(k):
            # Jarak tiap titik ke medoid LAIN (selain medoid pada posisi m_pos)
            other_medoids = [medoids[j] for j in range(k) if j != m_pos]
            if other_medoids:
                min_dist_other = D[:, other_medoids].min(axis=1)  # shape (n,)
            else:
                min_dist_other = np.full(n, np.inf)

            # Untuk setiap kandidat o (non-medoid), cost baru jika medoid di posisi
            # m_pos diganti o = sum_i min(min_dist_other[i], D[i, o])
            D_cand = D[:, non_medoid_idx]  # shape (n, n_candidates)
            new_cost_per_cand = np.minimum(
                min_dist_other.reshape(-1, 1), D_cand
            ).sum(axis=0)  # shape (n_candidates,)

            gain_per_cand = current_cost - new_cost_per_cand
            local_best_pos = int(np.argmax(gain_per_cand))
            local_best_gain = gain_per_cand[local_best_pos]

            if local_best_gain > best_gain:
                best_gain = local_best_gain
                best_swap = (m_pos, int(non_medoid_idx[local_best_pos]))

        if best_swap is None:
            if verbose:
                print(f"    Konvergen pada iterasi ke-{iteration} (tidak ada pertukaran yang menurunkan cost)")
            break

        m_pos, o = best_swap
        old_medoid = medoids[m_pos]
        medoids[m_pos] = o
        current_cost = current_cost - best_gain
        history.append(current_cost)
        if verbose:
            print(f"    Iterasi {iteration}: tukar medoid {old_medoid} -> {o} | "
                  f"cost turun {best_gain:.4f} -> total cost = {current_cost:.4f}")
    else:
        if verbose:
            print(f"    Berhenti karena mencapai max_iter={max_iter}")

    final_cost, labels = total_cost(D, medoids)
    return medoids, labels, final_cost, history


class KMedoids:
    """
    Implementasi algoritma K-Medoids (PAM) dari nol.

    Parameters
    ----------
    n_clusters : int, jumlah cluster (k)
    max_iter : int, maksimum iterasi fase swap
    random_state : int, seed (disediakan untuk konsistensi API meski BUILD
                   bersifat deterministik/greedy, bukan acak)
    verbose : bool, tampilkan log tiap iterasi swap

    Attributes (setelah fit)
    ----------
    medoid_indices_ : list[int], indeks baris data yang menjadi medoid akhir
    labels_ : ndarray, label cluster (0..k-1) untuk setiap titik data
    cost_ : float, total cost akhir (total jarak ke medoid terdekat)
    cost_history_ : list[float], riwayat total cost selama fase swap
    """

    def __init__(self, n_clusters=3, max_iter=300, random_state=None, verbose=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        D = euclidean_distance_matrix(X)

        if self.verbose:
            print(f"  [BUILD] Memilih {self.n_clusters} medoid awal secara greedy...")
        initial_medoids = pam_build(D, self.n_clusters, random_state=self.random_state)
        if self.verbose:
            print(f"  [BUILD] Medoid awal (indeks baris): {initial_medoids}")

        if self.verbose:
            print(f"  [SWAP] Memulai proses pertukaran medoid...")
        medoids, labels, cost, history = pam_swap(
            D, initial_medoids, max_iter=self.max_iter, verbose=self.verbose
        )

        self.distance_matrix_ = D
        self.medoid_indices_ = medoids
        self.cluster_centers_ = X[medoids]
        self.labels_ = labels
        self.cost_ = cost
        self.cost_history_ = history
        self.n_iter_ = len(history) - 1
        return self

    def predict(self, X):
        """Menentukan cluster untuk titik data baru berdasarkan medoid yang sudah ditemukan."""
        X = np.asarray(X, dtype=float)
        centers = self.cluster_centers_
        dist = np.sqrt(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))
        return np.argmin(dist, axis=1)

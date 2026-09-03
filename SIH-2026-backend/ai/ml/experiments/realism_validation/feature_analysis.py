"""Feature separability, ranking, and pairwise overlap analysis."""

from __future__ import annotations

import csv
from typing import Any

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import CLASS_ID, CLASS_NAMES, FEATURE_NAMES
from .data import AuditData


def _bhattacharyya_overlap(x: np.ndarray, y: np.ndarray) -> float:
    """Gaussian overlap coefficient in [0,1]; higher = more overlap."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mu1, mu2 = float(np.mean(x)), float(np.mean(y))
    v1 = max(float(np.var(x, ddof=1)), 1e-12)
    v2 = max(float(np.var(y, ddof=1)), 1e-12)
    term = 0.25 * np.log(0.25 * (v1 / v2 + v2 / v1 + 2.0)) + 0.25 * ((mu1 - mu2) ** 2) / (v1 + v2)
    return float(np.exp(-term))


def _fisher_ratio(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    between = (np.mean(x) - np.mean(y)) ** 2
    within = np.var(x, ddof=1) + np.var(y, ddof=1)
    return float(between / max(within, 1e-12))


def _pairwise_feature_stats(
    X: np.ndarray,
    y: np.ndarray,
    class_a: str,
    class_b: str,
) -> list[dict[str, Any]]:
    ia, ib = CLASS_ID[class_a], CLASS_ID[class_b]
    rows = []
    for fi, fname in enumerate(FEATURE_NAMES):
        xa = X[y == ia, fi]
        xb = X[y == ib, fi]
        overlap = _bhattacharyya_overlap(xa, xb)
        fisher = _fisher_ratio(xa, xb)
        try:
            _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append(
            {
                "feature": fname,
                "class_a": class_a,
                "class_b": class_b,
                "mean_a": float(np.mean(xa)),
                "mean_b": float(np.mean(xb)),
                "std_a": float(np.std(xa, ddof=0)),
                "std_b": float(np.std(xb, ddof=0)),
                "bhattacharyya_overlap": overlap,
                "fisher_ratio": fisher,
                "mannwhitney_p": float(p),
                "separability_score": float(fisher * (1.0 - overlap)) if np.isfinite(overlap) else float(fisher),
            }
        )
    rows.sort(key=lambda r: r["separability_score"], reverse=True)
    return rows


def feature_separability_analysis(data: AuditData) -> dict[str, Any]:
    X = data.dev_features()
    y = data.dev_labels()

    mi = mutual_info_classif(X, y, random_state=42)
    global_rows = []
    for fi, fname in enumerate(FEATURE_NAMES):
        groups = [X[y == cid, fi] for cid in CLASS_NAMES]
        try:
            f_stat, p_val = stats.f_oneway(*groups)
            eta_sq = float(f_stat / (f_stat + len(y) - len(CLASS_NAMES))) if f_stat >= 0 else 0.0
        except ValueError:
            f_stat, p_val, eta_sq = 0.0, 1.0, 0.0
        global_rows.append(
            {
                "feature": fname,
                "mutual_information": float(mi[fi]),
                "anova_f": float(f_stat),
                "anova_p": float(p_val),
                "anova_eta_squared_proxy": eta_sq,
                "variance": float(np.var(X[:, fi], ddof=0)),
            }
        )
    global_rows.sort(key=lambda r: r["mutual_information"], reverse=True)

    pair_rankings = {
        "fsk_vs_tone": _pairwise_feature_stats(X, y, "fsk", "tone"),
        "ook_vs_benign": _pairwise_feature_stats(X, y, "ook", "benign"),
        "ook_vs_tone": _pairwise_feature_stats(X, y, "ook", "tone"),
        "tone_vs_benign": _pairwise_feature_stats(X, y, "tone", "benign"),
    }

    diagnostic_trees = {}
    for pair_name, (a, b) in {
        "fsk_vs_tone": ("fsk", "tone"),
        "ook_vs_benign": ("ook", "benign"),
    }.items():
        ia, ib = CLASS_ID[a], CLASS_ID[b]
        mask = (y == ia) | (y == ib)
        Xp = X[mask]
        yp = (y[mask] == ib).astype(int)
        best_feat = None
        best_acc = 0.0
        for fi, fname in enumerate(FEATURE_NAMES):
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)
            dt.fit(Xp[:, [fi]], yp)
            acc = float(dt.score(Xp[:, [fi]], yp))
            if acc > best_acc:
                best_acc = acc
                best_feat = fname
        diagnostic_trees[pair_name] = {
            "best_single_feature": best_feat,
            "best_single_feature_accuracy": best_acc,
            "note": "Diagnostic only — not for production.",
        }

    return {
        "global_feature_ranking": global_rows,
        "pairwise_rankings": pair_rankings,
        "diagnostic_single_feature_trees": diagnostic_trees,
        "top_5_global": [r["feature"] for r in global_rows[:5]],
        "top_5_fsk_tone": [r["feature"] for r in pair_rankings["fsk_vs_tone"][:5]],
        "top_5_ook_benign": [r["feature"] for r in pair_rankings["ook_vs_benign"][:5]],
        "top_5_ook_tone": [r["feature"] for r in pair_rankings["ook_vs_tone"][:5]],
    }


def pairwise_overlap_analysis(data: AuditData) -> dict[str, Any]:
    X = data.dev_features()
    y = data.dev_labels()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pairs = [
        ("fsk", "tone"),
        ("ook", "benign"),
        ("ook", "tone"),
        ("tone", "benign"),
    ]
    results: dict[str, Any] = {}

    for a, b in pairs:
        ia, ib = CLASS_ID[a], CLASS_ID[b]
        mask = (y == ia) | (y == ib)
        Xp = Xs[mask]
        yp = y[mask]
        na = int((yp == ia).sum())
        nb = int((yp == ib).sum())

        ca = Xp[yp == ia].mean(axis=0)
        cb = Xp[yp == ib].mean(axis=0)
        centroid_distance = float(np.linalg.norm(ca - cb))

        cov = np.cov(Xp, rowvar=False) + np.eye(Xp.shape[1]) * 1e-6
        try:
            inv = np.linalg.inv(cov)
            diff = ca - cb
            mahalanobis = float(np.sqrt(diff @ inv @ diff))
        except np.linalg.LinAlgError:
            mahalanobis = float("nan")

        nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
        nn.fit(Xp)
        _, idx = nn.kneighbors(Xp)
        purity_a, purity_b = [], []
        for i, neighbors in enumerate(idx[:, 1:]):
            labels = yp[neighbors]
            if yp[i] == ia:
                purity_a.append(float((labels == ia).mean()))
            else:
                purity_b.append(float((labels == ib).mean()))
        mean_purity = float(np.mean(purity_a + purity_b))

        pca = PCA(n_components=2, random_state=42)
        pca.fit(Xp)
        proj = pca.fit_transform(Xp)
        overlap_2d = _bhattacharyya_overlap(proj[yp == ia, 0], proj[yp == ib, 0])

        results[f"{a}_vs_{b}"] = {
            "n_a": na,
            "n_b": nb,
            "centroid_distance_standardized": centroid_distance,
            "mahalanobis_distance": mahalanobis,
            "knn_local_purity_mean": mean_purity,
            "pca_2d_first_component_overlap": overlap_2d,
            "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "classes_overlapping_in_feature_space": centroid_distance < 2.0 or mean_purity < 0.85,
        }

    return {
        "pairs": results,
        "methodology": [
            "StandardScaler on development features.",
            "PCA used for visualization/diagnostics only.",
            "Mahalanobis on pooled pairwise covariance.",
        ],
    }


def save_feature_separability_csv(rows: list[dict[str, Any]], path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

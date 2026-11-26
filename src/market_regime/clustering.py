import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple, Any


class RegimeClusterer:
    """
    Discovers latent market regimes using Gaussian Mixture Models (GMM)
    over volatility, return and momentum distributions.
    """

    FEATURE_COLS = [
        "log_return",
        "realized_vol_20",
        "parkinson_vol_10",
        "trend_strength",
        "natr_14",
    ]

    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            max_iter=300,
            random_state=random_state,
        )
        self.regime_labels_map: Dict[int, str] = {}
        self.is_fitted = False

    def fit_predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fits GMM on feature matrix and predicts regime probabilities.
        Returns:
            (regimes: np.ndarray, probabilities: np.ndarray, stats: dict)
        """
        X = df[self.FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)

        raw_clusters = self.gmm.fit_predict(X_scaled)
        probabilities = self.gmm.predict_proba(X_scaled)

        # Semantically sort clusters based on average return:
        # 0: Bear/High-Vol (lowest return), 1: Sideways (middle), 2: Bull (highest return)
        cluster_means = []
        for c in range(self.n_regimes):
            mask = raw_clusters == c
            avg_ret = df["log_return"].values[mask].mean() if np.any(mask) else 0.0
            avg_vol = df["realized_vol_20"].values[mask].mean() if np.any(mask) else 0.0
            cluster_means.append((c, avg_ret, avg_vol))

        # Sort by return ascending
        sorted_by_ret = sorted(cluster_means, key=lambda x: x[1])
        reorder_map = {old_c: new_c for new_c, (old_c, _, _) in enumerate(sorted_by_ret)}

        sorted_clusters = np.array([reorder_map[c] for c in raw_clusters])
        sorted_probs = np.zeros_like(probabilities)
        for old_c, new_c in reorder_map.items():
            sorted_probs[:, new_c] = probabilities[:, old_c]

        self.regime_labels_map = {
            0: "Bear (High-Vol)",
            1: "Sideways (Neutral)",
            2: "Bull (Uptrend)",
        }

        # Calculate silhouette score for cluster separation quality
        sil_score = float(silhouette_score(X_scaled, sorted_clusters))

        # Calculate transition matrix
        trans_matrix = np.zeros((self.n_regimes, self.n_regimes))
        for t in range(len(sorted_clusters) - 1):
            curr_r = sorted_clusters[t]
            next_r = sorted_clusters[t + 1]
            trans_matrix[curr_r, next_r] += 1
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_prob_matrix = np.divide(trans_matrix, row_sums, out=np.zeros_like(trans_matrix), where=row_sums != 0)

        stats = {
            "silhouette_score": sil_score,
            "bic": float(self.gmm.bic(X_scaled)),
            "aic": float(self.gmm.aic(X_scaled)),
            "transition_matrix": trans_prob_matrix,
            "regime_distribution": {
                self.regime_labels_map[r]: float((sorted_clusters == r).mean())
                for r in range(self.n_regimes)
            },
        }

        self.is_fitted = True
        return sorted_clusters, sorted_probs, stats

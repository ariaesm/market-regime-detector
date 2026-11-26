import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from typing import Dict, Any, Tuple


class RegimeForecaster:
    """
    Supervised predictive model to forecast market regime transitions at t+1.
    Strictly enforces chronological splits and independent training-set featurization.
    """

    FEATURE_COLS = [
        "log_return",
        "log_return_5d",
        "realized_vol_20",
        "realized_vol_10",
        "parkinson_vol_10",
        "trend_strength",
        "natr_14",
        "volume_zscore",
        "rsi_14",
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
        )
        self.feature_importances: Dict[str, float] = {}

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        regimes: np.ndarray,
        train_ratio: float = 0.75,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        Chronologically splits data, fits scaling and model on train set,
        and evaluates on future unseen test data.
        """
        # Target: regime at t+1 (shifted backward)
        y = np.roll(regimes, -1)
        # Drop the last row since future target is unknown
        X = df[self.FEATURE_COLS].iloc[:-1].values
        y = y[:-1]

        n_samples = len(X)
        split_idx = int(n_samples * train_ratio)

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Fit scaler ONLY on training data (Strict Featurization Ordering)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Random Forest
        self.clf.fit(X_train_scaled, y_train)

        # Predict on unseen test horizon
        y_pred = self.clf.predict(X_test_scaled)
        y_pred_proba = self.clf.predict_proba(X_test_scaled)

        # Extract feature importances
        self.feature_importances = {
            col: float(imp)
            for col, imp in sorted(
                zip(self.FEATURE_COLS, self.clf.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        # Evaluation metrics
        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        conf_mat = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            "test_accuracy": acc,
            "test_macro_f1": macro_f1,
            "confusion_matrix": conf_mat,
            "classification_report": report,
            "feature_importances": self.feature_importances,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        return metrics, y_test, y_pred

"""
ml/model.py — Clean wrapper around the Random Forest model.

# Random Forest because it handles noisy financial data well.
# Deep learning would be overkill here — too much data needed, too slow to train,
# and I can't explain why it made a decision. With RF I can see feature importances.

Keeps the model lifecycle (train, predict, retrain) in one clean place.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional


class CrossoverModel:
    """
    Random Forest classifier for predicting crossover profitability.
    Simple wrapper so the rest of the code doesn't need to know about sklearn.
    """
    # this took me a while to tune — n_estimators=100, max_depth=6 works well
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1  # use all CPU cores — makes training noticeably faster
        )
        self.is_trained = False
        self.feature_names = [
            'ltq_ratio',
            'depth_imbalance',
            'smma_spread',
            'smma_slope',
            'vol_surge_ratio',
            'price_dev'
        ]

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train (or retrain) the model.
        Returns a summary dict with accuracy info.
        """
        from sklearn.metrics import accuracy_score, precision_score

        self.model.fit(X, y)
        self.is_trained = True

        y_pred = self.model.predict(X)
        acc  = accuracy_score(y, y_pred) * 100
        prec = precision_score(y, y_pred, zero_division=0) * 100

        importances = dict(zip(self.feature_names, self.model.feature_importances_))

        return {
            'accuracy':   round(acc, 1),
            'precision':  round(prec, 1),
            'importances': importances
        }

    def predict_win_probability(self, feature_vector: list) -> float:
        """
        Predict the probability that this crossover is a winning trade.
        Returns a 0–100 float. Returns 30.0 if model not trained yet.
        """
        if not self.is_trained:
            return 30.0

        X = np.array([feature_vector])
        probs   = self.model.predict_proba(X)[0]
        classes = list(self.model.classes_)

        if 1 in classes:
            idx  = classes.index(1)
            prob = probs[idx] * 100
        else:
            prob = 30.0

        return round(float(prob), 1)

    def get_feature_importances(self) -> Optional[dict]:
        """Returns feature importance dict, or None if not trained."""
        if not self.is_trained:
            return None
        return dict(zip(self.feature_names, self.model.feature_importances_))

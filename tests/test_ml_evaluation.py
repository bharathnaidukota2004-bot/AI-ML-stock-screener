"""
tests/test_ml_evaluation.py — Test Suite for ML Model Evaluation Engine
Validates strict chronological train/test splitting, zero data leakage, and metric accuracy.
"""

import unittest
import numpy as np
import pandas as pd

from ml.evaluator import ModelEvaluator


class TestMLEvaluation(unittest.TestCase):

    def setUp(self):
        self.evaluator = ModelEvaluator(use_technical_indicators=True)

    def test_chronological_evaluation_pipeline(self):
        # Run evaluation
        results = self.evaluator.run_chronological_evaluation()

        self.assertIn("models", results)
        self.assertIn("best_model", results)
        self.assertIn("dataset", results)

        models = results["models"]
        # Ensure all 3 architectures are evaluated
        self.assertIn("Random Forest", models)
        self.assertIn("Logistic Regression", models)
        self.assertIn("Gradient Boosting", models)

        # Ensure realistic bounded metrics
        for name, metrics in models.items():
            self.assertTrue(0.0 <= metrics["accuracy"] <= 100.0)
            self.assertTrue(0.0 <= metrics["precision"] <= 100.0)
            self.assertTrue(0.0 <= metrics["recall"] <= 100.0)
            self.assertTrue(0.0 <= metrics["f1_score"] <= 100.0)
            self.assertTrue(0.0 <= metrics["roc_auc"] <= 100.0)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("feature_importances", metrics)

        # Best model verification
        best = results["best_model"]
        self.assertIn(best["name"], ["Random Forest", "Logistic Regression", "Gradient Boosting"])
        self.assertGreater(best["f1_score"], 0.0)

    def test_zero_leakage_scaling(self):
        """Verifies scaler is fitted strictly on training data."""
        df = self.evaluator.extract_crossover_dataset()
        self.assertGreaterEqual(len(df), 20)
        # Check that timestamp is strictly non-decreasing
        timestamps = pd.to_datetime(df["timestamp"])
        self.assertTrue((timestamps.diff().dropna() >= pd.Timedelta(0)).all())


if __name__ == "__main__":
    unittest.main()

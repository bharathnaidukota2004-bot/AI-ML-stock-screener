"""
tests/test_mlops_monitoring.py — Test Suite for MLflow Tracking & Model Monitoring
Validates artifact versioning, backup protection, PSI drift detection, and outcome resolution.
"""

import unittest
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from mlops.mlflow_tracker import MLflowExperimentTracker
from mlops.monitoring import ModelMonitor, DriftReport


class TestMLOpsMonitoring(unittest.TestCase):

    def setUp(self):
        self.tracker = MLflowExperimentTracker(experiment_name="Test_Experiment")
        self.monitor = ModelMonitor()

    def test_mlflow_run_logging_and_backup(self):
        # Create a toy model
        clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        X = np.random.randn(20, 6)
        y = np.random.choice([0, 1], size=20)
        clf.fit(X, y)

        run = self.tracker.log_experiment_run(
            model=clf,
            model_name="Random Forest",
            params={'n_estimators': 10, 'max_depth': 3},
            metrics={'accuracy': 85.0, 'precision': 80.0, 'f1_score': 82.5},
            dataset_info={'samples': 20},
            features_list=['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
        )

        self.assertIn("run_id", run)
        self.assertTrue(os.path.exists(run["artifact_path"]))
        self.assertTrue(os.path.exists(run["active_path"]))

        # Verify load_model
        loaded = self.tracker.load_model("Random Forest")
        self.assertIsNotNone(loaded)

    def test_prediction_store_and_delayed_resolution(self):
        pred = self.monitor.log_prediction(
            pred_id="test_pred_001",
            ticker="TATASTEEL",
            model_name="Random Forest",
            features={'ltq': 1.2, 'rsi': 0.55},
            predicted_label=1,
            predicted_probability=75.0
        )
        # Must be PENDING_OUTCOME initially (never marked incorrect without ground truth)
        self.assertEqual(pred["status"], "PENDING_OUTCOME")
        self.assertIsNone(pred["actual_label"])

        # When outcome becomes available:
        success = self.monitor.record_actual_outcome(
            pred_id="test_pred_001",
            actual_label=1,
            actual_return_pct=1.5,
            exit_reason="TARGET_HIT"
        )
        self.assertTrue(success)

        # Check performance metrics
        perf = self.monitor.compute_performance_metrics()
        self.assertGreaterEqual(perf["resolved_count"], 1)

    def test_psi_and_data_drift_detection(self):
        # Baseline data
        np.random.seed(42)
        base = np.random.normal(0, 1, 500)
        # Identical distribution -> PSI should be very close to 0 (< 0.10)
        live_same = np.random.normal(0, 1, 500)
        psi_same = self.monitor.calculate_psi(base, live_same)
        self.assertLess(psi_same, 0.10)

        # Shifted distribution (mean shift by 3 std devs) -> PSI should spike (> 0.25)
        live_shifted = np.random.normal(3.0, 1, 500)
        psi_drift = self.monitor.calculate_psi(base, live_shifted)
        self.assertGreater(psi_drift, 0.25)


if __name__ == "__main__":
    unittest.main()

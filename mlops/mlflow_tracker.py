"""
mlops/mlflow_tracker.py — MLflow Experiment Tracking & Model Versioning Engine
Tracks model parameters, dataset versions, metrics, feature sets, and artifacts.
Implements automatic backup before updating active models to prevent accidental overwrites.
Works seamlessly with MLflow and provides an autonomous local metadata repository.
"""

import os
import json
import shutil
import joblib
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ARCHIVE_DIR = os.path.join(MODELS_DIR, "archive")
ACTIVE_DIR = os.path.join(MODELS_DIR, "active")
BACKUP_DIR = os.path.join(MODELS_DIR, "backup")
EXPERIMENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "experiments_history.json"
)


class MLflowExperimentTracker:
    """
    Manages MLflow experiment tracking, runs, metrics comparison,
    model artifact storage, and versioning.
    """

    def __init__(self, experiment_name: str = "NSE_Stock_Crossover_Predictor"):
        self.experiment_name = experiment_name
        self._ensure_directories()
        
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
        self.use_mlflow = MLFLOW_AVAILABLE
        if self.use_mlflow:
            try:
                # Use SQLite database backend for MLflow tracking
                db_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data",
                    "mlflow.db"
                ).replace(os.sep, "/")
                mlflow.set_tracking_uri(f"sqlite:///{db_path}")
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                print(f"[MLflowTracker] Note: MLflow tracking initialization fallback to local JSON ({e})")
                self.use_mlflow = False

    def _ensure_directories(self):
        for d in [ARCHIVE_DIR, ACTIVE_DIR, BACKUP_DIR, os.path.dirname(EXPERIMENTS_FILE)]:
            os.makedirs(d, exist_ok=True)

    def log_experiment_run(
        self,
        model: Any,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        features_list: List[str]
    ) -> Dict[str, Any]:
        """
        Logs a full machine learning training run:
        1. Backs up previous active model if one exists.
        2. Saves versioned artifact to models/archive/.
        3. Updates active model in models/active/.
        4. Logs to MLflow (if available) and local experiments registry.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_tag = f"v_{timestamp_str}"
        clean_model_name = model_name.replace(" ", "_").lower()
        run_id = f"{clean_model_name}_{version_tag}"

        # 1. Safe Backup of Active Model before overwriting
        active_path = os.path.join(ACTIVE_DIR, f"{clean_model_name}.joblib")
        if os.path.exists(active_path):
            backup_path = os.path.join(BACKUP_DIR, f"{clean_model_name}_backup_{timestamp_str}.joblib")
            shutil.copy2(active_path, backup_path)

        # 2. Save new versioned artifact in Archive
        archive_path = os.path.join(ARCHIVE_DIR, f"{clean_model_name}_{version_tag}.joblib")
        joblib.dump(model, archive_path)

        # 3. Update active model
        joblib.dump(model, active_path)

        # 4. MLflow Logging (if available)
        mlflow_run_id = None
        if self.use_mlflow:
            try:
                with mlflow.start_run(run_name=run_id):
                    mlflow.log_params(params)
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("dataset_version", dataset_info.get("date_range", "latest"))
                    mlflow.set_tag("feature_count", len(features_list))
                    try:
                        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
                    except Exception:
                        mlflow.sklearn.log_model(model, name="model")
                    mlflow_run_id = mlflow.active_run().info.run_id
            except Exception as e:
                print(f"[MLflowTracker] MLflow log_run exception: {e}")

        # 5. Local Registry JSON Logging
        run_record = {
            'run_id': run_id,
            'mlflow_run_id': mlflow_run_id,
            'model_name': model_name,
            'version': version_tag,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'parameters': params,
            'metrics': {k: round(v, 2) for k, v in metrics.items()},
            'dataset_info': dataset_info,
            'features': features_list,
            'artifact_path': archive_path,
            'active_path': active_path
        }

        self._append_experiment_record(run_record)
        return run_record

    def _append_experiment_record(self, record: Dict[str, Any]):
        records = self.get_all_experiments()
        records.append(record)
        with open(EXPERIMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Reads all experiment records from the local repository."""
        if not os.path.exists(EXPERIMENTS_FILE):
            return []
        try:
            with open(EXPERIMENTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def get_experiments_dataframe(self) -> pd.DataFrame:
        """Returns comparison table of all historical runs."""
        experiments = self.get_all_experiments()
        if not experiments:
            return pd.DataFrame(columns=[
                'Run ID', 'Model', 'Timestamp', 'Accuracy', 'Precision', 'F1-Score', 'ROC-AUC'
            ])

        rows = []
        for exp in experiments:
            metrics = exp.get('metrics', {})
            rows.append({
                'Run ID': exp.get('run_id'),
                'Model': exp.get('model_name'),
                'Timestamp': exp.get('timestamp'),
                'Accuracy': metrics.get('accuracy', 0.0),
                'Precision': metrics.get('precision', 0.0),
                'Recall': metrics.get('recall', 0.0),
                'F1-Score': metrics.get('f1_score', 0.0),
                'ROC-AUC': metrics.get('roc_auc', 0.0),
                'Features': len(exp.get('features', [])),
                'Artifact': os.path.basename(exp.get('artifact_path', ''))
            })
        df = pd.DataFrame(rows)
        return df.sort_values(by='F1-Score', ascending=False).reset_index(drop=True)

    def get_best_experiment(self, metric: str = 'f1_score') -> Optional[Dict[str, Any]]:
        """Identifies best historical validated run based on designated metric."""
        exps = self.get_all_experiments()
        if not exps:
            return None
        return max(exps, key=lambda x: x.get('metrics', {}).get(metric, 0.0))

    def load_model(self, run_id_or_name: str) -> Optional[Any]:
        """Loads a model artifact by run_id or clean model name."""
        clean = run_id_or_name.replace(" ", "_").lower()
        active_target = os.path.join(ACTIVE_DIR, f"{clean}.joblib")
        if os.path.exists(active_target):
            return joblib.load(active_target)

        # Check archive
        for f in os.listdir(ARCHIVE_DIR):
            if run_id_or_name in f:
                return joblib.load(os.path.join(ARCHIVE_DIR, f))

        return None

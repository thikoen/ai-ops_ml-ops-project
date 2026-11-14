import json
from types import SimpleNamespace
from unittest.mock import Mock

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rail_delay import pipelines
from rail_delay.settings import FEATURES


def test_inference_uses_registered_version_and_current_request(tmp_path, monkeypatch):
    # Fit a small real model; mock Hopsworks and object storage.
    x = pd.DataFrame(
        {
            "delay_mean_7d": [1.0] * 4,
            "departure_delay_minutes": [0.0, 1.0, 10.0, 20.0],
            "departure_hour": [10.5] * 4,
            "is_weekend": [0.0] * 4,
        }
    )
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, [0, 0, 1, 1])
    joblib.dump(model, tmp_path / "model.joblib")
    manifest = {
        "run_id": "test-run",
        "features": FEATURES,
        "inference_not_before": "2026-08-20T00:00:00+00:00",
        "model_name": "ice_model",
        "model_version": 7,
        "feature_view": "ice_view",
        "feature_view_version": 2,
        "training_dataset_version": 3,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(manifest), encoding="utf-8")
    store, project = Mock(), Mock()
    store.get_json.return_value = manifest
    registry = project.get_model_registry.return_value
    registry.get_model.return_value.download.return_value = str(tmp_path)
    view = project.get_feature_store.return_value.get_feature_view.return_value
    view.get_batch_data.return_value = pd.DataFrame(
        {
            "trip_id": ["trip-1"],
            "event_time": [pd.Timestamp("2026-08-21 08:30Z")],
            "delay_mean_7d": [1.0],
            "departure_delay_minutes": [99.0],
            "departure_hour": [99.0],
            "is_weekend": [1.0],
        }
    )
    monkeypatch.setattr(pipelines, "store_for", lambda _: store)
    monkeypatch.setattr(pipelines, "connect", lambda _: project)
    monkeypatch.setattr(pipelines, "monthly_files", dict)
    monkeypatch.setattr(pipelines, "load_rail_month", lambda *args: None)
    monkeypatch.setattr(
        pipelines,
        "departure_requests",
        lambda _: pd.DataFrame(
            {
                "trip_id": ["trip-1"],
                "event_time": [pd.Timestamp("2026-08-21 08:30Z")],
                "departure_delay_minutes": [0.0],
            }
        ),
    )
    pipelines.inference_pipeline(SimpleNamespace(bucket="test-bucket"), "2026-08-21")
    registry.get_model.assert_called_once_with("ice_model", version=7)
    view.init_batch_scoring.assert_called_once_with(training_dataset_version=3)
    key, result = store.put_frame.call_args.args
    assert key.startswith("predictions/2026-08-21/")
    assert result.iloc[0]["departure_delay_minutes"] == 0.0
    assert result.iloc[0]["departure_hour"] == 10.5
    assert result.iloc[0]["is_weekend"] == 0.0
    assert result.iloc[0]["prediction_delayed_5min"] == 0
    assert result.iloc[0]["model_version"] == 7

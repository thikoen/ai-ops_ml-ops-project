"""Three explicit FTI steps. MinIO stores files; Hopsworks stores features and models."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd

from .features import (
    add_calendar_features,
    build_features,
    departure_requests,
    midnight,
    route_trips,
)
from .hops import connect, feature_group, feature_view
from .modeling import predict, train_and_evaluate
from .object_store import ObjectStore
from .settings import FEATURES, MODEL, TARGET, VIEW
from .sources import load_rail_month, monthly_files

LOG = logging.getLogger(__name__)


def store_for(settings):
    store = ObjectStore(settings)
    store.ensure_bucket()
    return store


def feature_pipeline(settings, start, end):
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError("--end must be at or after --start")
    store = store_for(settings)
    # Eight preceding days provide a full seven-day window with a publication lag.
    warmup = (pd.Timestamp(start) - pd.Timedelta(days=8)).date().isoformat()
    months = pd.period_range(warmup, pd.Timestamp(end) + pd.Timedelta(days=1), freq="M")
    files = monthly_files()
    raw = pd.concat(
        [load_rail_month(str(month), store, files) for month in months], ignore_index=True
    )
    trips = route_trips(raw)
    examples = build_features(trips, start, end)
    project = connect(settings)
    group = feature_group(project, settings.version)
    group.insert(examples, write_options={"wait_for_job": True})
    report = {
        "start": start,
        "end": end,
        "source_start": warmup,
        "source_rows": len(raw),
        "valid_trips_including_warmup": len(trips),
        "feature_rows": len(examples),
        "positive_labels": int(examples[TARGET].sum()),
        "features": FEATURES,
        "feature_group_version": settings.version,
        "mode": "historical replay of archived DB observations",
    }
    store.put_json("reports/features/latest.json", report)
    LOG.info("Feature pipeline complete: %s", report)


def training_pipeline(settings, cutoff):
    """Train as of cutoff (local midnight), leaving later days exclusively for replay."""
    store = store_for(settings)
    project = connect(settings)
    view = feature_view(project, settings.version)
    # At cutoff, labels for the day immediately before cutoff are not assumed published.
    train_end = midnight(pd.Timestamp(cutoff) - pd.Timedelta(days=1))
    run_id = uuid4().hex
    x, y = view.training_data(
        end_time=train_end.to_pydatetime(),
        description=f"ICE chronological training {run_id}; as of {cutoff}",
        primary_key=True,
        event_time=True,
        training_helper_columns=True,
        statistics_config=False,
    )
    dataset_version = view.get_last_accessed_training_dataset()
    if dataset_version is None:
        raise RuntimeError("Hopsworks did not return a training dataset version")
    frame = x.copy()
    if len(frame) != len(y) or not frame.index.equals(y.index):
        raise ValueError("Hopsworks returned misaligned features and labels")
    frame[TARGET] = y[TARGET].astype(int)
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["label_available_at"] = pd.to_datetime(frame["label_available_at"], utc=True)
    frame = frame[
        (frame["event_time"] < train_end) & (frame["label_available_at"] <= midnight(cutoff))
    ]
    model, report = train_and_evaluate(frame)
    artifact_dir = settings.work / "models" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_dir / "model.joblib")
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "inference_not_before": midnight(cutoff).isoformat(),
        "training_end_exclusive": train_end.isoformat(),
        "feature_view": VIEW,
        "feature_view_version": settings.version,
        "training_dataset_version": int(dataset_version),
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    registry = project.get_model_registry()
    registered = registry.python.create_model(
        name=MODEL,
        description="Logistic regression: ICE Hamburg-Berlin arrival delay >= 5 minutes; historical replay",
        metrics={
            key: value for key, value in report["validation"].items() if isinstance(value, float)
        },
        feature_view=view,
        training_dataset_version=int(dataset_version),
        input_example=frame[FEATURES].head(1),
    )
    registered.save(str(artifact_dir))
    manifest = {**metadata, "model_name": MODEL, "model_version": int(registered.version)}
    store.put_json(f"reports/training/{run_id}.json", {**manifest, **report})
    # Promote only after the model and report have both been saved successfully.
    store.put_json("models/latest.json", manifest)
    LOG.info(
        "Training complete: model version %s, validation %s",
        registered.version,
        report["validation"],
    )


def validate_replay_day(day, manifest):
    start = midnight(day)
    if start < pd.Timestamp(manifest["inference_not_before"]):
        raise ValueError("Replay date overlaps the training period; choose a date at/after cutoff")
    return start, midnight(pd.Timestamp(day) + pd.Timedelta(days=1))


def inference_pipeline(settings, day):
    store = store_for(settings)
    manifest = store.get_json("models/latest.json")
    start, end = validate_replay_day(day, manifest)
    project = connect(settings)
    registered = project.get_model_registry().get_model(
        manifest["model_name"],
        version=manifest["model_version"],
    )
    artifact_dir = Path(registered.download())
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["run_id"] != manifest["run_id"] or metadata["features"] != FEATURES:
        raise ValueError("Model metadata does not match the selected run or feature schema")
    model = joblib.load(artifact_dir / "model.joblib")
    view = project.get_feature_store().get_feature_view(
        metadata["feature_view"],
        version=metadata["feature_view_version"],
    )
    view.init_batch_scoring(training_dataset_version=metadata["training_dataset_version"])
    batch = view.get_batch_data(
        start_time=start.to_pydatetime(),
        end_time=end.to_pydatetime(),
        primary_key=True,
        event_time=True,
    )
    if TARGET in batch:
        raise ValueError("Feature View unexpectedly returned labels for inference")
    batch["event_time"] = pd.to_datetime(batch["event_time"], utc=True)
    batch = batch[batch["event_time"].between(start, end, inclusive="left")]
    # The aggregate comes from the Feature View. Current inputs are independently
    # reconstructed from ORIGIN records in the held-out period (no arrival labels).
    raw = load_rail_month(day[:7], store, monthly_files())
    requests = departure_requests(raw)[["trip_id", "event_time", "departure_delay_minutes"]]
    batch = batch.drop(columns=["departure_delay_minutes", "departure_hour", "is_weekend"]).merge(
        requests,
        on=["trip_id", "event_time"],
        how="left",
        validate="one_to_one",
    )
    if batch["departure_delay_minutes"].isna().any():
        raise ValueError("Missing simulated current departure observation")
    current_inputs = add_calendar_features(batch)
    results = predict(model, current_inputs)
    results["model_version"] = manifest["model_version"]
    results["mode"] = "historical_replay"
    key = f"predictions/{day}/{uuid4().hex}.parquet"
    store.put_frame(key, results)
    LOG.info("Inference complete: %d predictions at s3://%s/%s", len(results), settings.bucket, key)
    print(results.head(10).to_string(index=False))

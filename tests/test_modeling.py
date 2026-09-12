import numpy as np
import pandas as pd
import pytest

from rail_delay.modeling import predict, train_and_evaluate
from rail_delay.pipelines import validate_replay_day
from rail_delay.settings import FEATURES, TARGET


def sample():
    size = 100
    return pd.DataFrame(
        {
            "trip_id": [str(i) for i in range(size)],
            "event_time": pd.date_range("2026-08-01 08:00Z", periods=size, freq="3h"),
            "label_available_at": pd.date_range("2026-08-03 08:00Z", periods=size, freq="3h"),
            "delay_mean_7d": np.tile([0.0, 1.0], size // 2),
            "departure_delay_minutes": np.tile([0.0, 10.0], size // 2),
            "departure_hour": np.tile([8.0, 18.0], size // 2),
            "is_weekend": np.tile([0.0, 1.0], size // 2),
            TARGET: np.tile([0, 1], size // 2),
        }
    )


def test_training_split_and_inference_use_only_features():
    frame = sample()
    model, report = train_and_evaluate(frame)
    days = frame.event_time.dt.tz_convert("Europe/Berlin").dt.date.astype(str)
    validation_start = frame.loc[days >= report["validation_start_local_date"], "event_time"].min()
    assert report["train_rows"] == int(
        (
            (days < report["validation_start_local_date"])
            & (frame.label_available_at <= validation_start)
        ).sum()
    )
    assert list(model.feature_names_in_) == FEATURES
    result = predict(model, frame.drop(columns=TARGET))
    assert result.probability_delayed_5min.between(0, 1).all()
    assert result.prediction_delayed_5min.nunique() == 2
    with pytest.raises(ValueError, match="target labels"):
        predict(model, frame)


def test_single_class_prefix_fails_with_actionable_message():
    frame = sample()
    frame[TARGET] = 0
    with pytest.raises(ValueError, match="one class"):
        train_and_evaluate(frame)


def test_replay_rejects_training_dates():
    manifest = {"inference_not_before": "2026-08-15T22:00:00+00:00"}
    with pytest.raises(ValueError, match="overlaps"):
        validate_replay_day("2026-08-15", manifest)
    start, end = validate_replay_day("2026-08-16", manifest)
    assert end > start

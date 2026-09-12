"""Small classifier with a chronological validation split."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .settings import FEATURES, TARGET


def score(y, predicted):
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix_0_1": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(frame):
    frame = frame.sort_values("event_time").reset_index(drop=True)
    days = frame["event_time"].dt.tz_convert("Europe/Berlin").dt.date
    unique_days = sorted(days.unique())
    if len(unique_days) < 5:
        raise ValueError("Need at least five training days for chronological validation")
    split_day = unique_days[min(len(unique_days) - 1, int(len(unique_days) * 0.8))]
    train, val = frame[days < split_day], frame[days >= split_day]
    # Even within validation, a label must have been available when validation starts.
    train = train[train["label_available_at"] <= val["event_time"].min()]
    if train[TARGET].nunique() != 2:
        raise ValueError("Training prefix has only one class; load a longer date range")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42)
    )
    model.fit(train[FEATURES], train[TARGET])
    baseline = DummyClassifier(strategy="most_frequent").fit(train[FEATURES], train[TARGET])
    report = {
        "validation": score(val[TARGET], model.predict(val[FEATURES])),
        "majority_baseline": score(val[TARGET], baseline.predict(val[FEATURES])),
        "always_delayed_baseline": score(val[TARGET], [1] * len(val)),
        "departure_delay_baseline": score(
            val[TARGET], (val["departure_delay_minutes"] >= 5).astype(int)
        ),
        "validation_start_local_date": str(split_day),
        "train_rows": len(train),
        "validation_rows": len(val),
        "train_positive_labels": int(train[TARGET].sum()),
        "validation_positive_labels": int(val[TARGET].sum()),
        "evaluation_note": "Metrics describe temporal holdout; saved model refitted on all pre-cutoff rows",
    }
    model.fit(frame[FEATURES], frame[TARGET])
    return model, report


def predict(model, frame):
    if frame.empty:
        raise ValueError("No inference rows in requested interval")
    if TARGET in frame:
        raise ValueError("Inference input must not contain target labels")
    values = frame[FEATURES].astype(float)
    if values.isna().any().any():
        raise ValueError("Inference features contain missing values")
    result = frame[["trip_id", "event_time", *FEATURES]].copy()
    result["prediction_delayed_5min"] = model.predict(values).astype(int)
    positive_index = list(model.classes_).index(1)
    result["probability_delayed_5min"] = model.predict_proba(values)[:, positive_index]
    return result

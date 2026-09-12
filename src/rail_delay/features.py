"""Shared feature logic. Inference time is the observed departure in Hamburg."""

import hashlib
import logging

import pandas as pd

from .settings import FEATURES, TARGET
from .sources import DESTINATION, ORIGIN

LOG = logging.getLogger(__name__)
ZONE = "Europe/Berlin"


def midnight(day):
    return pd.Timestamp(day).tz_localize(ZONE).tz_convert("UTC")


def parse_local(series):
    parsed = pd.to_datetime(series, format="mixed", dayfirst=True, errors="coerce")
    return parsed.dt.tz_localize(ZONE, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def unique_stops(raw):
    frame = raw[raw["train_type"].eq("ICE")].drop_duplicates().copy()
    # The ride hash alone repeats across service days. The stop id additionally
    # contains the initial departure date/time; remove only its final stop number.
    frame = frame[frame["id"].notna()]
    frame["trip_id"] = (
        frame["id"]
        .str.rsplit("-", n=1)
        .str[0]
        .map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:32])
    )
    return frame[~frame.duplicated(["trip_id", "station_name"], keep=False)]


def departure_requests(raw):
    """Simulated live requests from ORIGIN only: no destination outcome is read."""
    origin = unique_stops(raw)
    origin = origin[
        origin["station_name"].eq(ORIGIN) & origin["departure_is_canceled"].eq(False)
    ].copy()
    planned = parse_local(origin["departure_planned_time"])
    observed = parse_local(origin["departure_change_time"])
    result = pd.DataFrame(
        {
            "trip_id": origin["trip_id"],
            "event_time": observed,
            "planned_departure": planned,
            "departure_delay_minutes": (observed - planned).dt.total_seconds() / 60,
        }
    )
    return result.dropna().reset_index(drop=True)


def route_trips(raw):
    origin = departure_requests(raw)
    dest = unique_stops(raw)
    dest = dest[dest["station_name"].eq(DESTINATION) & dest["arrival_is_canceled"].eq(False)].copy()
    dest["planned_arrival"] = parse_local(dest["arrival_planned_time"])
    dest["arrival_time"] = parse_local(dest["arrival_change_time"])
    joined = origin.merge(
        dest[["trip_id", "planned_arrival", "arrival_time"]], on="trip_id", validate="one_to_one"
    )
    duration = (joined["planned_arrival"] - joined["planned_departure"]).dt.total_seconds() / 60
    # Exclude the reverse direction and trips that had already arrived at prediction time.
    result = joined[
        duration.between(1, 360) & (joined["arrival_time"] > joined["event_time"])
    ].copy()
    result["delay_minutes"] = (
        result["arrival_time"] - result["planned_arrival"]
    ).dt.total_seconds() / 60
    result[TARGET] = (result["delay_minutes"] >= 5).astype("int64")
    # Archive publication times are not live measurements. Replay assumes a label
    # is usable from the second local calendar day after arrival.
    local_arrival = result["arrival_time"].dt.tz_convert(ZONE).dt.tz_localize(None)
    result["label_available_at"] = (
        (local_arrival.dt.normalize() + pd.Timedelta(days=2))
        .dt.tz_localize(ZONE)
        .dt.tz_convert("UTC")
    )
    LOG.info("Route selection: %d matches, %d valid forward journeys", len(joined), len(result))
    if result.empty:
        raise ValueError("No valid ICE trips Hamburg -> Berlin with observed departure and arrival")
    return result.sort_values("event_time").reset_index(drop=True)


def add_calendar_features(frame):
    """Known at request time, calculated identically for training and inference."""
    result = frame.copy()
    local = pd.to_datetime(result["event_time"], utc=True).dt.tz_convert(ZONE)
    result["departure_hour"] = local.dt.hour + local.dt.minute / 60
    result["is_weekend"] = (local.dt.dayofweek >= 5).astype(float)
    return result


def build_features(trips, start, end):
    start_time = midnight(start)
    end_time = midnight(pd.Timestamp(end) + pd.Timedelta(days=1))
    examples = trips[trips["event_time"].between(start_time, end_time, inclusive="left")].copy()
    means = []
    for timestamp in examples["event_time"]:
        day = timestamp.tz_convert(ZONE).tz_localize(None).normalize()
        history = trips[
            trips["event_time"].between(
                midnight(day - pd.Timedelta(days=8)),
                midnight(day - pd.Timedelta(days=1)),
                inclusive="left",
            )
            & (trips["label_available_at"] <= timestamp)
        ]
        means.append(history["delay_minutes"].mean())
    examples["delay_mean_7d"] = means
    if examples.empty or examples["delay_mean_7d"].isna().any():
        raise ValueError("Missing past trips for aggregate; include eight preceding source days")
    result = add_calendar_features(examples)
    result[FEATURES] = result[FEATURES].astype("float64")
    return result[["trip_id", "event_time", "label_available_at", *FEATURES, TARGET]]

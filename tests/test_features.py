import pandas as pd

from rail_delay.features import (
    add_calendar_features,
    build_features,
    departure_requests,
    midnight,
    parse_local,
    route_trips,
)
from rail_delay.settings import TARGET
from rail_delay.sources import DESTINATION, ORIGIN, filter_month


def row(trip="a", station=ORIGIN, arrival="10.08.2025 10:00", actual=None, **changes):
    value = {
        "station_name": station,
        "train_type": "ICE",
        "id": f"{trip}-2508101000-{1 if station == ORIGIN else 2}",
        "arrival_is_canceled": False,
        "departure_is_canceled": False,
        "arrival_planned_time": arrival,
        "arrival_change_time": actual or arrival,
        "departure_planned_time": "10.08.2025 10:00",
        "departure_change_time": "10.08.2025 10:05",
    }
    value.update(changes)
    return value


def test_route_direction_label_and_quality():
    rows = [row(), row(station=DESTINATION, arrival="10.08.2025 12:00", actual="10.08.2025 12:10")]
    rows += [
        row(trip="reverse"),
        row(trip="reverse", station=DESTINATION, arrival="10.08.2025 09:30"),
    ]
    rows += [
        row(trip="missing"),
        row(
            trip="missing",
            station=DESTINATION,
            arrival="10.08.2025 12:00",
            arrival_change_time=None,
        ),
    ]
    rows += [
        row(trip="cancel"),
        row(
            trip="cancel", station=DESTINATION, arrival="10.08.2025 12:00", arrival_is_canceled=True
        ),
    ]
    rows.append(rows[0].copy())
    trips = route_trips(pd.DataFrame(rows))
    assert len(trips) == 1
    assert trips.iloc[0][TARGET] == 1
    assert trips.iloc[0]["delay_minutes"] == 10
    assert trips.iloc[0]["departure_delay_minutes"] == 5
    assert trips.iloc[0]["event_time"] == pd.Timestamp("2025-08-10 08:05Z")
    assert trips.iloc[0]["label_available_at"] == midnight("2025-08-12")


def test_month_filter_keeps_only_ice_at_both_stations(tmp_path):
    frame = pd.DataFrame(
        [row(), row(station=DESTINATION), row(train_type="RE"), row(station="Kiel Hbf")]
    )
    path = tmp_path / "month.parquet"
    frame.to_parquet(path, index=False)
    selected = filter_month(path)
    assert selected.station_name.tolist() == [ORIGIN, DESTINATION]


def test_departure_request_needs_no_destination_outcome():
    raw = pd.DataFrame([row()])
    requests = departure_requests(raw)
    assert requests.departure_delay_minutes.tolist() == [5.0]
    assert TARGET not in requests
    assert "arrival_time" not in requests


def test_ride_key_includes_service_date():
    raw = pd.DataFrame([row(id="123-2508101000-1"), row(id="123-2508111000-1")])
    assert departure_requests(raw).trip_id.nunique() == 2


def test_dst_ambiguous_time_is_not_guessed():
    values = parse_local(pd.Series(["25.10.2026 02:30", "29.03.2026 02:30", "10.08.2026 10:00"]))
    assert values.isna().tolist() == [True, True, False]


def test_aggregate_ignores_yesterday_today_and_future_labels():
    times = pd.date_range("2026-08-01 08:00", periods=12, freq="D", tz="UTC")
    trips = pd.DataFrame(
        {
            "trip_id": [str(i) for i in range(12)],
            "event_time": times,
            "label_available_at": [midnight(f"2026-08-{i + 3:02}") for i in range(12)],
            "delay_minutes": list(range(12)),
            "departure_delay_minutes": [0.0] * 12,
            TARGET: [0] * 12,
        }
    )
    before = build_features(trips, "2026-08-09", "2026-08-09")
    assert before.iloc[0]["delay_mean_7d"] == 3.0
    trips.loc[7:, "delay_minutes"] = 9999
    after = build_features(trips, "2026-08-09", "2026-08-09")
    pd.testing.assert_frame_equal(before, after)


def test_calendar_features_use_local_time_and_replace_stale_values():
    frame = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2026-08-07 23:30Z", "2026-08-10 08:45Z"]),
            "departure_hour": [99.0, 99.0],
            "is_weekend": [0.0, 1.0],
        }
    )
    result = add_calendar_features(frame)
    assert result.departure_hour.tolist() == [1.5, 10.75]
    assert result.is_weekend.tolist() == [1.0, 0.0]

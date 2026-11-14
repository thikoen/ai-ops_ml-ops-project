# ICE Hamburg–Berlin delay prediction

CAS AI Operations project: predict whether an ICE from Hamburg Hbf to Berlin
Hauptbahnhof arrives at least five minutes late. Requests are simulated at the
observed departure time using archived data.

The original S3 Fehraltorf–Zürich route had too few delayed journeys (only two in
validation, F1 = 0). Hamburg–Berlin provides more examples of both classes.
Temperature was also tested, but produced identical predictions with and without
it (F1 = 0.470), so weather was removed. This does not mean weather never matters.

## Data and features

Source: [Deutsche Bahn Data by Piet Brömmel](https://github.com/piebro/deutsche-bahn-data),
available on [Hugging Face](https://huggingface.co/datasets/piebro/deutsche-bahn-data).
Attribution: Deutsche Bahn and Piet Brömmel, CC BY 4.0. The project filters and
transforms the data; the dataset revision and download hashes are pinned.

| Feature | Meaning |
|---|---|
| `delay_mean_7d` | Aggregated arrival delay over days D−8 to D−2 |
| `departure_delay_minutes` | Current departure delay, simulated from the archive |
| `departure_hour` | Local departure hour, including minutes |
| `is_weekend` | Saturday/Sunday indicator |

The label is arrival delay >= 5 minutes. `trip_id` is the primary key and
`event_time` the observed departure in UTC. Canceled, incomplete and ambiguous
journeys are excluded. Labels are assumed available two local days after arrival.

## Pipelines

Implementation: `src/rail_delay/pipelines.py`, with separate modules for data
retrieval, features, modeling, Hopsworks and object storage.

1. **Features:** download data, match journeys, calculate features and labels,
   then insert into a Hopsworks Feature Group. MinIO stores source extracts and metadata.
2. **Training:** create a Feature View and versioned training dataset, train and
   validate chronologically, then save the model in the Hopsworks Model Registry.
3. **Inference:** retrieve historical features through the Feature View, reconstruct
   current departure inputs, load the registered model and write predictions to MinIO.
   Arrival labels are not used for prediction.

The model is `StandardScaler` + logistic regression with balanced class weights
and a 0.5 threshold. Validation uses the last roughly 20% of eligible training days;
unavailable labels are excluded. The final model is refitted before the September
cutoff, leaving a publication buffer. October is held out for testing.

## Run

Requires Docker Compose v2 with Linux containers, internet access and a Hopsworks
project. The container uses Python 3.12 and dependencies from `uv.lock`.

Copy `.env.example` to `.env` unless it already exists. The host and project name
are prefilled for this project; change them for your own Hopsworks project. Set
`HOPSWORKS_API_KEY` with feature/model access and your own MinIO credentials
(password at least eight characters). `.env` is excluded from Git.

```sh
docker compose build minio pipeline
docker compose up -d minio
docker compose run --rm pipeline features --start 2025-06-01 --end 2025-10-30
docker compose run --rm pipeline train --cutoff 2025-09-01
docker compose run --rm pipeline infer --day 2025-10-30
```

The first run downloads about 624 MB, including history for the rolling window.
MinIO is built from official source because the previously pinned images were
unavailable; the first build takes several minutes. Later runs reuse caches.
Run features before inference so the requested day exists in Hopsworks.

Inspect features, datasets and models in Hopsworks. Raw data, reports and prediction
Parquet files are in the MinIO bucket `rail-delay`. Open the MinIO console on
port 9001 (use private port forwarding in Codespaces). `docker compose down` stops
the services and keeps the stored data.

## Results and limitations

Validation F1: **0.498**. On 516 held-out October journeys:

| Predictor | Precision | Recall | F1 |
|---|---:|---:|---:|
| Logistic regression | 0.786 | 0.405 | 0.534 |
| Departure delay >= 5 minutes | 0.837 | 0.321 | 0.465 |
| Always delayed | 0.651 | 1.000 | 0.789 |

The model finds 136 of 336 delays with 37 false alarms. It misses many delays and
is not clearly better than the simple departure rule. Always predicting delay
has higher F1 but warns incorrectly for all 180 on-time journeys.

This is a historical replay, not a live travel service. Archived timestamps may
have been revised later; exact historical availability cannot be guaranteed.
Missing data and excluded cancellations can also bias results.

All three pipelines passed in Codespaces with real Hopsworks and MinIO on
2026-11-14: **3,057 feature rows, model version 1 and 18 predictions** for October 30.
Independent readback passed, as did 15 tests and Ruff. See [verification](docs/verification.json).

To run the checks locally with Python 3.12 and uv:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
```

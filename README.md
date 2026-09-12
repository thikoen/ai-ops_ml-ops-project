# ICE Hamburg–Berlin: a minimal MLOps project

CAS AI Operations, **Feature–Training–Inference (FTI)**: Will an ICE train from
Hamburg Hbf to Berlin Hauptbahnhof arrive at least five minutes late? Prediction
requests are simulated at the observed departure time in Hamburg.

Three Python pipelines use **Hopsworks** as the feature store and model registry.
**MinIO** stores raw data, source metadata, reports and predictions. Docker Compose
contains only MinIO and one reusable Python pipeline container.

## Design decisions, including the rejected weather hypothesis

**Original route:** S3 Fehraltorf → Zürich HB. The initial validation contained only
two delayed journeys out of 75; the training partition also had only two positive
examples. The model always predicted “on time”. Accuracy of 97.3% with F1 = 0 was
not a successful result. We therefore chose a different route and a longer history
rather than merely documenting the weak results.

**Switch to Germany:** A preliminary check of August 2025 found 585 matching ICE
journeys from Hamburg Hbf to Berlin Hauptbahnhof with complete timestamps, including
305 arrivals delayed by at least five minutes. Both classes were much better
represented. The final processing also handles month boundaries and the local
departure date, so its monthly counts differ slightly from this preliminary check.
The route was selected using data volume and class distribution, not the later
October test results.

**Weather removed:** We tested whether temperature helped prediction. On the same
chronological August validation, the results were:

| Features | F1 |
|---|---:|
| Historical mean delay + current departure delay + temperature | 0.470 |
| Historical mean delay + current departure delay | 0.470 |
| The same two features + time of day + weekend | 0.498 |

Class predictions were identical with and without temperature. Weather retrieval
was therefore removed entirely. The small benefit of the calendar features is an
observation on this validation set, not a statistically established general effect.
**We do not claim that weather never causes delays.** We simply did not demonstrate
additional predictive value in this model and period.

**Model:** Logistic regression is small and easy to explain. A limited comparison
on the earlier three-feature variant gave
`GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42)` an F1 of
0.360, compared with 0.470 for logistic regression. We therefore did not introduce
a more complex model type. The final variant uses four features and no weather service.

The data source, period and features all changed. Improvements over the S3 experiment
cannot be attributed solely to switching countries.

## Architecture and features

```text
Archived DB data (Parquet)
            |
    Feature pipeline ---------- Raw data/source metadata → MinIO
            |
 Hopsworks Feature Group
            |
  Hopsworks Feature View
            |
    Training pipeline --------- Evaluation report → MinIO
            |
 Hopsworks Model Registry
            |
Feature View + simulated current departure observation
            |
    Inference pipeline -------- Predictions as Parquet → MinIO
```

| Feature | Type | Meaning |
|---|---|---|
| `delay_mean_7d` | Aggregated | Mean arrival delay on the route over seven completed days. For prediction day D: D−8 through D−2. |
| `departure_delay_minutes` | Current, simulated in replay | Archived departure time in Hamburg minus scheduled departure time. |
| `departure_hour` | Known at request time | Local hour including fractional minutes, e.g. 10.5 for 10:30. |
| `is_weekend` | Known at request time | 1 on Saturday/Sunday, otherwise 0. |

Departure delay satisfies the RT-feature requirement. Prediction takes place
**at observed departure**, not before scheduled departure. Before departure, that
information would not yet be known, and using it would leak future information.

## Running in Codespaces or with local Docker

Requirements: a running Linux container engine, Docker Compose v2, internet access
and a Hopsworks project. `docker info` must show a server as well as a client.
Python **3.12** and the packages in `uv.lock` are installed inside the container.

On GitHub, select **Code → Codespaces → Create codespace on main**. Run the following
in the repository terminal; the same commands apply on local Linux:

```sh
# Do not overwrite an existing .env.
cp .env.example .env
```

Replace the example values in `.env` with your own:

- `HOPSWORKS_HOST`: the host from your project URL, without https:// or a path.
- `HOPSWORKS_PROJECT`: your project name.
- `HOPSWORKS_API_KEY`: your key with read/write access to features and models.
- `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`: your own local credentials;
  the password must contain at least eight characters.

Local credentials are **not** transferred to Codespaces. `.env` and SDK certificates
are excluded from Git and the image. In PowerShell, use `Copy-Item .env.example .env`.

```sh
docker compose up -d minio
docker compose build pipeline

# 1. Write features and labels to Hopsworks.
docker compose run --rm pipeline features --start 2025-06-01 --end 2025-10-30

# 2. Train using only history before the September cutoff.
docker compose run --rm pipeline train --cutoff 2025-09-01

# 3. Simulate current requests for a held-out day.
docker compose run --rm pipeline infer --day 2025-10-30
```

The feature pipeline automatically loads eight preceding days for warm-up and one
day after `--end` for arrivals crossing midnight. The source supplies monthly files,
so this example reads May–October: approximately **624 MB** in total. Ending on
October 30 keeps the following day in the same monthly archive. Other month
boundaries may require an additional file.

Each file is downloaded, checked against its published SHA-256 and filtered in
batches. Only ICE stops at the two stations remain in the MinIO cache. Interrupted
downloads are retried up to twice; incomplete files are not saved as valid extracts.
Subsequent runs use the cache. Only months present in the pinned dataset revision
are accepted. Features for the inference day must already exist in the feature store.

### Inspecting the outputs

- **Hopsworks:** `ice_hamburg_berlin_examples` (Feature Group),
  `ice_hamburg_berlin_prediction` (Feature View), `ice_hamburg_berlin_classifier` (model).
- **Terminal:** row counts, validation metrics and the first predictions.
- **MinIO:** bucket `rail-delay`, containing `raw/`, `reports/`, `models/latest.json`
  and `predictions/<date>/<run-id>.parquet`.
- **MinIO console:** locally at [localhost:9001](http://localhost:9001). In Codespaces,
  privately forward port `9001` in the **Ports** tab and open its URL.

`docker compose down` stops the services and retains the named MinIO volume.
Each training run creates a new model version. The `models/latest.json` manifest
is updated only after both the model and its report have been saved successfully.

## Data and label

Source: [Deutsche Bahn Data by Piet Brömmel](https://github.com/piebro/deutsche-bahn-data),
published on [Hugging Face](https://huggingface.co/datasets/piebro/deutsche-bahn-data).
The underlying sources are public DB APIs. **Attribution: Deutsche Bahn and Piet
Brömmel; data license: CC BY 4.0.** This project filters and transforms the data.
Dataset revision `6edf2ef689245f4e291cf0599149519d899eeae5` is pinned. Retrieval time,
original URL, revision, size and file hash are recorded in MinIO.

Monthly files contain scheduled and archived changed timestamps. A journey is
matched across both stations. Its key includes the service date, not just the
recurring journey identifier. Reverse journeys, canceled relevant stops, ambiguous
duplicate stops, missing timestamps and arrivals before the request are excluded.
Scheduled travel time must be in the plausible range of 1–360 minutes.

**Label `delayed_5min`:** `(arrival_change_time - arrival_planned_time) >= 5 minutes`
in Berlin. This is our project definition, not the official DB punctuality threshold.
Cancellations are not treated as on-time journeys. Missing times are not replaced
with zero delays. The source itself processes and fills API fields; we use its
archived view, not independent physical measurements.

`trip_id` is the primary key; `event_time` is observed departure in UTC. Neither is
a model feature. Local timestamps are converted using `Europe/Berlin`; ambiguous
daylight-saving timestamps are discarded. `label_available_at` is a training helper
column and is also excluded from the model.

## The three pipelines and their time boundaries

**Feature:** Retrieve source data → preserve raw data → match journeys and calculate
labels → aggregate history and derive current features → insert the DataFrame into
a Hopsworks Feature Group. The pipeline waits for the upload to complete.

**Training:** Load the Feature Group → create a Feature View selecting features and
label → read a versioned training dataset → train and validate chronologically →
save the model with `joblib` and register it in Hopsworks. Standardization and logistic
regression form one saved Scikit-Learn pipeline. `class_weight="balanced"` accounts
for class frequencies; the classification threshold remains at probability 0.5.

Approximately the first 80% of eligible training days train the validation model;
later days validate it. Labels unavailable at validation start are removed from
the training partition. After evaluation, the saved model is refitted on all
eligible training days. Reported validation metrics describe the model before
this final refit.

**Inference:** Load the specific model version selected by the manifest → initialize
its Feature View → retrieve historical features for the simulated day → independently
read current departure delay from Hamburg source records → calculate calendar
features using the shared function → predict → save results in MinIO.
**The Berlin arrival label is not used.** The Feature View returns no labels for inference.

Replay assumes that historical labels become available on the second local calendar
day after arrival. The seven-day aggregation window therefore ends at D−2. With a
September 1 cutoff, training uses departures before August 31, additionally requiring
`label_available_at <= cutoff`. August 31 is a publication buffer; inference is
allowed from September 1 onward. This check prevents overlapping periods.

## Results and limitations

Features were selected on the August validation set. September had already been
inspected during development. The final variant was therefore also checked on
**October 1–30**, without adjusting the model or threshold using that test set.

- Feature rows from June through October 30: **3,057**.
- Training partition before validation: **1,591**, including **718** delayed journeys.
- Validation from August 12: **357**, including **201** delayed journeys.
- Final refit before the September cutoff: **1,967** journeys.
- October test: **516**, including **336** delayed journeys.

| October test | Precision | Recall | F1 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.786 | 0.405 | 0.534 | 0.541 | 0.600 |
| Departure rule: departure delay >= 5 min → delayed arrival | 0.837 | 0.321 | 0.465 | 0.517 | 0.602 |
| Always delayed | 0.651 | 1.000 | 0.789 | 0.651 | 0.500 |
| Training majority class: always on time | 0.000 | 0.000 | 0.000 | 0.349 | 0.500 |

The model detects **136 of 336** delayed journeys and raises **37 false alarms**.
It detects more delays than the departure rule but also raises more false alarms.
Balanced accuracy is practically equal. Because delays are common, “always delayed”
achieves higher F1 and accuracy, but incorrectly warns for all 180 on-time journeys.
**High F1 alone therefore does not establish practical value.** The model has higher
warning precision than “always delayed”, still misses many delays and is not a
reliable travel service. Its performance is not presented as generally superior.

Further limitations:

- **Historical replay, not a live service:** The assignment allows simulated current
  requests. Monthly data is retrospectively processed. Replay treats the archived
  departure time as if it were available at departure; this cannot establish a
  guaranteed historical as-of data state.
- Assumed label availability is not a publication guarantee from the monthly archive.
- The provider documents collection gaps. Missing journeys and excluded cancellations
  can bias the distribution, even when files have been downloaded in full.
- Construction work, network occupancy and other influences are absent. Modeling
  time of day as a simple linear feature is another simplification.
- When changing feature semantics, increment `HOPSWORKS_FEATURE_VERSION` and rerun
  all three steps. Existing views are not silently redefined.

## Verification status and tests

As of **2026-09-12**: 15 automated tests passed; Ruff reported no issues. Real-data
and model checks used the periods listed above. Compose configuration and Linux
dependency resolution from `uv.lock` passed. Actual Hopsworks SDK login and metadata
access to the feature store and registry succeeded (backend 5.0.3, SDK 5.0.7).

**Still pending in Codespaces:** Container build and the complete flow from MinIO
through Hopsworks upload, training dataset creation, model registration and inference.
No local Docker engine was running. The native Windows attempt stopped before upload
because the Hopsworks Delta dependency is unavailable on Windows. The Linux container
installs it through `hopsworks[python]`. Local file-cache and mocked S3 tests do not
replace this integration test. This implementation has not yet created features or
models in the Hopsworks project.

On Linux with Python 3.12 and uv:

```sh
uv sync --locked
uv run pytest -q
uv run ruff check src tests
```

The Dockerfile installs `build-essential` because the SDK dependency `twofish`
requires a C compiler. Dependencies are defined in `pyproject.toml` and `uv.lock`.
Tests cover keys, direction, label quality, time windows, current inputs, training
splits, replay cutoff, download hashes and S3 Parquet/JSON round trips.

## Assignment requirements review

The supplied assignment explicitly excludes **model performance** from assessment:
accuracy and F1 are not grading criteria. Assessment is pass/fail and focuses on
FTI structure, use of the feature store, pipeline completeness, reproducibility and
honest reflection. Simulating current inputs by holding out historical data is
explicitly allowed. Optional extensions are not assessed.

The implementation covers the required scope, but full operational compliance is
**not yet verified**: successful feature-store storage and the complete subsequent
training/inference flow must still be demonstrated in Codespaces. Documenting a
problem is required by the assignment; it does not guarantee acceptance of an
unfinished pipeline.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Data source, target and historical/current split | Public DB archive, arrival delay >= 5 minutes, chronological cutoff and held-out replay | Implemented; real-data checks completed |
| At least one aggregated feature | `delay_mean_7d` in `features.py`; read from the Feature View for inference | Implemented and locally tested |
| At least one current RT feature | Departure observation reconstructed for the simulated request; used by the model | Implemented and locally tested |
| Feature pipeline: retrieve, engineer, label, FG with primary key/event time, insert | `feature_pipeline()` and `hops.feature_group()` | Implemented; actual Hopsworks upload pending |
| Training pipeline: FG, FV query, training dataset, fit/evaluate, save/register | `training_pipeline()` and `hops.feature_view()` | Implemented; local model checks passed, service integration pending |
| Inference pipeline: current inputs, FV retrieval, registry download, prediction | `inference_pipeline()` | Implemented; mocked integration test passed, actual service flow pending |
| Hopsworks feature store | Explicit SDK calls; no local substitute in the application | Login/metadata verified; stored features not yet verified |
| Python below 3.14 and defined dependencies | Python 3.12, `pyproject.toml`, `uv.lock`, Dockerfile | Defined; Linux dependency resolution passed |
| README describing features, model, architecture and execution | This document and `.env.example` | Completed |
| Honest discussion of limitations and incomplete functionality | Results and verification sections above | Completed |
| GitHub repository containing all three pipelines | This repository, `src/rail_delay/pipelines.py` | Completed |
| Object store and container setup (additional project choices) | MinIO and Docker Compose | Implemented; complete container run pending |

Before submission, run the three commands above in order and confirm that Hopsworks
contains the Feature Group rows, a Feature View/training dataset and a registered
model, and that MinIO contains the inference Parquet output. Record the successful
run and any remaining issues here. Submit the repository link through the Teams
assignment by **2026-11-15 23:59**, as specified in the brief.

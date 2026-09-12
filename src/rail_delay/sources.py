"""DB API archive by Piet Broemmel (CC BY 4.0)."""

import hashlib
import logging
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

LOG = logging.getLogger(__name__)
DATASET = "piebro/deutsche-bahn-data"
# Pin the data as well as the Python packages: later corrections stay explicit.
REVISION = "6edf2ef689245f4e291cf0599149519d899eeae5"
ORIGIN = "Hamburg Hbf"
DESTINATION = "Berlin Hauptbahnhof"
COLUMNS = [
    "station_name",
    "train_type",
    "id",
    "arrival_is_canceled",
    "departure_is_canceled",
    "arrival_planned_time",
    "arrival_change_time",
    "departure_planned_time",
    "departure_change_time",
]


def monthly_files():
    response = requests.get(
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/monthly_processed_data",
        timeout=60,
    )
    response.raise_for_status()
    return {Path(item["path"]).name: item for item in response.json() if item["type"] == "file"}


def download_file(url, destination, expected_sha256):
    """Stream to scratch disk, retry interrupted downloads, verify the published hash."""
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with open(destination, "wb") as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        output.write(block)
                        digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("Downloaded Parquet does not match the source SHA-256")
            return
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            LOG.warning("Download interrupted or hash mismatch; retry %d/2", attempt + 1)
            time.sleep(2)


def filter_month(path):
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=COLUMNS):
        frame = batch.to_pandas()
        parts.append(
            frame[frame["train_type"].eq("ICE") & frame["station_name"].isin([ORIGIN, DESTINATION])]
        )
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
    if result.empty:
        raise ValueError("No ICE observations at the selected stations")
    return result


def load_rail_month(month, store, files):
    key = f"raw/rail/{REVISION}/{month}.parquet"
    if store.exists(key) and store.exists(key + ".provenance.json"):
        return store.get_frame(key)
    name = f"data-{month}.parquet"
    if name not in files:
        raise ValueError(f"No {name} in the pinned dataset revision; choose an available month")
    entry = files[name]
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
    digest = entry["lfs"]["oid"]
    LOG.info(
        "Read %s (%.0f MB), filter ICE stops at Hamburg and Berlin", month, entry["size"] / 1e6
    )
    with tempfile.TemporaryDirectory(prefix="rail-delay-") as directory:
        path = Path(directory) / name
        download_file(url, path, digest)
        frame = filter_month(path)
    store.put_frame(key, frame)
    store.put_json(
        key + ".provenance.json",
        {
            "source_url": url,
            "revision": REVISION,
            "source_sha256": digest,
            "source_bytes": entry["size"],
            "rows_retained": len(frame),
            "filter": "ICE; Hamburg Hbf or Berlin Hauptbahnhof",
            "retrieved_at": datetime.now(UTC).isoformat(),
            "attribution": "Deutsche Bahn; archived and processed by Piet Broemmel; CC BY 4.0",
        },
    )
    return frame

from types import SimpleNamespace

import pandas as pd
from moto import mock_aws

from rail_delay.object_store import ObjectStore


@mock_aws
def test_s3_roundtrip_and_missing_keys():
    store = ObjectStore(
        SimpleNamespace(
            bucket="test-rail-delay", endpoint=None, access_key="test", secret_key="test"
        )
    )
    store.ensure_bucket()
    store.ensure_bucket()
    assert not store.exists("missing")
    frame = pd.DataFrame({"temperature": [4.5, -2.0], "label": [0, 1]})
    store.put_frame("data.parquet", frame)
    assert store.exists("data.parquet")
    pd.testing.assert_frame_equal(store.get_frame("data.parquet"), frame)
    store.put_json("metadata.json", {"source": "Zürich", "rows": 2})
    assert store.get_json("metadata.json") == {"source": "Zürich", "rows": 2}

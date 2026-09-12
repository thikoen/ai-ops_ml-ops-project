import hashlib
from unittest.mock import MagicMock, Mock

import pytest
import requests

from rail_delay import sources


def response(chunks):
    result = MagicMock()
    result.__enter__.return_value = result
    result.iter_content.return_value = iter(chunks)
    return result


def test_interrupted_download_retries_and_verifies_hash(tmp_path, monkeypatch):
    download = Mock(
        side_effect=[requests.ConnectionError("partial read"), response([b"valid", b"data"])]
    )
    monkeypatch.setattr(sources.requests, "get", download)
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    path = tmp_path / "download.parquet"
    sources.download_file("https://example.invalid", path, hashlib.sha256(b"validdata").hexdigest())
    assert download.call_count == 2
    assert path.read_bytes() == b"validdata"


def test_hash_mismatch_fails_instead_of_using_corrupt_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sources.requests, "get", lambda *args, **kwargs: response([b"corrupt"]))
    monkeypatch.setattr(sources.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="SHA-256"):
        sources.download_file("https://example.invalid", tmp_path / "partial", "0" * 64)


def test_failed_download_is_not_cached(monkeypatch):
    monkeypatch.setattr(
        sources, "download_file", Mock(side_effect=requests.ConnectionError("offline"))
    )
    store = Mock()
    store.exists.return_value = False
    files = {
        "data-2025-08.parquet": {
            "path": "data-2025-08.parquet",
            "size": 100,
            "lfs": {"oid": "0" * 64},
        }
    }
    with pytest.raises(requests.ConnectionError):
        sources.load_rail_month("2025-08", store, files)
    store.put_frame.assert_not_called()
    store.put_json.assert_not_called()

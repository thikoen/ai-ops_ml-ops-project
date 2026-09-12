import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

FEATURES = ["delay_mean_7d", "departure_delay_minutes", "departure_hour", "is_weekend"]

TARGET = "delayed_5min"
GROUP = "ice_hamburg_berlin_examples"
VIEW = "ice_hamburg_berlin_prediction"
MODEL = "ice_hamburg_berlin_classifier"


@dataclass(frozen=True)
class Settings:
    host: str
    project: str
    api_key: str
    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    version: int
    work: Path

    @classmethod
    def load(cls):
        load_dotenv()
        return cls(
            host=os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
            project=os.getenv("HOPSWORKS_PROJECT", ""),
            api_key=os.getenv("HOPSWORKS_API_KEY", ""),
            endpoint=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
            bucket=os.getenv("S3_BUCKET", "rail-delay"),
            access_key=os.getenv("MINIO_ROOT_USER", ""),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", ""),
            version=int(os.getenv("HOPSWORKS_FEATURE_VERSION", "1")),
            work=Path(os.getenv("WORK_DIR", "data")),
        )

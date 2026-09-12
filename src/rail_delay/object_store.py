"""MinIO/S3 keeps source extracts, provenance, reports and predictions."""

import io
import json

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError


class ObjectStore:
    def __init__(self, settings):
        self.bucket = settings.bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint,
            aws_access_key_id=settings.access_key,
            aws_secret_access_key=settings.secret_key,
            region_name="us-east-1",
            config=Config(s3={"addressing_style": "path"}),
        )

    def ensure_bucket(self):
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
                raise
            self.client.create_bucket(Bucket=self.bucket)

    def exists(self, key):
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def put_bytes(self, key, data, content_type="application/octet-stream"):
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

    def get_bytes(self, key):
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        with response["Body"] as stream:
            return stream.read()

    def put_json(self, key, value):
        self.put_bytes(
            key,
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode(),
            "application/json",
        )

    def get_json(self, key):
        return json.loads(self.get_bytes(key))

    def put_frame(self, key, frame):
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        self.put_bytes(key, buffer.getvalue())

    def get_frame(self, key):
        return pd.read_parquet(io.BytesIO(self.get_bytes(key)))

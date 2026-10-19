"""
Tiny S3-compatible storage helper (built for Cloudflare R2, works with any S3).

Everything for one training run lives under  <bucket>/<prefix>/ :
    state.json                 run status + pointer to the latest GOOD checkpoint
    data/                      the converted dataset (so every restart uses identical data)
    ckpt/checkpoint-<step>/    training checkpoints
    final-lora/                the finished LoRA adapter
    logs/                      failure logs
"""
import json
import os

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

MB = 1024 * 1024


class Store:
    def __init__(self, endpoint_url, key_id, secret, bucket, prefix, region="auto"):
        # R2 rejects the newer default CRC32 checksums, so only send them when required.
        cfg = Config(
            retries={"max_attempts": 8, "mode": "standard"},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        )
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name=region,
            config=cfg,
        )
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.tc = TransferConfig(
            multipart_threshold=16 * MB, multipart_chunksize=16 * MB, max_concurrency=8
        )

    # ---- keys -------------------------------------------------------------
    def key(self, *parts):
        return "/".join([self.prefix] + [p.strip("/") for p in parts if p])

    # ---- small json blobs -------------------------------------------------
    def get_json(self, name, default=None):
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self.key(name))["Body"].read()
            return json.loads(body)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return default
            raise

    def put_json(self, name, obj):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key(name),
            Body=json.dumps(obj, indent=2).encode(),
            ContentType="application/json",
        )

    def put_text(self, name, text):
        self.s3.put_object(Bucket=self.bucket, Key=self.key(name), Body=text.encode())

    # ---- directories ------------------------------------------------------
    def upload_dir(self, local_dir, remote_dir):
        n = 0
        for root, _, files in os.walk(local_dir):
            for f in files:
                path = os.path.join(root, f)
                rel = os.path.relpath(path, local_dir).replace(os.sep, "/")
                self.s3.upload_file(
                    path, self.bucket, self.key(remote_dir, rel), Config=self.tc
                )
                n += 1
        return n

    def _list(self, remote_dir):
        prefix = self.key(remote_dir) + "/"
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                yield obj["Key"], prefix
            if not resp.get("IsTruncated"):
                return
            token = resp["NextContinuationToken"]

    def download_dir(self, remote_dir, local_dir):
        n = 0
        for key, prefix in self._list(remote_dir):
            rel = key[len(prefix):]
            if not rel or rel.endswith("/"):
                continue
            dest = os.path.join(local_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            self.s3.download_file(self.bucket, key, dest, Config=self.tc)
            n += 1
        return n

    def delete_dir(self, remote_dir):
        # One at a time: R2 is picky about batch DeleteObjects headers.
        keys = [k for k, _ in self._list(remote_dir)]
        for k in keys:
            self.s3.delete_object(Bucket=self.bucket, Key=k)
        return len(keys)

    def list_subdirs(self, remote_dir):
        """Names of the immediate 'folders' under remote_dir."""
        prefix = self.key(remote_dir) + "/"
        names, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            names += [p["Prefix"][len(prefix):].strip("/") for p in resp.get("CommonPrefixes", [])]
            if not resp.get("IsTruncated"):
                return names
            token = resp["NextContinuationToken"]


def from_env(env=os.environ):
    """Build a Store from R2_* environment variables, or raise a clear error."""
    need = ["R2_ENDPOINT_URL", "R2_KEY_ID", "R2_SECRET", "R2_BUCKET"]
    missing = [k for k in need if not env.get(k)]
    if missing:
        raise RuntimeError("missing environment variables: " + ", ".join(missing))
    return Store(
        env["R2_ENDPOINT_URL"],
        env["R2_KEY_ID"],
        env["R2_SECRET"],
        env["R2_BUCKET"],
        env.get("R2_PREFIX", "gemma4-discord"),
    )

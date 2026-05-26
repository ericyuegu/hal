"""Cloudflare R2 (S3-compatible) client + bucket resolution.

Single source of truth for the R2 endpoint and `AWS_*` credentials, shared by
fixture downloads (`hal.fixtures`) and training checkpoint sync
(`hal.training.checkpoints`). One client constructor means creds/endpoint live
in exactly one place — see `.env.example` for the variables.
"""

import os
from typing import Final

import boto3

_CRED_VARS: Final[tuple[str, ...]] = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


class R2Error(RuntimeError):
    pass


def missing_credentials() -> list[str]:
    """Required R2 env vars that are unset (empty list ⇒ ready to connect)."""
    return [v for v in _CRED_VARS if not os.environ.get(v)]


def bucket() -> str:
    name = os.environ.get("AWS_BUCKET")
    if not name:
        raise R2Error("AWS_BUCKET env var not set. See .env.example.")
    return name


def client():  # type: ignore[no-untyped-def]
    """boto3 S3 client against R2's endpoint. Raises `R2Error` if creds are missing."""
    missing = missing_credentials()
    if missing:
        raise R2Error(f"missing env vars for R2: {missing}. See .env.example.")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )

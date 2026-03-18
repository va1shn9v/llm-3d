from __future__ import annotations

from types import SimpleNamespace

from config import StorageConfig
from data import storage


def test_bucket_uri_uses_explicit_namespace(monkeypatch):
    cfg = StorageConfig(hf_bucket="llm3d-data", hf_bucket_namespace="my-org")

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    storage._resolve_hf_bucket_namespace.cache_clear()

    assert storage.bucket_uri("meshes/file.obj", cfg) == "hf://buckets/my-org/llm3d-data/meshes/file.obj"


def test_bucket_uri_resolves_namespace_from_token(monkeypatch):
    cfg = StorageConfig(hf_bucket="llm3d-data", hf_bucket_namespace="")

    monkeypatch.setenv("HF_TOKEN", "test-token")
    storage._resolve_hf_bucket_namespace.cache_clear()

    class FakeHfApi:
        def __init__(self, token: str):
            assert token == "test-token"

        def whoami(self):
            return {"name": "demo-user"}

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    assert storage.bucket_uri("meshes/file.obj", cfg) == "hf://buckets/demo-user/llm3d-data/meshes/file.obj"


def test_ensure_bucket_exists_creates_expected_bucket(monkeypatch):
    cfg = StorageConfig(hf_bucket="llm3d-data", hf_bucket_namespace="")

    monkeypatch.setenv("HF_TOKEN", "test-token")
    storage._resolve_hf_bucket_namespace.cache_clear()

    class FakeHfApi:
        def __init__(self, token: str):
            assert token == "test-token"

        def whoami(self):
            return {"name": "demo-user"}

        def create_bucket(self, bucket_id: str, exist_ok: bool):
            calls.append(SimpleNamespace(bucket_id=bucket_id, exist_ok=exist_ok, token="test-token"))

    calls: list[SimpleNamespace] = []

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)

    uri = storage.ensure_bucket_exists(cfg)

    assert uri == "hf://buckets/demo-user/llm3d-data"
    assert len(calls) == 1
    assert calls[0].bucket_id == "demo-user/llm3d-data"
    assert calls[0].exist_ok is True
    assert calls[0].token == "test-token"

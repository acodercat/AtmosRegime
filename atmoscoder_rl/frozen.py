"""Artifact records for K teaching and D preparation, independent of the frozen T input generator.

T's existing kc serializer is retained unchanged for byte-for-byte compatibility with its bundle.
"""
import hashlib
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False)+'\n').encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bundle(out: Path, artifacts: dict[str, bytes]) -> None:
    for name, blob in artifacts.items():
        path = out/name
        require(not path.exists() or path.read_bytes() == blob, f'refusing to overwrite different artifact: {path}')
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in artifacts.items():
        (out/name).write_bytes(blob)

"""Canonical JSON codecs for DIBLE Lab research artifacts; not a wire standard."""
from __future__ import annotations
import base64
import json
from pathlib import Path
from research.dible_v2 import Ciphertext, PublicKey, SecretKey


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def public_key_to_dict(key: PublicKey) -> dict:
    return {"kind": "diblelab-public-key", "version": 1, "a": [list(row) for row in key.a], "b": list(key.b), "device_commitment": _b64(key.device_commitment)}


def secret_key_to_dict(key: SecretKey) -> dict:
    return {"kind": "diblelab-secret-key", "version": 1, "s": list(key.s), "device_commitment": _b64(key.device_commitment)}


def ciphertext_to_dict(value: Ciphertext) -> dict:
    return {"kind": "diblelab-ciphertext", "version": 1, "u": list(value.u), "v": value.v, "device_commitment": _b64(value.device_commitment)}


def public_key_from_dict(data: dict) -> PublicKey:
    _check(data, "diblelab-public-key")
    return PublicKey(tuple(tuple(int(x) for x in row) for row in data["a"]), tuple(int(x) for x in data["b"]), _unb64(data["device_commitment"]))


def secret_key_from_dict(data: dict) -> SecretKey:
    _check(data, "diblelab-secret-key")
    return SecretKey(tuple(int(x) for x in data["s"]), _unb64(data["device_commitment"]))


def ciphertext_from_dict(data: dict) -> Ciphertext:
    _check(data, "diblelab-ciphertext")
    return Ciphertext(tuple(int(x) for x in data["u"]), int(data["v"]), _unb64(data["device_commitment"]))


def write_artifact(path: str | Path, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    target.chmod(0o600)


def read_artifact(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check(data: dict, kind: str) -> None:
    if data.get("kind") != kind or data.get("version") != 1:
        raise ValueError("unsupported or malformed DIBLE Lab artifact")

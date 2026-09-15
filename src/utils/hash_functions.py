"""Secure hash utilities."""
from __future__ import annotations
import hashlib
import hmac


def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def sha3_512(data: bytes) -> bytes:
    return hashlib.sha3_512(data).digest()

def blake2b_256(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()

def constant_time_compare(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)

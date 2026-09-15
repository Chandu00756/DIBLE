"""DIBLE v2 research-only device-bound lattice KEM experiment.

This module is intentionally not exposed through the vault CLI and must not be
used to secure real data. It exists to make the DIBLE construction testable,
parameterized, and reproducible as a cryptography research artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import secrets
from typing import Sequence

Vector = tuple[int, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True)
class Parameters:
    n: int = 32
    q: int = 12_289
    eta: int = 2
    domain: bytes = b"DIBLE-V2-RESEARCH-ONLY"

    def __post_init__(self) -> None:
        if self.n < 8 or self.n > 128:
            raise ValueError("n must be in [8, 128] for this reference implementation")
        if self.q < 257 or self.q % 2 == 0:
            raise ValueError("q must be an odd modulus >= 257")
        if self.eta < 1:
            raise ValueError("eta must be positive")


@dataclass(frozen=True)
class PublicKey:
    a: Matrix
    b: Vector
    device_commitment: bytes


@dataclass(frozen=True)
class SecretKey:
    s: Vector
    device_commitment: bytes


@dataclass(frozen=True)
class Ciphertext:
    u: Vector
    v: int
    device_commitment: bytes


def device_commitment(device_claim: str, salt: bytes | None = None) -> bytes:
    """Create a non-reversible commitment; never use raw hardware identifiers."""
    if not device_claim:
        raise ValueError("device_claim is required")
    salt = salt or secrets.token_bytes(32)
    return salt + hashlib.sha3_256(salt + device_claim.encode()).digest()


def _rng_int(q: int) -> int:
    return secrets.randbelow(q)


def _sample_small(eta: int) -> int:
    return sum(secrets.randbelow(2) for _ in range(eta)) - sum(secrets.randbelow(2) for _ in range(eta))


def _dot(left: Sequence[int], right: Sequence[int], q: int) -> int:
    return sum(x * y for x, y in zip(left, right)) % q


def _mat_vec(matrix: Matrix, vector: Vector, q: int) -> Vector:
    return tuple(_dot(row, vector, q) for row in matrix)


def _transpose(matrix: Matrix) -> Matrix:
    return tuple(tuple(row[i] for row in matrix) for i in range(len(matrix[0])))


def _add(left: Sequence[int], right: Sequence[int], q: int) -> Vector:
    return tuple((x + y) % q for x, y in zip(left, right))


def _bit_from_shared(value: int, q: int) -> int:
    return 1 if q // 4 <= value < (3 * q) // 4 else 0


def keygen(params: Parameters, commitment: bytes) -> tuple[PublicKey, SecretKey]:
    if len(commitment) != 64:
        raise ValueError("commitment must be 64 bytes from device_commitment")
    a = tuple(tuple(_rng_int(params.q) for _ in range(params.n)) for _ in range(params.n))
    s = tuple(_sample_small(params.eta) for _ in range(params.n))
    e = tuple(_sample_small(params.eta) for _ in range(params.n))
    b = _add(_mat_vec(a, s, params.q), e, params.q)
    return PublicKey(a, b, commitment), SecretKey(s, commitment)


def encapsulate(params: Parameters, public_key: PublicKey, commitment: bytes) -> tuple[Ciphertext, bytes]:
    if not secrets.compare_digest(public_key.device_commitment, commitment):
        raise ValueError("device commitment does not match public key")
    r = tuple(_sample_small(params.eta) for _ in range(params.n))
    e1 = tuple(_sample_small(params.eta) for _ in range(params.n))
    e2 = _sample_small(params.eta)
    bit = secrets.randbelow(2)
    u = _add(_mat_vec(_transpose(public_key.a), r, params.q), e1, params.q)
    v = (_dot(public_key.b, r, params.q) + e2 + bit * (params.q // 2)) % params.q
    ct = Ciphertext(u, v, commitment)
    return ct, _derive_shared(params, bit, ct)


def decapsulate(params: Parameters, secret_key: SecretKey, ciphertext: Ciphertext, commitment: bytes) -> bytes:
    if not secrets.compare_digest(secret_key.device_commitment, commitment):
        raise ValueError("device commitment does not match secret key")
    if not secrets.compare_digest(ciphertext.device_commitment, commitment):
        raise ValueError("ciphertext is not bound to this device")
    bit = _bit_from_shared((ciphertext.v - _dot(secret_key.s, ciphertext.u, params.q)) % params.q, params.q)
    return _derive_shared(params, bit, ciphertext)


def _derive_shared(params: Parameters, bit: int, ciphertext: Ciphertext) -> bytes:
    body = b"".join(x.to_bytes(2, "big") for x in ciphertext.u) + ciphertext.v.to_bytes(2, "big")
    return hashlib.sha3_256(params.domain + bytes([bit]) + ciphertext.device_commitment + body).digest()


def security_notes(params: Parameters) -> dict[str, int | str]:
    return {
        "dimension": params.n,
        "modulus": params.q,
        "estimated_public_key_bytes": params.n * params.n * math.ceil(params.q.bit_length() / 8),
        "status": "research-only; no security level, IND-CCA claim, or quantum-resistance claim",
    }

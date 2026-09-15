#!/usr/bin/env python3
"""Post-quantum KEM stub using LWE lattice primitives."""
from __future__ import annotations
import secrets
import hashlib
from src.core.lattice import LatticeOperations


class PostQuantumKEM:
    """Toy LWE-based key-encapsulation mechanism (research / educational)."""

    def __init__(self, dimension: int = 256):
        self.lwe = LatticeOperations(dimension=dimension)

    def keygen(self) -> dict:
        return self.lwe.generate_keypair()

    def encapsulate(self, pub: dict) -> tuple[bytes, dict]:
        """Return (shared_secret_32_bytes, ciphertext_dict)."""
        bits = [secrets.randbelow(2) for _ in range(256)]
        cts = [self.lwe.encrypt_vector(pub['public'], b) for b in bits]
        raw = bytes(bits)
        shared = hashlib.sha3_256(raw).digest()
        return shared, {'cts': cts}

    def decapsulate(self, priv: dict, ctxt: dict) -> bytes:
        bits = [self.lwe.decrypt_vector(priv, ct) for ct in ctxt['cts']]
        raw = bytes(bits)
        return hashlib.sha3_256(raw).digest()

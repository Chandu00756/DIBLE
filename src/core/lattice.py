#!/usr/bin/env python3
"""Learning-With-Errors lattice primitives (research-grade, stdlib + numpy only)."""
from __future__ import annotations
import secrets
import numpy as np


class LatticeOperations:
    def __init__(self, dimension: int = 256, modulus: int = 2 ** 16 - 17):
        self.dimension = dimension
        self.modulus = modulus

    def _rng(self) -> np.random.Generator:
        seed = int.from_bytes(secrets.token_bytes(8), "big")
        return np.random.default_rng(seed)

    def generate_random_matrix(self, rows: int, cols: int) -> np.ndarray:
        return self._rng().integers(0, self.modulus, size=(rows, cols), dtype=np.int64)

    def generate_keypair(self) -> dict:
        rng = self._rng()
        A = rng.integers(0, self.modulus, size=(self.dimension, self.dimension), dtype=np.int64)
        s = rng.integers(0, 2, size=self.dimension, dtype=np.int64)
        e = rng.integers(0, 4, size=self.dimension, dtype=np.int64)
        b = (A @ s + e) % self.modulus
        return {"public": {"A": A, "b": b}, "private": {"s": s}}

    def encrypt_vector(self, pub: dict, message_bit: int) -> dict:
        rng = self._rng()
        A, b = pub["A"], pub["b"]
        r  = rng.integers(0, 2, size=self.dimension, dtype=np.int64)
        e1 = rng.integers(0, 4, size=self.dimension, dtype=np.int64)
        e2 = int(rng.integers(0, 4))
        u = (A.T @ r + e1) % self.modulus
        v = (int(b @ r) + e2 + (self.modulus // 2) * message_bit) % self.modulus
        return {"u": u, "v": v}

    def decrypt_vector(self, priv: dict, ctxt: dict) -> int:
        s = priv["s"]
        v, u = ctxt["v"], ctxt["u"]
        phase = (v - int(s @ u)) % self.modulus
        return 1 if abs(phase - self.modulus // 2) < self.modulus // 4 else 0

#!/usr/bin/env python3
"""Quantum-inspired mathematical helpers (no external quantum SDK required)."""
from __future__ import annotations
import cmath
import secrets
from typing import List


class QuantumOps:
    @staticmethod
    def hadamard_mix(bits: List[int]) -> List[complex]:
        return [(1 + 1j) / (2 ** 0.5) * b for b in bits]

    @staticmethod
    def measure(amplitudes: List[complex]) -> List[int]:
        return [1 if abs(a) >= 0.5 else 0 for a in amplitudes]

    @staticmethod
    def random_qubit_bytes(n: int) -> bytes:
        return secrets.token_bytes(n)

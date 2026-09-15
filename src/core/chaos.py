#!/usr/bin/env python3
"""Chaos-theory entropy helpers seeded from OS CSPRNG (not time)."""
from __future__ import annotations
import secrets
from typing import List


class ChaosTheoryManager:
    SYSTEMS = {"logistic", "lorenz", "henon"}

    def __init__(self, device_id_transform: int = 0):
        raw = secrets.token_bytes(8)
        self._x = 0.1 + (int.from_bytes(raw, "big") % 10_000) / 100_000.0
        self._r = 3.9

    def generate_chaotic_sequence(self, length: int = 64,
                                   system: str = "logistic") -> List[float]:
        """Return `length` chaotic floats in (0, 1) via logistic map."""
        x = self._x
        # warm-up
        for _ in range(200):
            x = self._r * x * (1.0 - x)
        seq: List[float] = []
        for _ in range(length):
            x = self._r * x * (1.0 - x)
            seq.append(x)
        return seq

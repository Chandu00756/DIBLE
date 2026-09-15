#!/usr/bin/env python3
"""Chaos-theory entropy helper seeded from OS CSPRNG."""
from __future__ import annotations
import secrets
import math
from typing import List


class ChaosTheoryManager:
    SYSTEMS = {'logistic', 'lorenz', 'henon'}

    def __init__(self, device_id_transform: int = 0):
        raw = secrets.token_bytes(8)
        self._x = 0.1 + (int.from_bytes(raw, 'big') % 10000) / 100000.0
        self._r = 3.9

    def generate_chaotic_sequence(self, length: int = 64, system: str = 'logistic') -> List[float]:
        seq: List[float] = []
        x = self._x
        for _ in range(length + 100):
            x = self._r * x * (1.0 - x)
        for _ in range(length):
            x = self._r * x * (1.0 - x)
            seq.append(x)
        return seq

#!/usr/bin/env python3
"""Multi-source entropy pool combining OS entropy and chaos."""
from __future__ import annotations
import math
import secrets
import hashlib
from typing import Any, Dict, List


class EntropyManager:
    def __init__(self, device_id_transform: int = 0):
        self._device_salt = device_id_transform.to_bytes(8, 'big', signed=False) if device_id_transform >= 0 else (device_id_transform & 0xFFFFFFFFFFFFFFFF).to_bytes(8, 'big')

    def collect(self, n_bytes: int = 64) -> bytes:
        """Return n_bytes of entropy mixed from OS RNG and device salt."""
        raw = secrets.token_bytes(n_bytes)
        h = hashlib.sha3_512(raw + self._device_salt)
        full = h.digest()
        while len(full) < n_bytes:
            full += hashlib.sha3_512(full + self._device_salt).digest()
        return full[:n_bytes]

    def multidimensional_entropy(self, data: Dict[str, Any]) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for key, value in data.items():
            values: List[float] = []
            if isinstance(value, (list, tuple)):
                values = [float(v) for v in value]
            elif isinstance(value, (int, float)):
                values = [float(value)]
            if not values:
                continue
            total = sum(values) or 1.0
            probs = [v / total for v in values if v > 0]
            shannon = -sum(p * math.log2(p) for p in probs if p > 0)
            results[f'{key}_shannon'] = shannon
        return results

#!/usr/bin/env python3
"""Multi-source entropy pool combining OS entropy and device salt."""
from __future__ import annotations
import hashlib
import math
import secrets
from typing import Any, Dict, List


class EntropyManager:
    def __init__(self, device_id_transform: int = 0):
        n = device_id_transform & 0xFFFFFFFFFFFFFFFF
        self._device_salt = n.to_bytes(8, "big")

    def collect(self, n_bytes: int = 64) -> bytes:
        """Return n_bytes of entropy mixed from OS RNG and device salt."""
        raw = secrets.token_bytes(max(n_bytes, 64))
        h = hashlib.sha3_512(raw + self._device_salt).digest()
        while len(h) < n_bytes:
            h += hashlib.sha3_512(h + self._device_salt).digest()
        return h[:n_bytes]

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
            results[f"{key}_shannon"] = shannon
        return results

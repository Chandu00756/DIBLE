#!/usr/bin/env python3
"""Polynomial ring operations for lattice cryptography."""
from __future__ import annotations
import secrets
from typing import Dict, List


class PolynomialOperations:
    def __init__(self, modulus: int = 97):
        self.modulus = modulus

    def random_polynomial(self, max_degree: int = 4, num_terms: int = 4,
                          device_id_transform: int = 0) -> Dict:
        coeffs = [
            int.from_bytes(secrets.token_bytes(4), 'big') % self.modulus
            for _ in range(num_terms)
        ]
        return {'coefficients': coeffs, 'modulus': self.modulus}

    def multiply(self, p1: List[int], p2: List[int]) -> List[int]:
        deg = len(p1) + len(p2) - 1
        result = [0] * deg
        for i, a in enumerate(p1):
            for j, b in enumerate(p2):
                result[i + j] = (result[i + j] + a * b) % self.modulus
        return result

"""Numpy tensor helpers for lattice operations."""
from __future__ import annotations
import numpy as np


def matmul_mod(A: np.ndarray, B: np.ndarray, q: int) -> np.ndarray:
    return np.mod(A @ B, q)

def inner_mod(a: np.ndarray, b: np.ndarray, q: int) -> int:
    return int(np.dot(a, b) % q)

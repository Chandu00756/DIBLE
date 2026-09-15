"""Math helpers for cryptographic operations."""
from __future__ import annotations
from typing import List


def mod_inverse(a: int, m: int) -> int:
    return pow(a, -1, m)

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def poly_add(a: List[int], b: List[int], q: int) -> List[int]:
    length = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) % q for i in range(length)]

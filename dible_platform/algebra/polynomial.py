from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Polynomial:
    coefficients: tuple[int,...]; modulus: int
    def __post_init__(self):
        if self.modulus < 3: raise ValueError('modulus must be >= 3')
    def evaluate(self, x: int) -> int:
        result=0
        for coefficient in reversed(self.coefficients): result=(result*x+coefficient)%self.modulus
        return result

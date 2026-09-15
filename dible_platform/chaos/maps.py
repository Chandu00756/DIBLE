from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class LogisticMap:
    r: float = 3.99
    def sequence(self, seed: float, count: int) -> tuple[float,...]:
        if not 0.0 < seed < 1.0 or not 0 < count <= 100000 or not 3.57 < self.r <= 4.0: raise ValueError('invalid logistic-map parameters')
        out=[]; x=seed
        for _ in range(count): x=self.r*x*(1-x); out.append(x)
        return tuple(out)

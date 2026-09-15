from __future__ import annotations
from dataclasses import dataclass
from research.dible_v2 import Parameters, device_commitment, keygen, encapsulate, decapsulate
@dataclass
class LatticeEngine:
    params: Parameters = Parameters()
    def keypair(self, claim: str, salt: bytes):
        commitment=device_commitment(claim,salt); return (*keygen(self.params,commitment), commitment)
    def round_trip(self, claim: str, salt: bytes) -> bool:
        pk,sk,c=self.keypair(claim,salt); ct,sent=encapsulate(self.params,pk,c); return decapsulate(self.params,sk,ct,c)==sent

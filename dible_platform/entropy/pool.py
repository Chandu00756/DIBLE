from __future__ import annotations
import hashlib, os, secrets
class EntropyPool:
    def __init__(self): self._counter=0
    def derive(self, context: bytes, length: int=32) -> bytes:
        if not context or not 1 <= length <= 64: raise ValueError('context and length 1..64 required')
        self._counter += 1; return hashlib.shake_256(b'DIBLE-ENTROPY-V1'+os.urandom(32)+secrets.token_bytes(32)+self._counter.to_bytes(8,'big')+context).digest(length)

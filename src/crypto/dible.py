#!/usr/bin/env python3
"""DIBLE core: AES-256-GCM encryption with PBKDF2 or key-file key derivation."""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import secrets
import struct
from dataclasses import dataclass, field
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

# Container format version
FORMAT_VERSION = 1
SALT_BYTES = 32
NONCE_BYTES = 12
KEY_BYTES = 32  # AES-256
PBKDF2_ITERS = 600_000  # OWASP 2024 recommendation


@dataclass
class DIBLEKey:
    key_id: str
    raw_key: bytes  # 32 bytes
    device_id: str
    algorithm: str = 'AES-256-GCM'
    kdf: str = 'PBKDF2-SHA3-256'
    created: str = ''

    def __post_init__(self):
        if not self.created:
            from datetime import datetime, timezone
            self.created = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            'key_id': self.key_id,
            'device_id': self.device_id,
            'algorithm': self.algorithm,
            'kdf': self.kdf,
            'created': self.created,
        }  # raw_key is NEVER serialised


class DIBLECore:
    """AES-256-GCM vault with password-based or raw-key encryption."""

    def __init__(self, device_id: str):
        self.device_id = device_id

    # ── key derivation ────────────────────────────────────────────────────
    def derive_key(self, password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
        """Return (raw_key_32, salt_32). Pass existing salt to reproduce key."""
        if salt is None:
            salt = secrets.token_bytes(SALT_BYTES)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_BYTES,
            salt=salt + self.device_id.encode(),
            iterations=PBKDF2_ITERS,
        )
        raw = kdf.derive(password.encode('utf-8'))
        return raw, salt

    def generate_key(self) -> DIBLEKey:
        """Generate a random 256-bit vault key."""
        raw = secrets.token_bytes(KEY_BYTES)
        kid = hashlib.sha256(raw + self.device_id.encode()).hexdigest()[:16]
        return DIBLEKey(key_id=kid, raw_key=raw, device_id=self.device_id)

    # ── encrypt / decrypt ────────────────────────────────────────────────
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Return versioned ciphertext blob: [version(1)|nonce(12)|tag(16)|ct]."""
        nonce = secrets.token_bytes(NONCE_BYTES)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext, self.device_id.encode())
        return struct.pack('B', FORMAT_VERSION) + nonce + ct

    def decrypt(self, blob: bytes, key: bytes) -> bytes:
        """Inverse of encrypt(). Raises ValueError on tamper / wrong key."""
        if len(blob) < 1 + NONCE_BYTES + 16:
            raise ValueError('Ciphertext too short')
        version = blob[0]
        if version != FORMAT_VERSION:
            raise ValueError(f'Unsupported container version {version}')
        nonce = blob[1: 1 + NONCE_BYTES]
        ct = blob[1 + NONCE_BYTES:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, self.device_id.encode())

    # ── file helpers ─────────────────────────────────────────────────────
    def encrypt_file(self, src: str, dst: str, key: bytes) -> None:
        blob = self.encrypt(open(src, 'rb').read(), key)
        open(dst, 'wb').write(blob)

    def decrypt_file(self, src: str, dst: str, key: bytes) -> None:
        pt = self.decrypt(open(src, 'rb').read(), key)
        open(dst, 'wb').write(pt)

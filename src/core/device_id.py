#!/usr/bin/env python3
"""Stable, deterministic device identity derived from hardware attributes."""
from __future__ import annotations
import hashlib
import platform
import socket
import uuid


class DeviceIDGenerator:
    """Derive a stable 64-hex device fingerprint from hardware characteristics."""

    def generate_device_id(self) -> str:
        parts = [
            platform.node(),
            platform.machine(),
            platform.processor() or 'unknown',
            str(uuid.getnode()),  # MAC address as integer
        ]
        raw = ':'.join(parts).encode()
        return hashlib.sha3_256(raw).hexdigest()

    def get_device_bytes(self) -> bytes:
        return bytes.fromhex(self.generate_device_id())

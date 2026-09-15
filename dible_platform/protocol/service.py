from __future__ import annotations
from dataclasses import dataclass
from dible_platform.identity.service import DeviceRegistry
from dible_platform.lattice.engine import LatticeEngine
@dataclass
class DibleProtocol:
    devices: DeviceRegistry
    lattice: LatticeEngine
    def validate_device_bound_exchange(self, device_id: str, claim: str, salt: bytes) -> bool:
        self.devices.require_active(device_id); return self.lattice.round_trip(claim,salt)

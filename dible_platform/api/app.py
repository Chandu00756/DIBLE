"""Framework-neutral local control-plane facade; HTTP adapter intentionally separate."""
from __future__ import annotations
from dible_platform.identity.service import DeviceRegistry
from dible_platform.protocol.service import DibleProtocol
from dible_platform.lattice.engine import LatticeEngine
from dible_platform.audit.chain import AuditChain
from dible_platform.observability.service import Observatory
class ControlPlane:
    def __init__(self): self.devices=DeviceRegistry(); self.protocol=DibleProtocol(self.devices,LatticeEngine()); self.audit=AuditChain(); self.observatory=Observatory()
    def enroll(self, claim: str, salt: bytes):
        device,salt=self.devices.enroll(claim,salt); self.audit.append('device.enrolled',device.device_id,{}); return device,salt
    def health(self) -> dict: return self.observatory.snapshot(self.audit)

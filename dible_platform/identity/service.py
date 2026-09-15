from __future__ import annotations
import hashlib, secrets
from dible_platform.models import Device, DeviceState, utcnow
class DeviceRegistry:
    def __init__(self): self._devices: dict[str, Device] = {}
    def enroll(self, claim: str, salt: bytes | None = None) -> tuple[Device, bytes]:
        if not claim.strip(): raise ValueError('device claim is required')
        salt = salt or secrets.token_bytes(32); digest = hashlib.sha3_256(salt + claim.encode()).hexdigest()
        device = Device(digest[:24], digest, DeviceState.ENROLLED, utcnow()); self._devices[device.device_id] = device; return device, salt
    def require_active(self, device_id: str) -> Device:
        device = self._devices.get(device_id)
        if not device or device.state is not DeviceState.ENROLLED: raise PermissionError('device is absent or not active')
        return device
    def revoke(self, device_id: str) -> Device:
        current = self.require_active(device_id); revoked = Device(current.device_id,current.commitment,DeviceState.REVOKED,current.created_at); self._devices[device_id]=revoked; return revoked

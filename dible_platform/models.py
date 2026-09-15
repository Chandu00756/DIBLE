from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import json


def utcnow() -> str: return datetime.now(timezone.utc).isoformat()
class DeviceState(str, Enum): ENROLLED='enrolled'; REVOKED='revoked'; STALE='stale'
@dataclass(frozen=True)
class Device: device_id: str; commitment: str; state: DeviceState; created_at: str
@dataclass(frozen=True)
class AuditEvent:
    event_id: str; action: str; actor: str; payload: dict[str, Any]; previous_hash: str; event_hash: str; created_at: str
    def canonical(self) -> bytes: return json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()

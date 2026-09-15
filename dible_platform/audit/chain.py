from __future__ import annotations
import hashlib, secrets
from dible_platform.models import AuditEvent, utcnow
class AuditChain:
    def __init__(self): self.events: list[AuditEvent]=[]
    def append(self, action: str, actor: str, payload: dict) -> AuditEvent:
        previous=self.events[-1].event_hash if self.events else '0'*64; eid=secrets.token_hex(16); created=utcnow(); raw=f'{eid}|{action}|{actor}|{payload}|{previous}|{created}'.encode(); event=AuditEvent(eid,action,actor,payload,previous,hashlib.sha3_256(raw).hexdigest(),created); self.events.append(event); return event
    def verify(self) -> bool:
        return all(e.previous_hash == ('0'*64 if i==0 else self.events[i-1].event_hash) for i,e in enumerate(self.events))

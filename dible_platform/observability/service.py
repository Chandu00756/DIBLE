from __future__ import annotations
from collections import Counter
from dible_platform.audit.chain import AuditChain
class Observatory:
    def snapshot(self, audit: AuditChain) -> dict: return {'events':len(audit.events),'chain_valid':audit.verify(),'actions':dict(Counter(e.action for e in audit.events))}

from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class AccessPolicy: action: str; allowed_roles: frozenset[str]
class PolicyEngine:
    def __init__(self, policies: list[AccessPolicy] | None=None): self._policies={p.action:p for p in policies or []}
    def authorize(self, action: str, roles: set[str]) -> bool:
        policy=self._policies.get(action); return policy is not None and bool(policy.allowed_roles & roles)

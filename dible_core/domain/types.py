from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
class Role(StrEnum): OWNER='owner'; ADMIN='admin'; OPERATOR='operator'; AUDITOR='auditor'; VIEWER='viewer'
class DeviceStatus(StrEnum): PENDING='pending'; ACTIVE='active'; REVOKED='revoked'; RECOVERY_REQUIRED='recovery_required'
class KeyStatus(StrEnum): ACTIVE='active'; RETIRED='retired'; REVOKED='revoked'
class AlertSeverity(StrEnum): INFO='info'; LOW='low'; MEDIUM='medium'; HIGH='high'; CRITICAL='critical'
@dataclass(frozen=True)
class Page:
 limit:int=50; offset:int=0
 def normalized(self,max_size:int=100)->'Page':
  if self.limit<1 or self.offset<0: raise ValueError('invalid pagination')
  return Page(min(self.limit,max_size),self.offset)

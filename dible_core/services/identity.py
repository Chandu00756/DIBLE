from __future__ import annotations
import hashlib,json,secrets
from dible_core.domain.types import DeviceStatus
from dible_core.storage.repository import Repository,ident,now
from dible_core.services.audit import AuditService
class IdentityService:
 def __init__(self,repo):self.repo=repo;self.audit=AuditService(repo)
 def enroll(self,org,actor,label,claim,salt,metadata=None):
  if len(salt)<16 or not claim.strip():raise ValueError('valid claim and at least 16 salt bytes required')
  commitment=hashlib.sha3_256(salt+claim.encode()).hexdigest(); did=ident(); stamp=now()
  self.repo.insert('devices',{'id':did,'org_id':org,'label':label,'commitment':commitment,'status':DeviceStatus.ACTIVE,'metadata':json.dumps(metadata or {},sort_keys=True),'created_at':stamp,'updated_at':stamp});self.audit.record(org,'device.enrolled',actor,{'device_id':did,'label':label});return {'id':did,'commitment':commitment,'status':'active'}
 def revoke(self,org,actor,device):
  with self.repo.transaction() as c:
   changed=c.execute("UPDATE devices SET status=?,updated_at=? WHERE id=? AND org_id=? AND status=?",(DeviceStatus.REVOKED,now(),device,org,DeviceStatus.ACTIVE)).rowcount
  if not changed:raise LookupError('active device not found')
  self.audit.record(org,'device.revoked',actor,{'device_id':device})
 def active(self,org,device):
  with self.repo.transaction() as c:r=c.execute('SELECT * FROM devices WHERE id=? AND org_id=? AND status=?',(device,org,DeviceStatus.ACTIVE)).fetchone()
  if not r:raise PermissionError('active device required')
  return dict(r)

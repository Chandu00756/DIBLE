from __future__ import annotations
import json
from dible_core.storage.repository import Repository,ident,now
from dible_core.services.audit import AuditService
class VaultService:
 def __init__(self,repo):self.repo=repo;self.audit=AuditService(repo)
 def create(self,org,actor,name,classification='research'):
  if not name.strip() or len(name)>120:raise ValueError('invalid vault name')
  vid=ident();self.repo.insert('vaults',{'id':vid,'org_id':org,'name':name,'classification':classification,'created_at':now()});self.audit.record(org,'vault.created',actor,{'vault_id':vid});return vid
 def rotate_key(self,org,actor,vault,algorithm,public_material,device=None):
  with self.repo.transaction() as c:
   exists=c.execute('SELECT 1 FROM vaults WHERE id=? AND org_id=?',(vault,org)).fetchone()
   if not exists:raise LookupError('vault not found')
   current=c.execute('SELECT COALESCE(MAX(version),0) value FROM key_versions WHERE vault_id=?',(vault,)).fetchone()['value'];version=current+1; kid=ident();c.execute('UPDATE key_versions SET status=?,retired_at=? WHERE vault_id=? AND status=?',('retired',now(),vault,'active'));c.execute('INSERT INTO key_versions VALUES(?,?,?,?,?,?,?,?,?)',(kid,vault,version,'active',algorithm,json.dumps(public_material,sort_keys=True),device,now(),None))
  self.audit.record(org,'key.rotated',actor,{'vault_id':vault,'key_id':kid,'version':version});return {'id':kid,'version':version,'status':'active'}
 def revoke_key(self,org,actor,key):
  with self.repo.transaction() as c:
   row=c.execute('SELECT k.vault_id FROM key_versions k JOIN vaults v ON v.id=k.vault_id WHERE k.id=? AND v.org_id=?',(key,org)).fetchone()
   if not row:raise LookupError('key not found')
   c.execute("UPDATE key_versions SET status=?,retired_at=? WHERE id=?",('revoked',now(),key))
  self.audit.record(org,'key.revoked',actor,{'key_id':key})

from __future__ import annotations
import hashlib,json
from dible_core.storage.repository import Repository,ident,now
class AuditService:
 def __init__(self,repo:Repository): self.repo=repo
 def record(self,org_id,action,actor_id,payload):
  with self.repo.transaction() as c:
   row=c.execute('SELECT event_hash FROM audit_events WHERE org_id=? ORDER BY created_at DESC LIMIT 1',(org_id,)).fetchone(); previous=row['event_hash'] if row else '0'*64; eid=ident(); created=now(); encoded=json.dumps(payload,sort_keys=True,separators=(',',':')); digest=hashlib.sha3_256(f'{eid}|{org_id}|{action}|{actor_id}|{encoded}|{previous}|{created}'.encode()).hexdigest(); c.execute('INSERT INTO audit_events VALUES(?,?,?,?,?,?,?,?)',(eid,org_id,action,actor_id,encoded,previous,digest,created)); return eid
 def verify(self,org_id):
  with self.repo.transaction() as c:rows=c.execute('SELECT previous_hash,event_hash FROM audit_events WHERE org_id=? ORDER BY created_at',(org_id,)).fetchall()
  return all(row['previous_hash']==('0'*64 if i==0 else rows[i-1]['event_hash']) for i,row in enumerate(rows))

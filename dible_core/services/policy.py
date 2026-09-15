from __future__ import annotations
import json
from dible_core.storage.repository import Repository
class PolicyService:
 def __init__(self,repo:Repository):self.repo=repo
 def allowed(self,org,resource,action,role):
  with self.repo.transaction() as c: rows=c.execute('SELECT roles FROM policies WHERE org_id=? AND resource IN (?,?) AND action=? AND enabled=1',(org,resource,'*',action)).fetchall()
  return any(role in json.loads(row['roles']) for row in rows)

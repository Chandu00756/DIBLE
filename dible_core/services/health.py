from __future__ import annotations
from dible_core.storage.repository import Repository
from dible_core.services.audit import AuditService
class HealthService:
 def __init__(self,repo):self.repo=repo
 def report(self,org=None):
  with self.repo.transaction() as c: tables={t:c.execute(f'SELECT COUNT(*) n FROM {t}').fetchone()['n'] for t in ('organizations','principals','devices','vaults','key_versions','research_runs','alerts','audit_events')}
  return {'status':'ok','persistence':'sqlite','counts':tables,'audit_chain_valid':AuditService(self.repo).verify(org) if org else None,'research_only':True}

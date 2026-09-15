from __future__ import annotations
import json,time
from research.dible_v2 import Parameters,device_commitment,keygen,encapsulate,decapsulate,security_notes
from dible_core.storage.repository import Repository,ident,now
from dible_core.services.audit import AuditService
class ResearchService:
 def __init__(self,repo):self.repo=repo;self.audit=AuditService(repo)
 def simulation(self,org,actor,rounds=100):
  if not 1<=rounds<=10000:raise ValueError('rounds must be 1..10000')
  p=Parameters();c=device_commitment('dible-simulation',b'D'*32);pk,sk=keygen(p,c);started=time.perf_counter();passed=0
  for _ in range(rounds):
   ct,a=encapsulate(p,pk,c);passed+=decapsulate(p,sk,ct,c)==a
  result={'rounds':rounds,'passed':passed,'failed':rounds-passed,'seconds':round(time.perf_counter()-started,6),'parameters':security_notes(p),'research_only':True};rid=ident();self.repo.insert('research_runs',{'id':rid,'org_id':org,'kind':'device_bound_kem_roundtrip','parameters':json.dumps({'rounds':rounds}),'result':json.dumps(result),'status':'complete','created_at':now()});self.audit.record(org,'research.simulation.completed',actor,{'run_id':rid,'rounds':rounds});return {'id':rid,**result}

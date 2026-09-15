from __future__ import annotations
import argparse,json
from dible_core.storage.repository import Repository
from dible_core.services.health import HealthService
from dible_core.services.research import ResearchService
from dible_core.services.identity import IdentityService
from dible_core.services.vault import VaultService
def main():
 p=argparse.ArgumentParser(prog='dible-core');s=p.add_subparsers(dest='command',required=True);s.add_parser('migrate');s.add_parser('health');sim=s.add_parser('simulate');sim.add_argument('--org',required=True);sim.add_argument('--actor',default='local');sim.add_argument('--rounds',type=int,default=100)
 a=p.parse_args();r=Repository();r.migrate()
 if a.command=='migrate':out={'migrated':True}
 elif a.command=='health':out=HealthService(r).report()
 else:out=ResearchService(r).simulation(a.org,a.actor,a.rounds)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()

from __future__ import annotations
import argparse, json, os
from dible_agent.agent import DeviceAgent
from dible_sdk.client import DibleClient
def main():
 p=argparse.ArgumentParser(prog='dible'); p.add_argument('--url',default=os.getenv('DIBLE_URL','http://127.0.0.1:8080')); sub=p.add_subparsers(dest='cmd',required=True)
 b=sub.add_parser('bootstrap'); b.add_argument('--organization',required=True); b.add_argument('--email',required=True); b.add_argument('--password',required=True)
 l=sub.add_parser('login'); l.add_argument('--email',required=True); l.add_argument('--password',required=True)
 e=sub.add_parser('device-enroll'); e.add_argument('--token',required=True); e.add_argument('--name',required=True); e.add_argument('--salt-hex',required=True)
 v=sub.add_parser('vault-create'); v.add_argument('--token',required=True); v.add_argument('--name',required=True)
 a=sub.add_parser('audit'); a.add_argument('--token',required=True)
 args=p.parse_args(); c=DibleClient(args.url,getattr(args,'token',None))
 if args.cmd=='bootstrap': out=c.request('POST','/v1/bootstrap',{'organization':args.organization,'email':args.email,'password':args.password})
 elif args.cmd=='login': out={'access_token':c.login(args.email,args.password)}
 elif args.cmd=='device-enroll': out=DeviceAgent().enroll(c,args.name,args.salt_hex)
 elif args.cmd=='vault-create': out=c.request('POST','/v1/vaults',{'name':args.name})
 else: out=c.request('GET','/v1/audit')
 print(json.dumps(out,indent=2))
if __name__=='__main__': main()

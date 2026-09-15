from __future__ import annotations
import json, urllib.request
class DibleClient:
 def __init__(self, base_url='http://127.0.0.1:8080', token=None): self.base_url=base_url.rstrip('/'); self.token=token
 def request(self, method,path,data=None):
  body=json.dumps(data).encode() if data is not None else None; headers={'Content-Type':'application/json'}
  if self.token: headers['Authorization']='Bearer '+self.token
  req=urllib.request.Request(self.base_url+path,data=body,headers=headers,method=method)
  with urllib.request.urlopen(req,timeout=15) as response:return json.load(response)
 def login(self,email,password): self.token=self.request('POST','/v1/auth/login',{'email':email,'password':password})['access_token']; return self.token

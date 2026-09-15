from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from dible_core.config.settings import Settings
from dible_core.storage.repository import Repository
from dible_core.services.health import HealthService
class Handler(BaseHTTPRequestHandler):
 repo=Repository(Settings())
 def log_message(self,*args):pass
 def send(self,status,payload):
  raw=json.dumps(payload).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def do_GET(self):
  if self.path=='/health':return self.send(200,HealthService(self.repo).report())
  if self.path=='/openapi.json':return self.send(200,{'openapi':'3.1.0','info':{'title':'DIBLE Control Plane','version':'0.1.0','description':'Experimental research system'},'paths':{'/health':{'get':{'responses':{'200':{'description':'health'}}}}}})
  self.send(404,{'error':'not found'})
def serve(host='127.0.0.1',port=8080):
 Handler.repo.migrate();ThreadingHTTPServer((host,port),Handler).serve_forever()
if __name__=='__main__':serve()

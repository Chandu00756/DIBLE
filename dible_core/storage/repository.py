from __future__ import annotations
import json, secrets, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from dible_core.config.settings import Settings

def now(): return datetime.now(timezone.utc).isoformat()
def ident(): return secrets.token_hex(16)
class Repository:
 def __init__(self,settings:Settings=Settings()): self.settings=settings
 @contextmanager
 def transaction(self):
  path=self.settings.sqlite_path; path.parent.mkdir(parents=True,exist_ok=True); c=sqlite3.connect(path); c.row_factory=sqlite3.Row
  try: yield c; c.commit()
  except: c.rollback(); raise
  finally:c.close()
 def migrate(self):
  with self.transaction() as c:c.executescript('''PRAGMA foreign_keys=ON;
 CREATE TABLE IF NOT EXISTS organizations(id TEXT PRIMARY KEY,name TEXT UNIQUE NOT NULL,created_at TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS principals(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),subject TEXT NOT NULL,role TEXT NOT NULL,secret_hash TEXT NOT NULL,disabled_at TEXT,created_at TEXT NOT NULL,UNIQUE(org_id,subject));
 CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,principal_id TEXT NOT NULL REFERENCES principals(id),token_hash TEXT UNIQUE NOT NULL,expires_at TEXT NOT NULL,created_at TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),label TEXT NOT NULL,commitment TEXT NOT NULL,status TEXT NOT NULL,metadata TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(org_id,commitment));
 CREATE TABLE IF NOT EXISTS vaults(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),name TEXT NOT NULL,classification TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(org_id,name));
 CREATE TABLE IF NOT EXISTS key_versions(id TEXT PRIMARY KEY,vault_id TEXT NOT NULL REFERENCES vaults(id),version INTEGER NOT NULL,status TEXT NOT NULL,algorithm TEXT NOT NULL,public_material TEXT NOT NULL,device_id TEXT REFERENCES devices(id),created_at TEXT NOT NULL,retired_at TEXT,UNIQUE(vault_id,version));
 CREATE TABLE IF NOT EXISTS policies(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),resource TEXT NOT NULL,action TEXT NOT NULL,roles TEXT NOT NULL,enabled INTEGER NOT NULL,created_at TEXT NOT NULL,UNIQUE(org_id,resource,action));
 CREATE TABLE IF NOT EXISTS research_runs(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),kind TEXT NOT NULL,parameters TEXT NOT NULL,result TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS alerts(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),severity TEXT NOT NULL,title TEXT NOT NULL,details TEXT NOT NULL,state TEXT NOT NULL,created_at TEXT NOT NULL,resolved_at TEXT);
 CREATE TABLE IF NOT EXISTS audit_events(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),action TEXT NOT NULL,actor_id TEXT NOT NULL,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT NOT NULL,created_at TEXT NOT NULL);
 ''')
 def insert(self,table:str,values:dict):
  cols=','.join(values); marks=','.join('?' for _ in values)
  with self.transaction() as c:c.execute(f'INSERT INTO {table} ({cols}) VALUES ({marks})',tuple(values.values()))
 def list(self,table:str,org_id:str,limit=50,offset=0):
  with self.transaction() as c:r=c.execute(f'SELECT * FROM {table} WHERE org_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?',(org_id,limit,offset)).fetchall()
  return [dict(x) for x in r]

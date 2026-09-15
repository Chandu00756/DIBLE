from __future__ import annotations
import hashlib, json, os, secrets, sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = os.getenv('DIBLE_DB_PATH', './data/dible.db')
def now() -> str: return datetime.now(timezone.utc).isoformat()
def h(value: str) -> str: return hashlib.sha3_256(value.encode()).hexdigest()
class Database:
 def __init__(self, path: str=DEFAULT_DB): self.path=path
 @contextmanager
 def connect(self):
  Path(self.path).parent.mkdir(parents=True, exist_ok=True); con=sqlite3.connect(self.path); con.row_factory=sqlite3.Row
  try: yield con; con.commit()
  except: con.rollback(); raise
  finally: con.close()
 def migrate(self):
  with self.connect() as c: c.executescript('''
   PRAGMA foreign_keys=ON;
   CREATE TABLE IF NOT EXISTS organizations(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL);
   CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),email TEXT NOT NULL,role TEXT NOT NULL,password_hash TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(org_id,email));
   CREATE TABLE IF NOT EXISTS tokens(token_hash TEXT PRIMARY KEY,user_id TEXT NOT NULL REFERENCES users(id),expires_at TEXT NOT NULL,created_at TEXT NOT NULL);
   CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),name TEXT NOT NULL,commitment TEXT NOT NULL,state TEXT NOT NULL,created_at TEXT NOT NULL,revoked_at TEXT);
   CREATE TABLE IF NOT EXISTS vaults(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),name TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(org_id,name));
   CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),vault_id TEXT REFERENCES vaults(id),kind TEXT NOT NULL,body TEXT NOT NULL,device_id TEXT REFERENCES devices(id),created_at TEXT NOT NULL);
   CREATE TABLE IF NOT EXISTS policies(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),action TEXT NOT NULL,roles TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(org_id,action));
   CREATE TABLE IF NOT EXISTS audit_events(id TEXT PRIMARY KEY,org_id TEXT NOT NULL REFERENCES organizations(id),action TEXT NOT NULL,actor TEXT NOT NULL,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT NOT NULL,created_at TEXT NOT NULL);
  ''')
 def audit(self, org: str, action: str, actor: str, payload: dict):
  with self.connect() as c:
   previous=c.execute('SELECT event_hash FROM audit_events WHERE org_id=? ORDER BY created_at DESC LIMIT 1',(org,)).fetchone(); prev=previous['event_hash'] if previous else '0'*64
   eid=secrets.token_hex(16); created=now(); encoded=json.dumps(payload,sort_keys=True,separators=(',',':')); event=h('|'.join((eid,org,action,actor,encoded,prev,created)))
   c.execute('INSERT INTO audit_events VALUES(?,?,?,?,?,?,?,?)',(eid,org,action,actor,encoded,prev,event,created)); return eid

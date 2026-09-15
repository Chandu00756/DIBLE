from __future__ import annotations
import json, os, secrets
from functools import wraps
from flask import Flask, g, jsonify, request
from dible_server.db import Database, h, now
from dible_server.security import expiry, password_hash, token, verify_password

def create_app(database: Database|None=None):
 app=Flask(__name__); db=database or Database(); db.migrate(); app.config['JSON_SORT_KEYS']=False
 def fail(msg,code=400): return jsonify({'error':msg}),code
 def body(*keys):
  data=request.get_json(silent=True) or {}; missing=[k for k in keys if not data.get(k)]
  if missing: raise ValueError('missing: '+', '.join(missing))
  return data
 def auth(role: str|None=None):
  def dec(fn):
   @wraps(fn)
   def wrapped(*a,**kw):
    raw=request.headers.get('Authorization',''); value=raw.removeprefix('Bearer ').strip()
    with db.connect() as c: row=c.execute('SELECT u.* FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token_hash=? AND t.expires_at>?',(h(value),now())).fetchone()
    if not row: return fail('authentication required',401)
    if role and row['role']!=role: return fail('insufficient role',403)
    g.user=dict(row); return fn(*a,**kw)
   return wrapped
  return dec
 @app.errorhandler(ValueError)
 def validation(e): return fail(str(e))
 @app.get('/health')
 def health(): return jsonify({'status':'ok','service':'dible-control-plane','research_only':True})
 @app.post('/v1/bootstrap')
 def bootstrap():
  data=body('organization','email','password'); oid=secrets.token_hex(12); uid=secrets.token_hex(12)
  with db.connect() as c:
   if c.execute('SELECT 1 FROM users LIMIT 1').fetchone(): return fail('already bootstrapped',409)
   c.execute('INSERT INTO organizations VALUES(?,?,?)',(oid,data['organization'],now())); c.execute('INSERT INTO users VALUES(?,?,?,?,?,?)',(uid,oid,data['email'].lower(),'admin',password_hash(data['password']),now()))
  db.audit(oid,'organization.bootstrapped',uid,{'organization':data['organization']}); return jsonify({'organization_id':oid,'user_id':uid}),201
 @app.post('/v1/auth/login')
 def login():
  data=body('email','password')
  with db.connect() as c: user=c.execute('SELECT * FROM users WHERE email=?',(data['email'].lower(),)).fetchone()
  if not user or not verify_password(data['password'],user['password_hash']): return fail('invalid credentials',401)
  raw=token()
  with db.connect() as c:c.execute('INSERT INTO tokens VALUES(?,?,?,?)',(h(raw),user['id'],expiry(),now()))
  db.audit(user['org_id'],'auth.login',user['id'],{}); return jsonify({'access_token':raw,'token_type':'bearer','expires_at':expiry()})
 @app.get('/v1/devices')
 @auth()
 def devices():
  with db.connect() as c: rows=c.execute('SELECT id,name,state,created_at,revoked_at FROM devices WHERE org_id=? ORDER BY created_at DESC',(g.user['org_id'],)).fetchall()
  return jsonify({'items':[dict(x) for x in rows]})
 @app.post('/v1/devices')
 @auth()
 def enroll():
  data=body('name','commitment'); did=secrets.token_hex(12)
  with db.connect() as c:c.execute('INSERT INTO devices VALUES(?,?,?,?,?,?,?)',(did,g.user['org_id'],data['name'],data['commitment'],'enrolled',now(),None))
  db.audit(g.user['org_id'],'device.enrolled',g.user['id'],{'device_id':did}); return jsonify({'id':did,'state':'enrolled'}),201
 @app.post('/v1/devices/<device_id>/revoke')
 @auth('admin')
 def revoke(device_id):
  with db.connect() as c: changed=c.execute("UPDATE devices SET state='revoked',revoked_at=? WHERE id=? AND org_id=? AND state='enrolled'",(now(),device_id,g.user['org_id'])).rowcount
  if not changed:return fail('active device not found',404)
  db.audit(g.user['org_id'],'device.revoked',g.user['id'],{'device_id':device_id}); return jsonify({'id':device_id,'state':'revoked'})
 @app.route('/v1/vaults',methods=['GET','POST'])
 @auth()
 def vaults():
  if request.method=='GET':
   with db.connect() as c:r=c.execute('SELECT * FROM vaults WHERE org_id=? ORDER BY created_at DESC',(g.user['org_id'],)).fetchall()
   return jsonify({'items':[dict(x) for x in r]})
  data=body('name'); vid=secrets.token_hex(12)
  with db.connect() as c:c.execute('INSERT INTO vaults VALUES(?,?,?,?)',(vid,g.user['org_id'],data['name'],now()))
  db.audit(g.user['org_id'],'vault.created',g.user['id'],{'vault_id':vid}); return jsonify({'id':vid,'name':data['name']}),201
 @app.post('/v1/artifacts')
 @auth()
 def artifact():
  data=body('kind','body'); aid=secrets.token_hex(12)
  with db.connect() as c:c.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',(aid,g.user['org_id'],data.get('vault_id'),data['kind'],json.dumps(data['body']),data.get('device_id'),now()))
  db.audit(g.user['org_id'],'artifact.created',g.user['id'],{'artifact_id':aid,'kind':data['kind']}); return jsonify({'id':aid}),201
 @app.get('/v1/audit')
 @auth()
 def audit():
  limit=min(max(int(request.args.get('limit',50)),1),200)
  with db.connect() as c:r=c.execute('SELECT id,action,actor,payload,event_hash,created_at FROM audit_events WHERE org_id=? ORDER BY created_at DESC LIMIT ?',(g.user['org_id'],limit)).fetchall()
  return jsonify({'items':[dict(x) for x in r]})
 return app
if __name__=='__main__': create_app().run(host='0.0.0.0',port=int(os.getenv('PORT','8080')))

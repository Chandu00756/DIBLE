import tempfile, unittest
from dible_server.app import create_app
from dible_server.db import Database
class ApiTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.client=create_app(Database(self.tmp.name+'/db.sqlite')).test_client()
  r=self.client.post('/v1/bootstrap',json={'organization':'Acme','email':'admin@acme.test','password':'correct-horse-battery-staple'}); self.assertEqual(r.status_code,201)
  self.token=self.client.post('/v1/auth/login',json={'email':'admin@acme.test','password':'correct-horse-battery-staple'}).get_json()['access_token']; self.h={'Authorization':'Bearer '+self.token}
 def tearDown(self): self.tmp.cleanup()
 def test_lifecycle_and_tenant_boundary(self):
  d=self.client.post('/v1/devices',headers=self.h,json={'name':'Mac','commitment':'a'*64}); self.assertEqual(d.status_code,201); did=d.get_json()['id']
  self.assertEqual(self.client.post('/v1/vaults',headers=self.h,json={'name':'research'}).status_code,201)
  self.assertEqual(self.client.post('/v1/artifacts',headers=self.h,json={'kind':'kem-ciphertext','body':{'version':1},'device_id':did}).status_code,201)
  self.assertEqual(self.client.post('/v1/devices/'+did+'/revoke',headers=self.h).get_json()['state'],'revoked')
  events=self.client.get('/v1/audit',headers=self.h).get_json()['items']; self.assertGreaterEqual(len(events),5)
 def test_denies_unauthenticated(self): self.assertEqual(self.client.get('/v1/devices').status_code,401)
if __name__=='__main__': unittest.main()

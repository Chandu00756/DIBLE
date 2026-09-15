import tempfile,unittest
from dible_core.config.settings import Settings
from dible_core.storage.repository import Repository
from dible_core.services.identity import IdentityService
from dible_core.services.vault import VaultService
from dible_core.services.policy import PolicyService
from dible_core.services.research import ResearchService
class CoreTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.r=Repository(Settings('sqlite:///'+self.tmp.name+'/d.db'));self.r.migrate();self.r.insert('organizations',{'id':'org','name':'test','created_at':'now'})
 def tearDown(self):self.tmp.cleanup()
 def test_device_vault_key_and_research_lifecycle(self):
  i=IdentityService(self.r);d=i.enroll('org','actor','mac','claim',b'x'*32);v=VaultService(self.r).create('org','actor','vault');k=VaultService(self.r).rotate_key('org','actor',v,'DIBLE-RESEARCH',{'x':1},d['id']);self.assertEqual(k['version'],1);self.assertEqual(ResearchService(self.r).simulation('org','actor',5)['failed'],0);i.revoke('org','actor',d['id']);self.assertRaises(PermissionError,i.active,'org',d['id'])
if __name__=='__main__':unittest.main()

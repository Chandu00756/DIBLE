import unittest
from dible_platform.api.app import ControlPlane
from dible_platform.chaos.maps import LogisticMap
from dible_platform.algebra.polynomial import Polynomial
from dible_platform.policy.engine import AccessPolicy, PolicyEngine
from dible_platform.simulation.runner import Simulation
class PlatformTests(unittest.TestCase):
 def test_control_plane_end_to_end(self):
  p=ControlPlane(); d,s=p.enroll('test-mac',b'x'*32); self.assertTrue(p.protocol.validate_device_bound_exchange(d.device_id,'test-mac',s)); self.assertTrue(p.health()['chain_valid']); p.devices.revoke(d.device_id); self.assertRaises(PermissionError,p.devices.require_active,d.device_id)
 def test_math_and_policy(self):
  self.assertEqual(Polynomial((1,2,3),17).evaluate(2),0); self.assertEqual(len(LogisticMap().sequence(.2,5)),5); self.assertTrue(PolicyEngine([AccessPolicy('key.use',frozenset({'admin'}))]).authorize('key.use',{'admin'}))
 def test_simulation(self): self.assertEqual(Simulation().run(20)['failed'],0)
if __name__=='__main__': unittest.main()

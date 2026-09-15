from __future__ import annotations
import hashlib, platform, uuid
from dible_sdk.client import DibleClient
class DeviceAgent:
 def commitment(self, salt: str) -> str:
  claim='|'.join((platform.node(),platform.machine(),str(uuid.getnode()))); return hashlib.sha3_256(bytes.fromhex(salt)+claim.encode()).hexdigest()
 def enroll(self, client:DibleClient, name:str, salt:str): return client.request('POST','/v1/devices',{'name':name,'commitment':self.commitment(salt)})

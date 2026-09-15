from __future__ import annotations
import base64, hashlib, hmac, os, secrets
from datetime import datetime, timedelta, timezone

def password_hash(password: str, salt: bytes|None=None) -> str:
 if len(password)<12: raise ValueError('password must be at least 12 characters')
 salt=salt or secrets.token_bytes(16); out=hashlib.scrypt(password.encode(),salt=salt,n=2**14,r=8,p=1,dklen=32); return base64.b64encode(salt+out).decode()
def verify_password(password: str, stored: str)->bool:
 raw=base64.b64decode(stored); return hmac.compare_digest(password_hash(password,raw[:16]),stored)
def token()->str: return secrets.token_urlsafe(48)
def expiry(hours:int=24)->str: return (datetime.now(timezone.utc)+timedelta(hours=hours)).isoformat()

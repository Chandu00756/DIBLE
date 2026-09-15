#!/usr/bin/env python3
"""PORTAL VII DIBLE – automated verification (no interactivity)."""
from __future__ import annotations
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

PASS, FAIL = [], []

def ok(msg): PASS.append(msg); print(f"  ✅ {msg}")
def fail(msg, err): FAIL.append(msg); print(f"  ❌ {msg}: {err}")

# 1. Device ID
try:
    from src.core.device_id import DeviceIDGenerator
    did = DeviceIDGenerator().generate_device_id()
    assert len(did) == 64
    ok(f"Device ID: {did[:16]}...")
except Exception as e: fail("Device ID", e)

# 2. Entropy
try:
    from src.core.entropy import EntropyManager
    seed = int.from_bytes(hashlib.sha256(did.encode()).digest()[:4], "big")
    em = EntropyManager(device_id_transform=seed)
    pool = em.collect(64)
    assert len(pool) == 64
    ok(f"Entropy pool: {len(pool)} bytes")
except Exception as e: fail("Entropy", e)

# 3. Lattice
try:
    from src.core.lattice import LatticeOperations
    lwe = LatticeOperations(dimension=32, modulus=2**10)
    kp = lwe.generate_keypair()
    ct = lwe.encrypt_vector(kp["public"], 1)
    bit = lwe.decrypt_vector(kp["private"], ct)
    assert bit == 1
    ok("LWE encrypt/decrypt bit=1")
except Exception as e: fail("Lattice", e)

# 4. Chaos
try:
    from src.core.chaos import ChaosTheoryManager
    seq = ChaosTheoryManager().generate_chaotic_sequence(32)
    assert len(seq) == 32 and all(0 < v < 1 for v in seq)
    ok(f"Chaos sequence: {len(seq)} values")
except Exception as e: fail("Chaos", e)

# 5. Polynomial
try:
    from src.core.polynomial import PolynomialOperations
    poly = PolynomialOperations(modulus=97)
    p = poly.random_polynomial()
    assert len(p["coefficients"]) > 0
    ok(f"Polynomial: {len(p['coefficients'])} coefficients")
except Exception as e: fail("Polynomial", e)

# 6. DIBLECore round-trip
try:
    from src.crypto.dible import DIBLECore
    core = DIBLECore(did)
    key_obj = core.generate_key()
    pt = b"PORTAL VII DIBLE self-test 2026"
    ct = core.encrypt(pt, key_obj.raw_key)
    assert core.decrypt(ct, key_obj.raw_key) == pt
    ok("AES-256-GCM raw-key round-trip")
except Exception as e: fail("DIBLECore raw-key", e)

# 7. PBKDF2 round-trip
try:
    import secrets as _s
    core2 = DIBLECore(did)
    pw = _s.token_urlsafe(16)
    k, salt = core2.derive_key(pw)
    ct2 = core2.encrypt(pt, k)
    k2, _ = core2.derive_key(pw, salt)
    assert core2.decrypt(ct2, k2) == pt
    ok("PBKDF2 KDF round-trip")
except Exception as e: fail("PBKDF2", e)

# 8. Tamper detection
try:
    bad = bytearray(ct); bad[-1] ^= 0xFF
    try: core.decrypt(bytes(bad), key_obj.raw_key); fail("Tamper detect", "no exception")
    except Exception: ok("Tamper detection (GCM auth)")
except Exception as e: fail("Tamper", e)

# 9. Post-quantum KEM stub
try:
    from src.quantum.post_quantum import PostQuantumKEM
    kem = PostQuantumKEM(dimension=32)
    kp2 = kem.keygen()
    ss1, ctx = kem.encapsulate(kp2)
    ss2 = kem.decapsulate(kp2, ctx)
    assert ss1 == ss2
    ok("Post-quantum KEM shared-secret agreement")
except Exception as e: fail("Post-quantum KEM", e)

# summary
print()
print(f"Passed: {len(PASS)}  Failed: {len(FAIL)}")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("✅ All DIBLE verification tests passed.")

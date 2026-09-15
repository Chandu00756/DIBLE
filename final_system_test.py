#!/usr/bin/env python3
"""PORTAL VII DIBLE – automated verification suite (CI-safe, no interactivity)."""
from __future__ import annotations
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS_LIST, FAIL_LIST = [], []


def ok(msg: str) -> None:
    PASS_LIST.append(msg)
    print(f"  \u2705 {msg}")


def fail(name: str, err: object) -> None:
    FAIL_LIST.append(name)
    print(f"  \u274c {name}: {err}")


# 1. Device ID
try:
    from src.core.device_id import DeviceIDGenerator
    did = DeviceIDGenerator().generate_device_id()
    assert len(did) == 64, f"expected 64 hex chars, got {len(did)}"
    ok(f"Device ID: {did[:16]}...")
except Exception as e:
    fail("Device ID", e)
    did = hashlib.sha256(b"fallback").hexdigest()

# 2. Entropy
try:
    from src.core.entropy import EntropyManager
    seed = int.from_bytes(hashlib.sha256(did.encode()).digest()[:4], "big")
    em = EntropyManager(device_id_transform=seed)
    pool = em.collect(64)
    assert len(pool) == 64
    ok(f"Entropy pool: {len(pool)} bytes")
except Exception as e:
    fail("Entropy", e)

# 3. Lattice LWE
try:
    from src.core.lattice import LatticeOperations
    lwe = LatticeOperations(dimension=32, modulus=2 ** 10)
    kp = lwe.generate_keypair()
    ct_one = lwe.encrypt_vector(kp["public"], 1)
    ct_zero = lwe.encrypt_vector(kp["public"], 0)
    assert lwe.decrypt_vector(kp["private"], ct_one) == 1
    assert lwe.decrypt_vector(kp["private"], ct_zero) == 0
    ok("LWE encrypt/decrypt bit=0 and bit=1")
except Exception as e:
    fail("Lattice", e)

# 4. Chaos
try:
    from src.core.chaos import ChaosTheoryManager
    seq = ChaosTheoryManager().generate_chaotic_sequence(32)
    assert len(seq) == 32
    assert all(0.0 < v < 1.0 for v in seq)
    ok(f"Chaos sequence: {len(seq)} values in (0,1)")
except Exception as e:
    fail("Chaos", e)

# 5. Polynomial
try:
    from src.core.polynomial import PolynomialOperations
    poly = PolynomialOperations(modulus=97)
    p = poly.random_polynomial()
    assert len(p["coefficients"]) > 0
    ok(f"Polynomial: {len(p['coefficients'])} coefficients")
except Exception as e:
    fail("Polynomial", e)

# 6. DIBLECore AES-256-GCM raw-key round-trip
try:
    from src.crypto.dible import DIBLECore
    core = DIBLECore(did)
    key_obj = core.generate_key()
    pt = b"PORTAL VII DIBLE self-test 2026"
    ct = core.encrypt(pt, key_obj.raw_key)
    recovered = core.decrypt(ct, key_obj.raw_key)
    assert recovered == pt
    ok("AES-256-GCM raw-key round-trip")
except Exception as e:
    fail("DIBLECore raw-key", e)
    key_obj = None
    ct = None

# 7. PBKDF2 round-trip
try:
    import secrets as _s
    core2 = DIBLECore(did)
    pw = _s.token_urlsafe(16)
    k, salt = core2.derive_key(pw)
    ct2 = core2.encrypt(pt, k)
    k2, _ = core2.derive_key(pw, salt)
    assert core2.decrypt(ct2, k2) == pt
    ok("PBKDF2-SHA256 / 600k iters round-trip")
except Exception as e:
    fail("PBKDF2", e)

# 8. Tamper detection
try:
    if ct and key_obj:
        bad = bytearray(ct)
        bad[-1] ^= 0xFF
        raised = False
        try:
            core.decrypt(bytes(bad), key_obj.raw_key)
        except Exception:
            raised = True
        assert raised, "no exception on tampered ciphertext"
        ok("Tamper detection (GCM auth tag)")
    else:
        fail("Tamper detection", "skipped (no ciphertext)")
except Exception as e:
    fail("Tamper detection", e)

# 9. Post-quantum KEM
try:
    from src.quantum.post_quantum import PostQuantumKEM
    kem = PostQuantumKEM(dimension=32)
    kp2 = kem.keygen()
    ss1, ctx = kem.encapsulate(kp2)
    ss2 = kem.decapsulate(kp2, ctx)
    assert ss1 == ss2
    ok("Post-quantum KEM shared-secret agreement")
except Exception as e:
    fail("Post-quantum KEM", e)

# ---------------------------------------------------------------------------
print()
print(f"Passed: {len(PASS_LIST)}   Failed: {len(FAIL_LIST)}")
if FAIL_LIST:
    print("FAILED tests:", FAIL_LIST)
    sys.exit(1)
print("\u2705 All DIBLE verification tests passed.")

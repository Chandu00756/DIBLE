# DIBLE Cryptography Research Track

DIBLE v2 is a fresh device-bound lattice KEM research construction. It does **not** use the vault's AES/PBKDF implementation, and it is deliberately isolated from every encryption, secrets-management, file-protection, API, and commercial workflow.

## Non-negotiable research boundary

This construction has no security proof, parameter justification, attack analysis, constant-time implementation, ciphertext validation proof, side-channel analysis, formal review, or independent audit. Consequently, it is **not quantum resistant**, cannot be assigned a security-bit estimate, and must never protect user data or be advertised as production cryptography.

## What exists now

- A dependency-free, readable reference implementation of a device-bound LWE-style KEM experiment
- Explicit parameter object and serializable research artifacts
- Non-reversible salted device commitments, never raw hardware identifiers
- Device-commitment checks at key generation, encapsulation, and decapsulation
- Repeatable round-trip and cross-device rejection tests

## Research program before any security claim

1. Define the exact hardness assumption and adversary model.
2. Derive parameters from a public reduction or conservative estimator.
3. Specify canonical encodings, domain separation, rejection behavior, and CCA transform.
4. Implement deterministic known-answer tests, fuzzing, statistical tests, and differential tests.
5. Commission external cryptographic review and publish a design paper and attack bounty.
6. Only then evaluate a separate, opt-in experimental release. Do not merge it into the vault crypto path.

## Run

```bash
python -m pytest -q tests/test_dible_v2.py
python - <<'PY'
from research.dible_v2 import Parameters, security_notes
print(security_notes(Parameters()))
PY
```

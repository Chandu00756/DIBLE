# DIBLE Launch Readiness Gate

## Current decision: do not launch as a security product

DIBLE Lab is a working research toolkit, not a production cryptographic product. The new device-bound lattice KEM is original experimental code with no formal proof, standardized parameter selection, CCA security construction, constant-time implementation, side-channel resistance, or independent audit. It cannot be marketed as secure, post-quantum, or quantum resistant.

## End-to-end capabilities included

- Local key generation, encapsulation, decapsulation, artifact validation, permission-restricted artifact writes, and device-binding checks
- Standard-library automated tests plus a randomized verification command
- Versioned JSON artifact codecs and a package console command
- Explicit research status included in every command output

## Mandatory release gates

1. Independent cryptographic design review and published specification.
2. Conservative parameters supported by a public attack estimator and documented failure probabilities.
3. IND-CCA security construction and test vectors.
4. Constant-time, memory-safe implementation; no Python reference code in any security boundary.
5. Fuzzing, malformed-input, timing, fault-injection, interoperability, and regression test coverage.
6. Third-party application-security audit, SBOM, signed builds, security.txt, disclosure policy, and incident response runbook.
7. Closed external beta with non-sensitive synthetic data only.

Until every gate is complete, DIBLE may be launched only as open-source cryptography research software.

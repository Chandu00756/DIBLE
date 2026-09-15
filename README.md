
# 🛡️ PORTAL VII DIBLE

![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-AES--256--GCM-green.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

> **Experimental device-bound encryption vault.**  
> Not security-audited. Do not use to protect sensitive production data.

---

## What It Does

PORTAL VII DIBLE is a local file-encryption vault with:

- **AES-256-GCM** authenticated encryption
- **Device-bound keys** – device fingerprint is mixed into every PBKDF2 derivation and used as GCM AAD
- **PBKDF2-SHA256 / 600 000 iterations** for password-derived keys
- **Audit log** – every operation is appended to `~/.portal7_dible/logs/audit.log`
- **Rich CLI** – non-interactive, scriptable, CI-friendly

---

## Install

```bash
git clone https://github.com/Chandu00756/DIBLE.git && cd DIBLE
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

---

## Usage

```bash
# Show vault status
dible-vault status
dible-vault status --json          # machine-readable

# Generate a key
dible-vault keygen
dible-vault keygen --name mykey

# List stored keys
dible-vault list-keys

# Encrypt a file
dible-vault encrypt -i secret.txt
dible-vault encrypt -i secret.txt -k <key_id>
dible-vault encrypt -i secret.txt -p mysecretpassword

# Decrypt a file
dible-vault decrypt -i ~/.portal7_dible/data/secret.txt.dible
dible-vault decrypt -i file.dible -o out.txt -p mysecretpassword

# Encrypt / decrypt inline text
dible-vault encrypt-text "hello world" -p mypassword
dible-vault decrypt-text <base64> -p mypassword -s <salt_hex>

# View audit log
dible-vault audit
dible-vault audit -n 50

# Delete a key (irreversible)
dible-vault delete-key <key_id> --confirm

# Run self-tests
dible-vault test
python final_system_test.py
```

---

## Architecture

```
DIBLE/
├── dible_vault_cli.py          ← CLI entry point (dible-vault)
├── src/
│   ├── core/
│   │   ├── device_id.py          SHA3-256 hardware fingerprint
│   │   ├── entropy.py            OS CSPRNG entropy pool
│   │   ├── lattice.py            LWE key operations
│   │   ├── chaos.py              Chaos theory entropy helpers
│   │   └── polynomial.py         Polynomial ring operations
│   ├── crypto/
│   │   ├── dible.py              AES-256-GCM core + PBKDF2 KDF
│   │   ├── homomorphic.py        Additive HE stub
│   │   ├── encryption.py         Re-export helper
│   │   └── decryption.py         Re-export helper
│   ├── quantum/
│   │   ├── post_quantum.py       LWE-based KEM stub
│   │   └── quantum_ops.py        Quantum-inspired helpers
│   └── utils/
│       ├── hash_functions.py     SHA3 / BLAKE2 helpers
│       ├── math_utils.py         Modular arithmetic
│       └── tensor_ops.py         NumPy lattice helpers
├── final_system_test.py        CI verification (9 tests)
└── requirements.txt
```

---

## ⚠️ Security Notice

This is experimental research software. Custom lattice, chaos, quantum, and homomorphic components are **not** standardised post-quantum cryptosystems and have not been independently audited.
